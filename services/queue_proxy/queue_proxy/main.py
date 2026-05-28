from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from queue_proxy.backend_registry import BackendRegistryClient
from queue_proxy.limiter import LimiterRegistry, QueueFull, QueueTimeout
from queue_proxy.metrics import (
    CONTENT_TYPE_LATEST,
    ERRORS,
    INPUT_TOKENS,
    LATENCY,
    OUTPUT_TOKEN_BUDGET,
    REQUESTS,
    generate_latest,
    record_snapshot,
)
from queue_proxy.policy import (
    PolicyError,
    PolicyRegistry,
    apply_token_policy,
    extract_model,
    load_policy_registry,
    strip_internal_fields,
)
from queue_proxy.settings import Settings

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


settings = Settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

policy_registry: PolicyRegistry = load_policy_registry(settings.config_path)
limiter_registry = LimiterRegistry()
backend_registry_client = (
    BackendRegistryClient(settings.backend_registry_url, settings.request_timeout_seconds)
    if settings.backend_registry_url
    else None
)
app = FastAPI(title="local-llm-orchestrator queue proxy", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"service": settings.service_name, "status": "healthy"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    return {
        "service": settings.service_name,
        "status": "healthy",
        "upstream_base_url": settings.upstream_base_url,
        "backend_registry_url": settings.backend_registry_url,
        "backend_registry_routing": settings.enable_backend_registry_routing,
        "models": sorted(policy_registry.policies.keys()),
    }


@app.get("/status")
async def proxy_status() -> dict[str, Any]:
    snapshots = []
    for snapshot in limiter_registry.snapshots():
        record_snapshot(snapshot.model, snapshot.active_requests, snapshot.queued_requests)
        snapshots.append(snapshot.__dict__)
    return {"models": snapshots}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def forward_openai_path(path: str, request: Request) -> Response:
    if settings.queue_proxy_api_key:
        auth_result = validate_proxy_auth(request)
        if auth_result is not None:
            return auth_result

    body = await request.body()
    payload, policy_metadata = prepare_payload(path, request, body)

    if payload is None:
        return await forward_without_limiter(path, request, body)

    policy = policy_registry.resolve(extract_model(payload))
    limiter = limiter_registry.get_or_create(
        policy.public_name,
        policy.max_active_requests,
        policy.max_queued_requests,
        policy.queue_timeout_seconds,
    )

    started_at = perf_counter()
    endpoint = f"/v1/{path}"

    try:
        await limiter.acquire()
    except QueueFull:
        ERRORS.labels(model=policy.public_name, error_type="queue_full").inc()
        return error_response(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "queue_full",
            "Too many queued requests for this model.",
        )
    except QueueTimeout:
        ERRORS.labels(model=policy.public_name, error_type="queue_timeout").inc()
        return error_response(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "queue_timeout",
            "Timed out waiting for an available model request slot.",
        )

    record_limiter_snapshot(limiter)
    backend_instance_id: str | None = None

    try:
        upstream_base_url, backend_instance_id = await resolve_upstream(policy.public_name)
        if upstream_base_url is None:
            await limiter.release()
            record_limiter_snapshot(limiter)
            ERRORS.labels(model=policy.public_name, error_type="no_ready_backend").inc()
            return error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "no_ready_backend",
                "No ready backend instance is available for this model.",
            )

        clean_payload = strip_internal_fields(payload)
        clean_body = json.dumps(clean_payload, separators=(",", ":")).encode("utf-8")
        response = await stream_upstream_response(
            path=path,
            request=request,
            body=clean_body,
            model=policy.public_name,
            endpoint=endpoint,
            limiter=limiter,
            started_at=started_at,
            upstream_base_url=upstream_base_url,
            backend_instance_id=backend_instance_id,
        )

        if policy_metadata:
            input_tokens = int(policy_metadata.get("estimated_input_tokens", 0))
            output_tokens = int(policy_metadata.get("effective_output_tokens", 0))
            if input_tokens:
                INPUT_TOKENS.labels(model=policy.public_name).inc(input_tokens)
            if output_tokens:
                OUTPUT_TOKEN_BUDGET.labels(model=policy.public_name).inc(output_tokens)
            if policy_metadata.get("output_tokens_capped"):
                response.headers["x-llm-output-tokens-capped"] = "true"

        return response
    except httpx.HTTPError as exc:
        await release_backend_lease(backend_instance_id)
        await limiter.release()
        record_limiter_snapshot(limiter)
        ERRORS.labels(model=policy.public_name, error_type=type(exc).__name__).inc()
        REQUESTS.labels(model=policy.public_name, endpoint=endpoint, status="502").inc()
        LATENCY.labels(model=policy.public_name, endpoint=endpoint).observe(
            perf_counter() - started_at
        )
        logger.warning("upstream_request_failed error_type=%s", type(exc).__name__)
        return error_response(
            status.HTTP_502_BAD_GATEWAY,
            "upstream_request_failed",
            "Upstream LLM gateway request failed.",
        )


def validate_proxy_auth(request: Request) -> JSONResponse | None:
    authorization = request.headers.get("authorization", "")
    expected = f"Bearer {settings.queue_proxy_api_key}"
    if authorization != expected:
        return error_response(
            status.HTTP_401_UNAUTHORIZED,
            "unauthorized",
            "Missing or invalid queue proxy API key.",
        )
    return None


def prepare_payload(
    path: str,
    request: Request,
    body: bytes,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if request.method.upper() != "POST":
        return None, {}

    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        return None, {}

    if not is_llm_generation_endpoint(path):
        return None, {}

    try:
        raw_payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise PolicyError(f"Invalid JSON request body: {exc.msg}", "invalid_json") from exc

    if not isinstance(raw_payload, dict):
        raise PolicyError("JSON request body must be an object.", "invalid_json")

    policy = policy_registry.resolve(extract_model(raw_payload))
    payload = apply_token_policy(raw_payload, policy)
    metadata = dict(payload.get("_orchestrator") or {})
    return payload, metadata


def is_llm_generation_endpoint(path: str) -> bool:
    normalized = path.strip("/")
    return normalized in {
        "chat/completions",
        "responses",
        "completions",
        "embeddings",
    }


async def forward_without_limiter(path: str, request: Request, body: bytes) -> Response:
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            upstream_response = await client.request(
                request.method,
                upstream_url(settings.upstream_base_url, path),
                headers=upstream_headers(request),
                content=body,
                params=request.query_params,
            )
    except httpx.HTTPError:
        return error_response(
            status.HTTP_502_BAD_GATEWAY,
            "upstream_request_failed",
            "Upstream LLM gateway request failed.",
        )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
    )


async def stream_upstream_response(
    path: str,
    request: Request,
    body: bytes,
    model: str,
    endpoint: str,
    limiter: Any,
    started_at: float,
    upstream_base_url: str,
    backend_instance_id: str | None,
) -> StreamingResponse:
    client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    upstream_request = client.build_request(
        request.method,
        upstream_url(upstream_base_url, path),
        headers=upstream_headers(request),
        content=body,
        params=request.query_params,
    )
    upstream_response = await client.send(upstream_request, stream=True)

    async def response_body() -> Any:
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()
            await release_backend_lease(backend_instance_id)
            await limiter.release()
            record_limiter_snapshot(limiter)
            REQUESTS.labels(
                model=model,
                endpoint=endpoint,
                status=str(upstream_response.status_code),
            ).inc()
            LATENCY.labels(model=model, endpoint=endpoint).observe(
                perf_counter() - started_at
            )

    return StreamingResponse(
        response_body(),
        status_code=upstream_response.status_code,
        headers=response_headers(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
    )


async def resolve_upstream(model: str) -> tuple[str | None, str | None]:
    if not settings.enable_backend_registry_routing or backend_registry_client is None:
        return settings.upstream_base_url, None

    try:
        backend = await backend_registry_client.choose_backend(model)
    except httpx.HTTPError as exc:
        logger.warning("backend_registry_lookup_failed error_type=%s", type(exc).__name__)
        if settings.require_backend_registry_backend:
            return None, None
        return settings.upstream_base_url, None

    if backend is None:
        if settings.require_backend_registry_backend:
            return None, None
        return settings.upstream_base_url, None

    try:
        await backend_registry_client.lease_backend(backend.instance_id)
    except httpx.HTTPError as exc:
        logger.warning("backend_registry_lease_failed error_type=%s", type(exc).__name__)
        if settings.require_backend_registry_backend:
            return None, None
        return settings.upstream_base_url, None

    return backend.base_url, backend.instance_id


async def release_backend_lease(instance_id: str | None) -> None:
    if instance_id is None or backend_registry_client is None:
        return
    try:
        await backend_registry_client.release_backend(instance_id)
    except httpx.HTTPError as exc:
        logger.warning("backend_registry_release_failed error_type=%s", type(exc).__name__)


def upstream_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/v1/{path.lstrip('/')}"


def upstream_headers(request: Request) -> dict[str, str]:
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }
    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"
    return headers


def response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }


def error_response(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error_type,
                "message": message,
            }
        },
    )


def record_limiter_snapshot(limiter: Any) -> None:
    snapshot = limiter.snapshot()
    record_snapshot(snapshot.model, snapshot.active_requests, snapshot.queued_requests)


@app.exception_handler(PolicyError)
async def policy_error_handler(_request: Request, exc: PolicyError) -> JSONResponse:
    ERRORS.labels(model="unknown", error_type=exc.error_type).inc()
    return error_response(exc.status_code, exc.error_type, exc.message)
