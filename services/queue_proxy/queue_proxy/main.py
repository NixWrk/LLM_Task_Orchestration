from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from orchestrator_core.logging import configure_json_logging

from queue_proxy.backend_registry import BackendRegistryClient
from queue_proxy.forwarder import ClientDisconnectedError, UpstreamForwarder
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
    extract_model,
    load_policy_registry,
    strip_internal_fields,
)
from queue_proxy.request_preparation import RequestPreparationService
from queue_proxy.responses import error_response
from queue_proxy.routing import BackendResolver
from queue_proxy.settings import Settings

settings = Settings()
configure_json_logging(settings.log_level)
logger = logging.getLogger(__name__)

policy_registry = load_policy_registry(settings.config_path)
limiter_registry = LimiterRegistry()
backend_registry_client = (
    BackendRegistryClient(settings.backend_registry_url, settings.request_timeout_seconds)
    if settings.backend_registry_url
    else None
)
request_preparer = RequestPreparationService(policy_registry)
backend_resolver = BackendResolver(settings, backend_registry_client, logger)
forwarder = UpstreamForwarder(settings.request_timeout_seconds, settings.upstream_api_key)
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
    payload, policy_metadata, effective_policy = request_preparer.prepare(
        path,
        request.method,
        request.headers,
        body,
    )

    if payload is None:
        return await forwarder.forward_without_limiter(
            path,
            request,
            body,
            settings.upstream_base_url,
        )

    policy = effective_policy or policy_registry.resolve(extract_model(payload))
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
    released = False

    async def release_once(status_code: int | None = None) -> None:
        nonlocal released
        if released:
            return
        released = True
        await backend_resolver.release(backend_instance_id)
        await limiter.release()
        record_limiter_snapshot(limiter)
        if status_code is not None:
            REQUESTS.labels(
                model=policy.public_name,
                endpoint=endpoint,
                status=str(status_code),
            ).inc()
            LATENCY.labels(model=policy.public_name, endpoint=endpoint).observe(
                perf_counter() - started_at
            )

    try:
        orchestration = payload.get("orchestration")
        orchestration_payload = orchestration if isinstance(orchestration, dict) else None
        upstream_base_url, backend_instance_id = await backend_resolver.resolve(
            policy.public_name,
            orchestration_payload,
        )
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
        if backend_instance_id is not None and "model" in clean_payload:
            clean_payload["model"] = policy.backend_model
        clean_body = json.dumps(clean_payload, separators=(",", ":")).encode("utf-8")
        response = await forwarder.stream_response(
            path=path,
            request=request,
            body=clean_body,
            upstream_base_url=upstream_base_url,
            on_finished=release_once,
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
        await release_once(502)
        ERRORS.labels(model=policy.public_name, error_type=type(exc).__name__).inc()
        logger.warning("upstream_request_failed error_type=%s", type(exc).__name__)
        return error_response(
            status.HTTP_502_BAD_GATEWAY,
            "upstream_request_failed",
            "Upstream LLM gateway request failed.",
        )
    except ClientDisconnectedError:
        await release_once(499)
        ERRORS.labels(model=policy.public_name, error_type="client_disconnected").inc()
        logger.info("client_disconnected_before_upstream_response")
        return Response(status_code=499)
    except BaseException:
        await release_once()
        raise


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


async def finish_upstream_response(
    backend_instance_id: str | None,
    limiter: Any,
    model: str,
    endpoint: str,
    status_code: int,
    started_at: float,
) -> None:
    await backend_resolver.release(backend_instance_id)
    await limiter.release()
    record_limiter_snapshot(limiter)
    REQUESTS.labels(
        model=model,
        endpoint=endpoint,
        status=str(status_code),
    ).inc()
    LATENCY.labels(model=model, endpoint=endpoint).observe(perf_counter() - started_at)


def record_limiter_snapshot(limiter: Any) -> None:
    snapshot = limiter.snapshot()
    record_snapshot(snapshot.model, snapshot.active_requests, snapshot.queued_requests)


@app.exception_handler(PolicyError)
async def policy_error_handler(_request: Request, exc: PolicyError) -> JSONResponse:
    ERRORS.labels(model="unknown", error_type=exc.error_type).inc()
    return error_response(exc.status_code, exc.error_type, exc.message)
