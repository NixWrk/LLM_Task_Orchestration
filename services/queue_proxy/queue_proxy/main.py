from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from orchestrator_core.openai import openai_url
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
from queue_proxy.request_preparation import RequestPreparationService, should_stream_response
from queue_proxy.responses import error_response
from queue_proxy.routing import BackendResolver
from queue_proxy.settings import Settings
from queue_proxy.task_queue import (
    StoredTask,
    TaskProtocolError,
    build_task_store,
    parse_task_queue_payload,
)

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
task_store = build_task_store(
    settings.task_store_path,
    backend=settings.task_store_backend,
    dsn=settings.task_store_dsn,
)
app = FastAPI(title="local-llm-orchestrator queue proxy", version="0.1.0")
task_executor_task: asyncio.Task[None] | None = None


@app.on_event("startup")
async def start_task_executor() -> None:
    global task_executor_task
    if settings.task_executor_enabled:
        task_executor_task = asyncio.create_task(task_executor_loop())


@app.on_event("shutdown")
async def stop_task_executor() -> None:
    if task_executor_task is None:
        return
    task_executor_task.cancel()
    with suppress(asyncio.CancelledError):
        await task_executor_task


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


@app.post("/tasks/queue")
async def submit_task_queue(request: Request) -> JSONResponse:
    if settings.queue_proxy_api_key:
        auth_result = validate_proxy_auth(request)
        if auth_result is not None:
            return auth_result

    try:
        payload = await request.json()
        tasks = parse_task_queue_payload(payload)
    except TaskProtocolError as exc:
        ERRORS.labels(model="task_queue", error_type="invalid_task_protocol").inc()
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_task_protocol",
            exc.message,
        )
    except json.JSONDecodeError as exc:
        ERRORS.labels(model="task_queue", error_type="invalid_json").inc()
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_json",
            f"Invalid JSON request body: {exc.msg}",
        )

    accepted, reused = task_store.submit_many(tasks)
    queue_lengths = task_store.queue_lengths_by_model()
    context_plans = task_store.context_plans_by_model()
    capacity = await reconcile_capacity(queue_lengths, context_plans)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "accepted_tasks": len(accepted),
            "reused_tasks": len(reused),
            "queue_lengths": queue_lengths,
            "context_plans": context_plans,
            "tasks": [
                *(task.to_summary() for task in accepted),
                *(task.to_summary(reused=True) for task in reused),
            ],
            "capacity": capacity,
        },
    )


@app.get("/tasks")
async def list_tasks(request: Request) -> JSONResponse:
    if settings.queue_proxy_api_key:
        auth_result = validate_proxy_auth(request)
        if auth_result is not None:
            return auth_result

    tenant = tenant_from_request(request)
    if tenant is None:
        return missing_tenant_response()
    raw_limit = request.query_params.get("limit", "100")
    try:
        limit = max(1, min(int(raw_limit), 500))
    except ValueError:
        return error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_task_query",
            "limit must be an integer.",
        )
    tasks = task_store.list_tasks(
        tenant,
        state=request.query_params.get("state"),
        model=request.query_params.get("model"),
        limit=limit,
    )
    return JSONResponse({"tasks": [task.to_summary() for task in tasks]})


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> JSONResponse:
    if settings.queue_proxy_api_key:
        auth_result = validate_proxy_auth(request)
        if auth_result is not None:
            return auth_result

    tenant = tenant_from_request(request)
    if tenant is None:
        return missing_tenant_response()
    task = task_store.get_task(tenant, task_id)
    if task is None:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "task_not_found",
            "Task was not found for this tenant.",
        )
    return JSONResponse(task.to_detail())


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str, request: Request) -> JSONResponse:
    if settings.queue_proxy_api_key:
        auth_result = validate_proxy_auth(request)
        if auth_result is not None:
            return auth_result

    tenant = tenant_from_request(request)
    if tenant is None:
        return missing_tenant_response()
    task = task_store.cancel_task(tenant, task_id)
    if task is None:
        return error_response(
            status.HTTP_404_NOT_FOUND,
            "task_not_found",
            "Task was not found for this tenant.",
        )
    return JSONResponse(task.to_detail())


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
        if should_stream_response(clean_payload):
            response = await forwarder.stream_response(
                path=path,
                request=request,
                body=clean_body,
                upstream_base_url=upstream_base_url,
                on_finished=release_once,
            )
        else:
            response = await forwarder.forward_buffered_response(
                path,
                request,
                clean_body,
                upstream_base_url,
                watch_disconnect=True,
            )
            await release_once(response.status_code)

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


async def reconcile_capacity(
    queue_lengths: dict[str, int],
    context_plans: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if backend_registry_client is None:
        return {
            "state": "skipped",
            "reason": "backend_registry_url_not_configured",
        }
    try:
        result = await backend_registry_client.reconcile(queue_lengths, context_plans)
    except httpx.HTTPError as exc:
        ERRORS.labels(model="task_queue", error_type=type(exc).__name__).inc()
        logger.warning("task_queue_reconcile_failed error_type=%s", type(exc).__name__)
        return {
            "state": "failed",
            "error_type": type(exc).__name__,
        }
    return {
        "state": "reconciled",
        "result": result,
    }


async def task_executor_loop() -> None:
    while True:
        task = task_store.claim_next()
        if task is None:
            await asyncio.sleep(settings.task_executor_interval_seconds)
            continue

        try:
            await execute_stored_task(task)
        except Exception as exc:
            logger.exception("task_execution_failed task_id=%s", task.task_id)
            record_task_failure(
                task,
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                retryable=True,
            )


async def execute_stored_task(task: StoredTask) -> None:
    if not task.payload:
        record_task_failure(
            task,
            {
                "type": "missing_task_payload",
                "message": "Durable task execution requires a stored OpenAI-compatible payload.",
            },
            retryable=False,
        )
        return

    try:
        policy = policy_registry.resolve(task.model)
    except PolicyError as exc:
        record_task_failure(
            task,
            {
                "type": exc.error_type,
                "message": exc.message,
            },
            retryable=False,
        )
        return

    upstream_base_url, backend_instance_id = await backend_resolver.resolve(
        policy.public_name,
        task.orchestration,
    )
    if upstream_base_url is None:
        record_task_failure(
            task,
            {
                "type": "no_ready_backend",
                "message": "No ready backend instance is available for this task.",
            },
            retryable=True,
        )
        return

    try:
        clean_payload = strip_internal_fields({**task.payload})
        clean_payload.setdefault("model", policy.public_name)
        if backend_instance_id is not None:
            clean_payload["model"] = policy.backend_model

        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.post(
                openai_url(upstream_base_url, task.endpoint),
                json=clean_payload,
                headers=task_executor_headers(),
            )
    except httpx.HTTPError as exc:
        record_task_failure(
            task,
            {
                "type": "upstream_request_failed",
                "message": "Upstream LLM gateway request failed.",
                "error_type": type(exc).__name__,
                "backend_instance_id": backend_instance_id,
            },
            retryable=True,
        )
        return
    finally:
        await backend_resolver.release(backend_instance_id)

    body = response_body(response)
    if response.status_code >= 400:
        record_task_failure(
            task,
            {
                "type": "upstream_status",
                "message": "Upstream LLM backend returned an error status.",
                "status_code": response.status_code,
                "body": body,
                "backend_instance_id": backend_instance_id,
            },
            retryable=is_retryable_upstream_status(response.status_code),
        )
        return

    task_store.record_result(
        task.task_id,
        {
            "status_code": response.status_code,
            "body": body,
            "backend_instance_id": backend_instance_id,
        },
    )


def record_task_failure(
    task: StoredTask,
    error: dict[str, Any],
    *,
    retryable: bool,
) -> StoredTask:
    enriched_error = {
        **error,
        "retryable": retryable,
        "attempt_count": task.attempt_count,
        "max_attempts": settings.task_executor_max_attempts,
    }
    if retryable and task.attempt_count < settings.task_executor_max_attempts:
        delay_seconds = retry_delay_seconds(task.attempt_count)
        next_attempt_at = (datetime.now(UTC) + timedelta(seconds=delay_seconds)).isoformat()
        enriched_error["next_attempt_at"] = next_attempt_at
        enriched_error["retry_delay_seconds"] = delay_seconds
        return task_store.record_retry(task.task_id, enriched_error, next_attempt_at)

    return task_store.record_error(task.task_id, enriched_error)


def retry_delay_seconds(attempt_count: int) -> float:
    exponent = max(0, attempt_count - 1)
    delay = settings.task_executor_retry_base_seconds * (2**exponent)
    return min(delay, settings.task_executor_retry_max_seconds)


def is_retryable_upstream_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or status_code >= 500


def task_executor_headers() -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"
    return headers


def response_body(response: httpx.Response) -> Any:
    content_type = response.headers.get("content-type", "")
    if "json" not in content_type.lower():
        return response.text
    try:
        return response.json()
    except json.JSONDecodeError:
        return response.text


def tenant_from_request(request: Request) -> str | None:
    tenant = request.query_params.get("tenant") or request.headers.get("x-tenant-id")
    if tenant is None or not tenant.strip():
        return None
    return tenant.strip()


def missing_tenant_response() -> JSONResponse:
    return error_response(
        status.HTTP_400_BAD_REQUEST,
        "missing_tenant",
        "Task status requests require tenant query parameter or X-Tenant-ID header.",
    )


@app.exception_handler(PolicyError)
async def policy_error_handler(_request: Request, exc: PolicyError) -> JSONResponse:
    ERRORS.labels(model="unknown", error_type=exc.error_type).inc()
    return error_response(exc.status_code, exc.error_type, exc.message)
