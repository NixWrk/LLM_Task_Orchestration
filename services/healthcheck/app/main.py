from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

import httpx
from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from orchestrator_core.logging import configure_json_logging
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.lmstudio_client import OpenAICompatibleClient
from app.metrics import (
    record_backend_health,
    record_error,
    record_loaded_models,
    record_readiness_latency,
)
from app.settings import Settings
from app.state import DEGRADED, HEALTHY, UNHEALTHY, overall_status


settings = Settings()
configure_json_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="local-llm-gateway healthcheck", version="0.1.0")
client = OpenAICompatibleClient(settings.request_timeout_seconds)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "service": settings.service_name,
        "status": HEALTHY,
    }


@app.get("/ready")
async def ready() -> JSONResponse:
    result = await run_readiness_check()
    response_status = (
        status.HTTP_200_OK
        if result["status"] == HEALTHY
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(result, status_code=response_status)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def run_readiness_check() -> dict[str, Any]:
    started_at = perf_counter()
    checks: list[dict[str, Any]] = []
    loaded_model_count = 0

    lmstudio_check = await check_lmstudio_models()
    checks.append(lmstudio_check)
    loaded_model_count = int(lmstudio_check.get("loaded_model_count", 0))

    if settings.enable_litellm_path_check:
        checks.append(await check_litellm_completion())

    elapsed = perf_counter() - started_at
    status_value = overall_status(checks)

    record_readiness_latency(elapsed)
    record_loaded_models(loaded_model_count)
    record_backend_health("lmstudio", lmstudio_check["status"])
    if settings.enable_litellm_path_check:
        record_backend_health("litellm", checks[-1]["status"])

    logger.info(
        "readiness_check_completed status=%s latency_seconds=%.3f checks=%d",
        status_value,
        elapsed,
        len(checks),
    )

    return {
        "service": settings.service_name,
        "status": status_value,
        "latency_seconds": round(elapsed, 3),
        "checks": checks,
    }


async def check_lmstudio_models() -> dict[str, Any]:
    started_at = perf_counter()
    check_name = "lmstudio_models"
    try:
        model_ids = await client.list_models(
            settings.lmstudio_openai_base_url,
            settings.lmstudio_api_key,
        )
    except (httpx.HTTPError, ValueError) as exc:
        record_error(check_name)
        logger.warning("lmstudio_models_check_failed error_type=%s", type(exc).__name__)
        return {
            "name": check_name,
            "status": UNHEALTHY,
            "latency_seconds": round(perf_counter() - started_at, 3),
            "error_type": type(exc).__name__,
            "loaded_model_count": 0,
        }

    expected_model = settings.lmstudio_model_id
    model_found = expected_model in model_ids
    check_status = HEALTHY if model_found else DEGRADED
    if not model_found:
        record_error(check_name)

    return {
        "name": check_name,
        "status": check_status,
        "latency_seconds": round(perf_counter() - started_at, 3),
        "expected_model": expected_model,
        "model_found": model_found,
        "loaded_model_count": len(model_ids),
        "loaded_models": model_ids,
    }


async def check_litellm_completion() -> dict[str, Any]:
    started_at = perf_counter()
    check_name = "litellm_chat_completion"
    try:
        response = await client.chat_completion(
            settings.litellm_base_url,
            settings.litellm_master_key,
            settings.public_model_name,
            settings.readiness_prompt,
            settings.readiness_max_tokens,
        )
    except (httpx.HTTPError, ValueError) as exc:
        record_error(check_name)
        logger.warning("litellm_completion_check_failed error_type=%s", type(exc).__name__)
        return {
            "name": check_name,
            "status": UNHEALTHY,
            "latency_seconds": round(perf_counter() - started_at, 3),
            "error_type": type(exc).__name__,
        }

    content = str(response.get("content") or "")
    check_status = HEALTHY if content.strip() else DEGRADED
    if check_status != HEALTHY:
        record_error(check_name)

    return {
        "name": check_name,
        "status": check_status,
        "latency_seconds": round(perf_counter() - started_at, 3),
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "content_present": bool(content.strip()),
    }
