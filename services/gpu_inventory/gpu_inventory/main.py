from __future__ import annotations

import logging

from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from orchestrator_core.logging import configure_json_logging
from orchestrator_core.prometheus import prom_labels

from gpu_inventory.nvidia_smi import GpuInventoryError, collect_inventory
from gpu_inventory.settings import Settings


settings = Settings()
configure_json_logging(settings.log_level)
logger = logging.getLogger(__name__)
app = FastAPI(title="local-llm-orchestrator GPU inventory", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"service": settings.service_name, "status": "healthy"}


@app.get("/gpus")
async def gpus() -> JSONResponse:
    try:
        snapshot = collect_inventory(
            settings.nvidia_smi_path,
            settings.command_timeout_seconds,
            settings.fake_gpu_inventory_json,
        )
    except GpuInventoryError as exc:
        logger.warning("gpu_inventory_failed error=%s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "source": "unavailable",
                "gpu_count": 0,
                "gpus": [],
                "error": {"type": "gpu_inventory_unavailable", "message": str(exc)},
            },
        )

    return JSONResponse(snapshot.to_dict())


@app.get("/metrics")
async def metrics() -> Response:
    try:
        snapshot = collect_inventory(
            settings.nvidia_smi_path,
            settings.command_timeout_seconds,
            settings.fake_gpu_inventory_json,
        )
    except GpuInventoryError:
        return Response(
            "gpu_inventory_available 0\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    lines = ["gpu_inventory_available 1"]
    for gpu in snapshot.gpus:
        labels = prom_labels(gpu=gpu.id, index=gpu.index, name=gpu.name)
        lines.append(f"gpu_memory_total_bytes{{{labels}}} {gpu.memory_total_mb * 1024 * 1024}")
        lines.append(f"gpu_memory_used_bytes{{{labels}}} {gpu.memory_used_mb * 1024 * 1024}")
        if gpu.utilization_gpu_percent is not None:
            lines.append(
                f"gpu_utilization_ratio{{{labels}}} {gpu.utilization_gpu_percent / 100}"
            )
    return Response(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
