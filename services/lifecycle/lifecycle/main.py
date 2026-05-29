from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from lifecycle.controller import LifecycleController, queue_lengths_from_payload
from lifecycle.registry import BackendRegistry
from lifecycle.settings import Settings


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
registry = BackendRegistry(settings.registry_path)
controller = LifecycleController(
    config_path=settings.config_path,
    registry=registry,
    gpu_inventory_url=settings.gpu_inventory_url,
    request_timeout_seconds=settings.request_timeout_seconds,
    dry_run=settings.dry_run,
    docker_binary=settings.docker_binary,
)
app = FastAPI(title="local-llm-orchestrator lifecycle", version="0.1.0")
reconcile_task: asyncio.Task[None] | None = None


@app.on_event("startup")
async def start_reconcile_loop() -> None:
    global reconcile_task
    if settings.enable_reconcile_loop:
        reconcile_task = asyncio.create_task(periodic_reconcile_loop())


@app.on_event("shutdown")
async def stop_reconcile_loop() -> None:
    if reconcile_task is None:
        return
    reconcile_task.cancel()
    with suppress(asyncio.CancelledError):
        await reconcile_task


@app.get("/health")
async def health() -> dict[str, str]:
    return {"service": settings.service_name, "status": "healthy"}


@app.get("/ready")
async def ready() -> dict[str, object]:
    return {
        "service": settings.service_name,
        "status": "healthy",
        "dry_run": settings.dry_run,
        "reconcile_loop_enabled": settings.enable_reconcile_loop,
    }


@app.get("/models")
async def models() -> dict[str, object]:
    return {
        "models": {
            name: profile.__dict__
            for name, profile in controller.profiles().items()
        }
    }


@app.get("/catalog/models")
async def catalog_models() -> dict[str, object]:
    return await controller.catalog()


@app.get("/registry")
async def backend_registry() -> dict[str, object]:
    return {"instances": [instance.to_dict() for instance in registry.list()]}


@app.post("/registry/{instance_id}/lease")
async def lease_backend(instance_id: str) -> JSONResponse:
    instance = registry.adjust_active_requests(instance_id, 1)
    return JSONResponse(instance.to_dict())


@app.delete("/registry/{instance_id}/lease")
async def release_backend(instance_id: str) -> JSONResponse:
    instance = registry.adjust_active_requests(instance_id, -1)
    return JSONResponse(instance.to_dict())


@app.post("/plan")
async def plan(request: Request) -> JSONResponse:
    payload = await request.json()
    result = await controller.plan(queue_lengths_from_payload(payload))
    return JSONResponse(result)


@app.post("/reconcile")
async def reconcile(request: Request) -> JSONResponse:
    payload = await request.json()
    result = await controller.reconcile(queue_lengths_from_payload(payload))
    return JSONResponse(result)


@app.post("/cleanup")
async def cleanup(request: Request) -> JSONResponse:
    payload = await request.json()
    result = await controller.cleanup(queue_lengths_from_payload(payload))
    return JSONResponse(result)


@app.post("/allocations")
async def allocate(request: Request) -> JSONResponse:
    payload = await request.json()
    try:
        result = await controller.allocate(payload)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result.get("instance") is None:
        return JSONResponse(result, status_code=409)
    return JSONResponse(result)


@app.get("/metrics")
async def metrics() -> Response:
    instances = registry.list()
    lines = [
        f'llm_backend_instances{{state="{state}",runtime="{runtime}"}} '
        f'{sum(1 for instance in instances if instance.state == state and instance.runtime == runtime)}'
        for state in ("starting", "warming", "ready", "draining", "failed", "stopped")
        for runtime in sorted({instance.runtime for instance in instances} or {"none"})
    ]
    for instance in instances:
        labels = prom_labels(model=instance.model, runtime=instance.runtime, state=instance.state)
        lines.append(f"llm_backend_active_requests{{{labels}}} {instance.active_requests}")
        lines.append(f"llm_backend_reserved_vram_mb{{{labels}}} {instance.reserved_vram_mb}")

    for (model, result), count in sorted(controller.allocation_results.items()):
        labels = prom_labels(model=model, result=result)
        lines.append(f"llm_allocations_total{{{labels}}} {count}")

    try:
        for gpu in await controller.gpu_states():
            labels = prom_labels(gpu_id=gpu.id, gpu_index=gpu.index, name=gpu.name)
            lines.append(f"llm_gpu_memory_total_mb{{{labels}}} {gpu.memory_total_mb}")
            lines.append(f"llm_gpu_memory_used_mb{{{labels}}} {gpu.memory_used_mb}")
            lines.append(f"llm_gpu_memory_free_mb{{{labels}}} {gpu.memory_free_mb}")
    except Exception:
        lines.append("llm_gpu_inventory_up 0")
    else:
        lines.append("llm_gpu_inventory_up 1")

    return Response(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def prom_labels(**labels: object) -> str:
    return ",".join(
        f'{name}="{prom_label_value(value)}"'
        for name, value in labels.items()
    )


def prom_label_value(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


async def periodic_reconcile_loop() -> None:
    while True:
        try:
            await controller.reconcile({})
        except Exception:
            logging.getLogger(__name__).exception("periodic_reconcile_failed")
        await asyncio.sleep(settings.reconcile_interval_seconds)
