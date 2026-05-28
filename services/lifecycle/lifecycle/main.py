from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import FastAPI, Request, Response
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


@app.get("/registry")
async def backend_registry() -> dict[str, object]:
    return {"instances": [instance.to_dict() for instance in registry.list()]}


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


@app.get("/metrics")
async def metrics() -> Response:
    instances = registry.list()
    lines = [
        f'llm_backend_instances{{state="{state}"}} '
        f'{sum(1 for instance in instances if instance.state == state)}'
        for state in ("starting", "warming", "ready", "draining", "failed", "stopped")
    ]
    return Response(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


async def periodic_reconcile_loop() -> None:
    while True:
        try:
            await controller.reconcile({})
        except Exception:
            logging.getLogger(__name__).exception("periodic_reconcile_failed")
        await asyncio.sleep(settings.reconcile_interval_seconds)
