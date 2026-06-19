from __future__ import annotations

import os

from fastapi import FastAPI, Request

app = FastAPI(title="fake backend registry")
active_requests = int(os.environ.get("FAKE_REGISTRY_ACTIVE_REQUESTS", "0"))
allocated_model = os.environ.get("FAKE_REGISTRY_MODEL", "local-main")
last_reconcile_payload: dict[str, object] = {}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/registry")
async def registry() -> dict[str, object]:
    return {
        "instances": [
            {
                "instance_id": "fake-ready",
                "model": allocated_model,
                "base_url": os.environ["FAKE_REGISTRY_BACKEND_URL"],
                "state": "ready",
                "active_requests": active_requests,
            }
        ]
    }


@app.post("/allocations")
async def allocations(request: Request) -> dict[str, object]:
    global allocated_model
    payload = await request.json()
    allocated_model = str(payload["model"])
    if allocated_model == os.environ.get("FAKE_REGISTRY_DENIED_MODEL"):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="model denied")
    return {
        "model": allocated_model,
        "created": True,
        "instance": {
            "instance_id": "fake-ready",
            "model": allocated_model,
            "base_url": os.environ["FAKE_REGISTRY_BACKEND_URL"],
            "state": "ready",
            "active_requests": active_requests,
        },
    }


@app.post("/reconcile")
async def reconcile(request: Request) -> dict[str, object]:
    global last_reconcile_payload
    payload = await request.json()
    last_reconcile_payload = payload
    queue_lengths = payload.get("queue_lengths", {})
    context_plans = payload.get("context_plans", {})
    return {
        "dry_run": True,
        "queue_lengths": queue_lengths,
        "context_plans": context_plans,
        "models": [
            {
                "model": model,
                "ready_replicas": 0,
                "active_replicas": 0,
                "desired_replicas": 1 if int(length) > 0 else 0,
                "decisions": [
                    {
                        "model": model,
                        "action": "start" if int(length) > 0 else "noop",
                        "gpu_id": "gpu0" if int(length) > 0 else None,
                        "reason": "vram_available" if int(length) > 0 else "empty_queue",
                        "required_vram_mb": 1024,
                        "available_vram_mb": 24576,
                    }
                ],
            }
            for model, length in dict(queue_lengths).items()
        ],
        "created_instances": [],
    }


@app.get("/last_reconcile")
async def last_reconcile() -> dict[str, object]:
    return last_reconcile_payload


@app.post("/registry/{_instance_id}/lease")
async def lease_backend(_instance_id: str) -> dict[str, int]:
    global active_requests
    active_requests += 1
    return {"active_requests": active_requests}


@app.delete("/registry/{_instance_id}/lease")
async def release_backend(_instance_id: str) -> dict[str, int]:
    global active_requests
    active_requests = max(0, active_requests - 1)
    return {"active_requests": active_requests}
