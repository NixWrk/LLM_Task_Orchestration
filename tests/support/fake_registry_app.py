from __future__ import annotations

import os

from fastapi import FastAPI, Request

app = FastAPI(title="fake backend registry")
active_requests = int(os.environ.get("FAKE_REGISTRY_ACTIVE_REQUESTS", "0"))
allocated_model = os.environ.get("FAKE_REGISTRY_MODEL", "local-main")


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
