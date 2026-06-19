from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class BackendInstance:
    instance_id: str
    model: str
    base_url: str
    state: str
    active_requests: int

    @property
    def is_ready(self) -> bool:
        return self.state == "ready" and self.base_url.startswith(("http://", "https://"))


class BackendRegistryClient:
    def __init__(self, registry_url: str, timeout_seconds: float) -> None:
        self.registry_url = registry_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def list_instances(self) -> list[BackendInstance]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(f"{self.registry_url}/registry")
            response.raise_for_status()
            payload = response.json()
        return parse_registry_instances(payload)

    async def choose_backend(self, model: str) -> BackendInstance | None:
        instances = await self.list_instances()
        candidates = [
            instance
            for instance in instances
            if instance.model == model and instance.is_ready
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda instance: instance.active_requests)

    async def ensure_allocation(
        self,
        model: str,
        orchestration: dict[str, Any] | None = None,
    ) -> BackendInstance | None:
        payload: dict[str, Any] = {"model": model}
        if orchestration:
            payload["orchestration"] = orchestration

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.registry_url}/allocations", json=payload)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

        instance = data.get("instance")
        if not isinstance(instance, dict):
            return None
        parsed = parse_registry_instances({"instances": [instance]})
        return parsed[0] if parsed else None

    async def reconcile(
        self,
        queue_lengths: dict[str, int],
        context_plans: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"queue_lengths": queue_lengths}
        if context_plans is not None:
            payload["context_plans"] = context_plans
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.registry_url}/reconcile",
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def explain_plan(
        self,
        queue_lengths: dict[str, int],
        context_plans: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"queue_lengths": queue_lengths}
        if context_plans is not None:
            payload["context_plans"] = context_plans
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.registry_url}/explain-plan",
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def lease_backend(self, instance_id: str) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.registry_url}/registry/{instance_id}/lease")
            response.raise_for_status()

    async def release_backend(self, instance_id: str) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.delete(f"{self.registry_url}/registry/{instance_id}/lease")
            response.raise_for_status()


def parse_registry_instances(payload: dict[str, Any]) -> list[BackendInstance]:
    raw_instances = payload.get("instances", [])
    if not isinstance(raw_instances, list):
        return []

    instances: list[BackendInstance] = []
    for item in raw_instances:
        if not isinstance(item, dict):
            continue
        instances.append(
            BackendInstance(
                instance_id=str(item.get("instance_id") or ""),
                model=str(item.get("model") or ""),
                base_url=str(item.get("base_url") or ""),
                state=str(item.get("state") or "unknown"),
                active_requests=int(item.get("active_requests") or 0),
            )
        )
    return instances
