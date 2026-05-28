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
