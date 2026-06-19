from __future__ import annotations

from pathlib import Path

from lifecycle.models import BackendInstance, now_iso
from lifecycle.registry_store import JsonFileRegistryStore, RegistryStore

ACTIVE_STATES = {"starting", "warming", "ready", "draining", "stopping"}
READY_STATES = {"ready"}


class BackendRegistry:
    def __init__(
        self,
        path: str | Path | None = None,
        store: RegistryStore | None = None,
    ) -> None:
        if store is None:
            if path is None:
                raise ValueError("BackendRegistry requires a path or store.")
            store = JsonFileRegistryStore(path)
        self.store = store
        self.path = Path(path) if path is not None else getattr(store, "path", None)
        self._instances: dict[str, BackendInstance] = {}
        self.load()

    def load(self) -> None:
        payload = self.store.load()
        instances = payload.get("instances", [])
        self._instances = {
            str(item["instance_id"]): BackendInstance.from_dict(item)
            for item in instances
            if isinstance(item, dict)
        }

    def save(self) -> None:
        payload = {"instances": [instance.to_dict() for instance in self.list()]}
        self.store.save(payload)

    def list(self) -> list[BackendInstance]:
        return list(self._instances.values())

    def get(self, instance_id: str) -> BackendInstance:
        return self._instances[instance_id]

    def active_for_model(self, model: str) -> list[BackendInstance]:
        return [
            instance
            for instance in self._instances.values()
            if instance.model == model and instance.state in ACTIVE_STATES
        ]

    def ready_for_model(self, model: str) -> list[BackendInstance]:
        return [
            instance
            for instance in self._instances.values()
            if instance.model == model and instance.state in READY_STATES
        ]

    def reserved_vram_by_gpu(self) -> dict[str, int]:
        reserved: dict[str, int] = {}
        for instance in self._instances.values():
            if instance.state not in ACTIVE_STATES:
                continue
            for gpu_id in instance.gpu_ids:
                reserved[gpu_id] = reserved.get(gpu_id, 0) + instance.reserved_vram_mb
        return reserved

    def next_host_port(self, model: str, host_port_start: int) -> int:
        used_ports = {
            instance.host_port
            for instance in self._instances.values()
            if instance.host_port is not None
        }
        offset = 0
        while host_port_start + offset in used_ports:
            offset += 1
        return host_port_start + offset

    def upsert(self, instance: BackendInstance) -> None:
        instance.updated_at = now_iso()
        self._instances[instance.instance_id] = instance
        self.save()

    def mark_state(self, instance_id: str, state: str) -> BackendInstance:
        instance = self._instances[instance_id]
        instance.state = state  # type: ignore[assignment]
        instance.updated_at = now_iso()
        self.save()
        return instance

    def mark_failed(self, instance_id: str, reason: str) -> BackendInstance:
        instance = self._instances[instance_id]
        instance.state = "failed"
        instance.failure_reason = reason
        instance.updated_at = now_iso()
        self.save()
        return instance

    def adjust_active_requests(self, instance_id: str, delta: int) -> BackendInstance:
        instance = self._instances[instance_id]
        instance.active_requests = max(0, instance.active_requests + delta)
        instance.updated_at = now_iso()
        if delta > 0:
            instance.last_used_at = instance.updated_at
        self.save()
        return instance

    def remove(self, instance_id: str) -> None:
        self._instances.pop(instance_id, None)
        self.save()
