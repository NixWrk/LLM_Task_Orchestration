from __future__ import annotations

import json
from pathlib import Path

from lifecycle.models import BackendInstance, now_iso

ACTIVE_STATES = {"starting", "warming", "ready"}
READY_STATES = {"ready"}


class BackendRegistry:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._instances: dict[str, BackendInstance] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._instances = {}
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        instances = payload.get("instances", [])
        self._instances = {
            str(item["instance_id"]): BackendInstance.from_dict(item)
            for item in instances
            if isinstance(item, dict)
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"instances": [instance.to_dict() for instance in self.list()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list(self) -> list[BackendInstance]:
        return list(self._instances.values())

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

    def remove(self, instance_id: str) -> None:
        self._instances.pop(instance_id, None)
        self.save()
