from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import httpx

from lifecycle.config import load_model_profiles
from lifecycle.models import BackendInstance, GpuState, ModelProfile, PlacementDecision, now_iso
from lifecycle.registry import BackendRegistry
from lifecycle.scheduler import choose_gpu, desired_replicas


class LifecycleController:
    def __init__(
        self,
        config_path: str,
        registry: BackendRegistry,
        gpu_inventory_url: str,
        request_timeout_seconds: float,
        dry_run: bool,
    ) -> None:
        self.config_path = config_path
        self.registry = registry
        self.gpu_inventory_url = gpu_inventory_url.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds
        self.dry_run = dry_run

    def profiles(self) -> dict[str, ModelProfile]:
        return load_model_profiles(self.config_path)

    async def gpu_states(self) -> list[GpuState]:
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.get(f"{self.gpu_inventory_url}/gpus")
            response.raise_for_status()
            payload = response.json()
        return [
            GpuState(
                id=str(item["id"]),
                index=int(item["index"]),
                name=str(item["name"]),
                memory_total_mb=int(item["memory_total_mb"]),
                memory_used_mb=int(item["memory_used_mb"]),
                memory_free_mb=int(item["memory_free_mb"]),
            )
            for item in payload.get("gpus", [])
        ]

    async def plan(self, queue_lengths: dict[str, int] | None = None) -> dict[str, Any]:
        queue_lengths = queue_lengths or {}
        profiles = self.profiles()
        gpus = await self.gpu_states()
        plans: list[dict[str, Any]] = []

        for profile in profiles.values():
            ready_count = len(self.registry.ready_for_model(profile.public_name))
            active_count = len(self.registry.active_for_model(profile.public_name))
            desired_count = desired_replicas(
                profile,
                ready_count,
                queue_lengths.get(profile.public_name, 0),
            )
            missing = max(0, desired_count - active_count)

            decisions: list[PlacementDecision] = []
            for _ in range(missing):
                decisions.append(choose_gpu(profile, gpus, self.registry))

            if not decisions and desired_count == active_count:
                decisions.append(
                    PlacementDecision(
                        model=profile.public_name,
                        action="noop",
                        gpu_id=None,
                        reason="desired_replicas_satisfied",
                        required_vram_mb=profile.estimated_vram_mb + profile.safety_margin_mb,
                    )
                )

            plans.append(
                {
                    "model": profile.public_name,
                    "ready_replicas": ready_count,
                    "active_replicas": active_count,
                    "desired_replicas": desired_count,
                    "decisions": [decision.to_dict() for decision in decisions],
                }
            )

        return {
            "dry_run": self.dry_run,
            "gpu_count": len(gpus),
            "models": plans,
        }

    async def reconcile(self, queue_lengths: dict[str, int] | None = None) -> dict[str, Any]:
        plan = await self.plan(queue_lengths)
        created: list[dict[str, Any]] = []
        profiles = self.profiles()

        for model_plan in plan["models"]:
            profile = profiles[model_plan["model"]]
            for decision_payload in model_plan["decisions"]:
                if decision_payload["action"] != "start" or not decision_payload["gpu_id"]:
                    continue
                instance = self.create_dry_run_instance(profile, decision_payload)
                self.registry.upsert(instance)
                created.append(instance.to_dict())

        plan["created_instances"] = created
        return plan

    def create_dry_run_instance(
        self,
        profile: ModelProfile,
        decision_payload: dict[str, Any],
    ) -> BackendInstance:
        gpu_id = str(decision_payload["gpu_id"])
        instance_id = instance_id_for(profile.public_name, gpu_id, now_iso())
        return BackendInstance(
            instance_id=instance_id,
            model=profile.public_name,
            backend_model=profile.backend_model,
            runtime=profile.runtime,
            base_url=f"dry-run://{profile.public_name}/{instance_id}",
            gpu_ids=[gpu_id],
            state="ready" if self.dry_run else "starting",
            reserved_vram_mb=int(decision_payload["required_vram_mb"]),
            dry_run=self.dry_run,
        )


def instance_id_for(model: str, gpu_id: str, seed: str) -> str:
    digest = hashlib.sha1(f"{model}:{gpu_id}:{seed}".encode("utf-8")).hexdigest()[:8]
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"{safe_model}-{gpu_id}-{digest}"


def queue_lengths_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("queue_lengths", {})
    if not isinstance(raw, dict):
        return {}
    return {str(model): int(length) for model, length in raw.items()}
