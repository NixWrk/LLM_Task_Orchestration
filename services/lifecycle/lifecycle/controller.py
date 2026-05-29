from __future__ import annotations

from typing import Any

import httpx

from lifecycle.allocation import (
    allocation_overrides,
    enrich_profile_from_lmstudio_metadata,
    queue_lengths_from_payload,
)
from lifecycle.cleanup import CleanupService, idle_seconds
from lifecycle.config import (
    load_dynamic_model_profile,
    load_dynamic_models_config,
    load_model_profiles,
)
from lifecycle.dynamic_policy import dynamic_model_allowed
from lifecycle.models import (
    BackendInstance,
    GpuState,
    ModelProfile,
    PlacementDecision,
)
from lifecycle.registry import BackendRegistry
from lifecycle.runtime import RuntimeLifecycleService, should_verify_before_start
from lifecycle.scheduler import choose_gpu, desired_replicas


class LifecycleController:
    def __init__(
        self,
        config_path: str,
        registry: BackendRegistry,
        gpu_inventory_url: str,
        request_timeout_seconds: float,
        dry_run: bool,
        docker_binary: str = "docker",
    ) -> None:
        self.config_path = config_path
        self.registry = registry
        self.gpu_inventory_url = gpu_inventory_url.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds
        self.dry_run = dry_run
        self.docker_binary = docker_binary
        self.allocation_results: dict[tuple[str, str], int] = {}
        self.runtime = RuntimeLifecycleService(
            registry=registry,
            request_timeout_seconds=request_timeout_seconds,
            dry_run=dry_run,
            docker_binary=docker_binary,
        )
        self.cleanup_service = CleanupService(registry, self.stop_instance)

    def profiles(self) -> dict[str, ModelProfile]:
        return load_model_profiles(self.config_path)

    def profile_for_model(
        self,
        model: str,
        overrides: dict[str, Any] | None = None,
    ) -> ModelProfile | None:
        profiles = self.profiles()
        if model in profiles:
            return profiles[model]
        return load_dynamic_model_profile(self.config_path, model, overrides)

    async def catalog(self) -> dict[str, Any]:
        profiles = self.profiles()
        dynamic_config = load_dynamic_models_config(self.config_path)
        dynamic_enabled = bool(dynamic_config.get("enabled", False))
        dynamic_models: list[dict[str, Any]] = []

        if dynamic_enabled:
            probe = load_dynamic_model_profile(self.config_path, "__catalog_probe__")
            if probe and probe.base_url:
                for model_id in await self.list_openai_model_ids(probe.base_url):
                    dynamic_models.append(
                        {
                            "id": model_id,
                            "allowed": dynamic_model_allowed(model_id, dynamic_config),
                            "source": dynamic_config.get("source", "lmstudio"),
                        }
                    )

        return {
            "configured_models": [
                {
                    "id": profile.public_name,
                    "backend_model": profile.backend_model,
                    "runtime": profile.runtime,
                }
                for profile in profiles.values()
            ],
            "dynamic_models_enabled": dynamic_enabled,
            "dynamic_models": dynamic_models,
        }

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
            # Scale one replica per reconcile cycle so placement can account for each
            # newly reserved backend before making the next decision.
            missing = 1 if desired_count > active_count else 0

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
        queue_lengths = queue_lengths or {}
        profiles = self.profiles()
        stopped = await self.stop_idle_instances(profiles, queue_lengths)
        plan = await self.plan(queue_lengths)
        created: list[dict[str, Any]] = []

        for model_plan in plan["models"]:
            profile = profiles[model_plan["model"]]
            gpus = {gpu.id: gpu for gpu in await self.gpu_states()}
            for decision_payload in model_plan["decisions"]:
                if decision_payload["action"] != "start" or not decision_payload["gpu_id"]:
                    continue
                instance = self.start_instance(profile, decision_payload, gpus)
                self.registry.upsert(instance)
                instance = await self.initialize_instance(profile, instance)
                created.append(instance.to_dict())

        plan["created_instances"] = created
        plan["stopped_instances"] = stopped
        return plan

    async def allocate(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = str(payload.get("model") or "").strip()
        if not model:
            self.record_allocation("unknown", "invalid_request")
            raise ValueError("Allocation requires a non-empty model.")

        overrides = allocation_overrides(payload)
        configured_profiles = self.profiles()
        profile = configured_profiles.get(model)
        dynamic_config: dict[str, Any] = {}
        if profile is None:
            dynamic_config = load_dynamic_models_config(self.config_path)
            if not dynamic_model_allowed(model, dynamic_config):
                self.record_allocation(model, "denied")
                raise PermissionError(f"Model {model} is not allowed by dynamic model policy.")
            profile = load_dynamic_model_profile(self.config_path, model, overrides)
        if profile is None:
            self.record_allocation(model, "not_configured")
            raise LookupError(f"Model {model} is not configured and dynamic models are disabled.")

        ready_instances = self.registry.ready_for_model(profile.public_name)
        if ready_instances:
            instance = min(ready_instances, key=lambda item: item.active_requests)
            self.record_allocation(profile.public_name, "reused")
            return {
                "model": profile.public_name,
                "created": False,
                "instance": instance.to_dict(),
            }

        profile, model_metadata = enrich_profile_from_lmstudio_metadata(
            profile,
            dynamic_config,
            payload,
        )
        if should_verify_before_start(profile):
            await self.verify_model_available(profile)
        gpus = await self.gpu_states()
        decision = choose_gpu(profile, gpus, self.registry)
        if decision.action != "start" or not decision.gpu_id:
            self.record_allocation(profile.public_name, "insufficient_vram")
            return {
                "model": profile.public_name,
                "created": False,
                "decision": decision.to_dict(),
                "instance": None,
            }

        gpu_by_id = {gpu.id: gpu for gpu in gpus}
        try:
            instance = self.start_instance(profile, decision.to_dict(), gpu_by_id)
        except Exception:
            self.record_allocation(profile.public_name, "failed")
            raise
        instance.metadata.update(model_metadata)
        self.registry.upsert(instance)
        instance = await self.initialize_instance(profile, instance)
        self.record_allocation(
            profile.public_name,
            "created" if instance.state == "ready" else "failed",
        )
        return {
            "model": profile.public_name,
            "created": instance.state == "ready",
            "decision": decision.to_dict(),
            "instance": instance.to_dict(),
        }

    def record_allocation(self, model: str, result: str) -> None:
        key = (model, result)
        self.allocation_results[key] = self.allocation_results.get(key, 0) + 1

    def start_instance(
        self,
        profile: ModelProfile,
        decision_payload: dict[str, Any],
        gpu_by_id: dict[str, GpuState],
    ) -> BackendInstance:
        return self.runtime.start_instance(profile, decision_payload, gpu_by_id)

    async def initialize_instance(
        self,
        profile: ModelProfile,
        instance: BackendInstance,
    ) -> BackendInstance:
        return await self.runtime.initialize_instance(profile, instance)

    async def wait_for_health(
        self,
        profile: ModelProfile,
        instance: BackendInstance,
    ) -> None:
        await self.runtime.wait_for_health(profile, instance)

    async def warmup(self, profile: ModelProfile, instance: BackendInstance) -> None:
        await self.runtime.warmup(profile, instance)

    async def verify_model_available(self, profile: ModelProfile) -> None:
        await self.runtime.verify_model_available(profile)

    async def list_openai_model_ids(self, base_url: str) -> list[str]:
        return await self.runtime.list_openai_model_ids(base_url)

    async def stop_idle_instances(
        self,
        profiles: dict[str, ModelProfile],
        queue_lengths: dict[str, int],
    ) -> list[dict[str, Any]]:
        return await self.cleanup_service.stop_idle_instances(profiles, queue_lengths)

    async def cleanup(self, queue_lengths: dict[str, int] | None = None) -> dict[str, Any]:
        return await self.cleanup_service.cleanup(
            profiles=self.profiles(),
            queue_lengths=queue_lengths or {},
            dynamic_config=load_dynamic_models_config(self.config_path),
            profile_for_model=self.profile_for_model,
        )

    def purge_stale_instances(self, ttl_seconds: int) -> list[dict[str, Any]]:
        return self.cleanup_service.purge_stale_instances(ttl_seconds)

    async def stop_instance(
        self,
        profile: ModelProfile,
        instance: BackendInstance,
    ) -> dict[str, Any]:
        return await self.runtime.stop_instance(profile, instance)
