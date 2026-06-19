from __future__ import annotations

from typing import Any

import httpx

from lifecycle.allocation import profile_with_context_plan
from lifecycle.allocation_service import AllocationService
from lifecycle.cleanup import CleanupService, idle_seconds
from lifecycle.config import (
    load_dynamic_model_profile,
    load_dynamic_models_config,
    load_model_profiles,
)
from lifecycle.dynamic_policy import dynamic_model_allowed
from lifecycle.models import (
    BackendInstance,
    ContextPlan,
    GpuState,
    ModelProfile,
    PlacementDecision,
    optional_int,
)
from lifecycle.registry import BackendRegistry
from lifecycle.runtime import RuntimeLifecycleService
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
        self.allocation_service = AllocationService(config_path, registry)
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

    async def plan(
        self,
        queue_lengths: dict[str, int] | None = None,
        context_plans: dict[str, ContextPlan] | None = None,
    ) -> dict[str, Any]:
        queue_lengths = queue_lengths or {}
        context_plans = context_plans or {}
        profiles = self.profiles()
        gpus = await self.gpu_states()
        plans: list[dict[str, Any]] = []

        for profile in profiles.values():
            context_plan = context_plans.get(profile.public_name)
            effective_profile = profile_with_context_plan(profile, context_plan)
            ready_count = len(self.registry.ready_for_model(profile.public_name))
            active_count = len(self.registry.active_for_model(profile.public_name))
            queue_length = max(
                queue_lengths.get(profile.public_name, 0),
                context_plan.queued_tasks if context_plan is not None else 0,
            )
            desired_count = desired_replicas(
                effective_profile,
                ready_count,
                queue_length,
            )
            # Scale one replica per reconcile cycle so placement can account for each
            # newly reserved backend before making the next decision.
            missing = 1 if desired_count > active_count else 0

            decisions: list[PlacementDecision] = []
            if context_plan is not None and context_plan.oversized_tasks:
                decisions.append(reject_oversized_decision(effective_profile, context_plan))
            else:
                for _ in range(missing):
                    decisions.append(choose_gpu(effective_profile, gpus, self.registry))

                if not decisions and queue_length > 0:
                    reload_decision = reload_decision_for_context_plan(
                        effective_profile,
                        self.registry.ready_for_model(profile.public_name),
                        context_plan,
                    )
                    if reload_decision is not None:
                        decisions.append(reload_decision)

            if not decisions and desired_count == active_count:
                decisions.append(
                    PlacementDecision(
                        model=effective_profile.public_name,
                        action="noop",
                        gpu_id=None,
                        reason="desired_replicas_satisfied",
                        required_vram_mb=(
                            effective_profile.estimated_vram_mb
                            + effective_profile.safety_margin_mb
                        ),
                        lms_context_length=effective_profile.lms_context_length,
                        lms_parallel=effective_profile.lms_parallel,
                        context_plan=context_plan.to_dict() if context_plan else None,
                    )
                )

            plans.append(
                {
                    "model": profile.public_name,
                    "ready_replicas": ready_count,
                    "active_replicas": active_count,
                    "desired_replicas": desired_count,
                    "desired_backend_shape": desired_backend_shape(effective_profile),
                    "context_plan": context_plan.to_dict() if context_plan else None,
                    "decisions": [decision.to_dict() for decision in decisions],
                }
            )

        return {
            "dry_run": self.dry_run,
            "gpu_count": len(gpus),
            "models": plans,
        }

    async def reconcile(
        self,
        queue_lengths: dict[str, int] | None = None,
        context_plans: dict[str, ContextPlan] | None = None,
    ) -> dict[str, Any]:
        queue_lengths = queue_lengths or {}
        context_plans = context_plans or {}
        profiles = self.profiles()
        stopped = await self.stop_idle_instances(profiles, queue_lengths)
        plan = await self.plan(queue_lengths, context_plans)
        created: list[dict[str, Any]] = []

        for model_plan in plan["models"]:
            profile = profile_with_context_plan(
                profiles[model_plan["model"]],
                context_plans.get(model_plan["model"]),
            )
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
        return await self.allocation_service.allocate(
            payload,
            profiles=self.profiles,
            gpu_states=self.gpu_states,
            verify_model_available=self.verify_model_available,
            start_instance=self.start_instance,
            initialize_instance=self.initialize_instance,
            record_allocation=self.record_allocation,
        )

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


def desired_backend_shape(profile: ModelProfile) -> dict[str, Any]:
    return {
        "runtime": profile.runtime,
        "lms_context_length": profile.lms_context_length,
        "lms_parallel": profile.lms_parallel,
        "lms_gpu": profile.lms_gpu,
        "required_vram_mb": profile.estimated_vram_mb + profile.safety_margin_mb,
        "estimated_vram_mb": profile.estimated_vram_mb,
        "safety_margin_mb": profile.safety_margin_mb,
    }


def reject_oversized_decision(
    profile: ModelProfile,
    context_plan: ContextPlan,
) -> PlacementDecision:
    return PlacementDecision(
        model=profile.public_name,
        action="reject_oversized",
        gpu_id=None,
        reason="context_plan_has_oversized_tasks",
        required_vram_mb=profile.estimated_vram_mb + profile.safety_margin_mb,
        lms_context_length=profile.lms_context_length,
        lms_parallel=profile.lms_parallel,
        context_plan=context_plan.to_dict(),
    )


def reload_decision_for_context_plan(
    profile: ModelProfile,
    ready_instances: list[BackendInstance],
    context_plan: ContextPlan | None,
) -> PlacementDecision | None:
    if context_plan is None or profile.runtime != "lmstudio":
        return None

    for instance in ready_instances:
        if backend_satisfies_profile_shape(instance, profile):
            continue
        current_context = lms_context_length_from_instance(instance)
        current_parallel = lms_parallel_from_instance(instance)
        return PlacementDecision(
            model=profile.public_name,
            action="reload",
            gpu_id=instance.gpu_ids[0] if instance.gpu_ids else None,
            reason="backend_shape_mismatch",
            required_vram_mb=profile.estimated_vram_mb + profile.safety_margin_mb,
            instance_id=instance.instance_id,
            lms_context_length=profile.lms_context_length,
            lms_parallel=profile.lms_parallel,
            current_lms_context_length=current_context,
            current_lms_parallel=current_parallel,
            context_plan=context_plan.to_dict(),
        )

    return None


def backend_satisfies_profile_shape(
    instance: BackendInstance,
    profile: ModelProfile,
) -> bool:
    desired_context = profile.lms_context_length
    desired_parallel = profile.lms_parallel
    if desired_context:
        current_context = lms_context_length_from_instance(instance)
        if current_context is None or current_context < desired_context:
            return False
    if desired_parallel:
        current_parallel = lms_parallel_from_instance(instance)
        if current_parallel is None or current_parallel < desired_parallel:
            return False
    return True


def lms_context_length_from_instance(instance: BackendInstance) -> int | None:
    return optional_int(instance.metadata.get("lms_context_length"))


def lms_parallel_from_instance(instance: BackendInstance) -> int | None:
    return optional_int(instance.metadata.get("lms_parallel"))
