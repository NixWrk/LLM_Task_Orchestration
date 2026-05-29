from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from lifecycle.allocation import (
    allocation_overrides,
    enrich_profile_from_lmstudio_metadata,
)
from lifecycle.config import load_dynamic_model_profile, load_dynamic_models_config
from lifecycle.dynamic_policy import dynamic_model_allowed
from lifecycle.models import BackendInstance, GpuState, ModelProfile
from lifecycle.registry import BackendRegistry
from lifecycle.runtime import should_verify_before_start
from lifecycle.scheduler import choose_gpu

ProfileProvider = Callable[[], dict[str, ModelProfile]]
GpuStateProvider = Callable[[], Awaitable[list[GpuState]]]
ModelVerifier = Callable[[ModelProfile], Awaitable[None]]
InstanceStarter = Callable[
    [ModelProfile, dict[str, Any], dict[str, GpuState]],
    BackendInstance,
]
InstanceInitializer = Callable[[ModelProfile, BackendInstance], Awaitable[BackendInstance]]
AllocationRecorder = Callable[[str, str], None]


class AllocationService:
    def __init__(self, config_path: str, registry: BackendRegistry) -> None:
        self.config_path = config_path
        self.registry = registry

    async def allocate(
        self,
        payload: dict[str, Any],
        *,
        profiles: ProfileProvider,
        gpu_states: GpuStateProvider,
        verify_model_available: ModelVerifier,
        start_instance: InstanceStarter,
        initialize_instance: InstanceInitializer,
        record_allocation: AllocationRecorder,
    ) -> dict[str, Any]:
        model = str(payload.get("model") or "").strip()
        if not model:
            record_allocation("unknown", "invalid_request")
            raise ValueError("Allocation requires a non-empty model.")

        profile, dynamic_config = self.resolve_profile(
            model,
            allocation_overrides(payload),
            profiles(),
            record_allocation,
        )

        ready_instances = self.registry.ready_for_model(profile.public_name)
        if ready_instances:
            instance = min(ready_instances, key=lambda item: item.active_requests)
            record_allocation(profile.public_name, "reused")
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
            await verify_model_available(profile)

        gpus = await gpu_states()
        decision = choose_gpu(profile, gpus, self.registry)
        if decision.action != "start" or not decision.gpu_id:
            record_allocation(profile.public_name, "insufficient_vram")
            return {
                "model": profile.public_name,
                "created": False,
                "decision": decision.to_dict(),
                "instance": None,
            }

        gpu_by_id = {gpu.id: gpu for gpu in gpus}
        try:
            instance = start_instance(profile, decision.to_dict(), gpu_by_id)
        except Exception:
            record_allocation(profile.public_name, "failed")
            raise

        instance.metadata.update(model_metadata)
        self.registry.upsert(instance)
        instance = await initialize_instance(profile, instance)
        record_allocation(
            profile.public_name,
            "created" if instance.state == "ready" else "failed",
        )
        return {
            "model": profile.public_name,
            "created": instance.state == "ready",
            "decision": decision.to_dict(),
            "instance": instance.to_dict(),
        }

    def resolve_profile(
        self,
        model: str,
        overrides: dict[str, Any],
        configured_profiles: dict[str, ModelProfile],
        record_allocation: AllocationRecorder,
    ) -> tuple[ModelProfile, dict[str, Any]]:
        profile = configured_profiles.get(model)
        if profile is not None:
            return profile, {}

        dynamic_config = load_dynamic_models_config(self.config_path)
        if not dynamic_model_allowed(model, dynamic_config):
            record_allocation(model, "denied")
            raise PermissionError(f"Model {model} is not allowed by dynamic model policy.")

        profile = load_dynamic_model_profile(self.config_path, model, overrides)
        if profile is None:
            record_allocation(model, "not_configured")
            raise LookupError(f"Model {model} is not configured and dynamic models are disabled.")

        return profile, dynamic_config
