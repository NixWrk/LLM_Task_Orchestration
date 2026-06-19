from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import httpx

from lifecycle.adapters import estimate_lmstudio_load_vram_mb
from lifecycle.allocation import profile_with_context_plan
from lifecycle.allocation_service import AllocationService
from lifecycle.cleanup import CleanupService, idle_seconds
from lifecycle.config import (
    load_dynamic_model_profile,
    load_dynamic_models_config,
    load_model_profiles,
)
from lifecycle.dynamic_policy import dynamic_model_allowed
from lifecycle.lmstudio import LmStudioLoad, inspect_loaded_model, loaded_models
from lifecycle.models import (
    BackendInstance,
    ContextPlan,
    GpuState,
    ModelProfile,
    PlacementDecision,
    optional_int,
    parse_iso,
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
            effective_profile = profile_with_lmstudio_estimate(
                profile_with_context_plan(profile, context_plan)
            )
            ready_count = len(self.registry.ready_for_model(profile.public_name))
            active_instances = [
                instance_with_live_lmstudio_shape(effective_profile, instance)
                for instance in self.registry.active_for_model(profile.public_name)
            ]
            active_count = len(active_instances)
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
                        active_instances,
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

    async def explain_plan(
        self,
        queue_lengths: dict[str, int] | None = None,
        context_plans: dict[str, ContextPlan] | None = None,
    ) -> dict[str, Any]:
        queue_lengths = queue_lengths or {}
        context_plans = context_plans or {}
        plan = await self.plan(queue_lengths, context_plans)
        model_explanations = [
            explain_model_plan(
                model_plan,
                queue_length=max(
                    queue_lengths.get(str(model_plan.get("model") or ""), 0),
                    int((model_plan.get("context_plan") or {}).get("queued_tasks") or 0),
                ),
            )
            for model_plan in plan["models"]
        ]
        return {
            "dry_run": plan["dry_run"],
            "gpu_count": plan["gpu_count"],
            "planning_inputs": {
                "queue_lengths": queue_lengths,
                "context_plans": {
                    model: context_plan.to_dict()
                    for model, context_plan in context_plans.items()
                },
            },
            "explanations": model_explanations,
            "plan": plan,
            "summary": summarize_plan_explanations(model_explanations),
        }

    async def reconcile(
        self,
        queue_lengths: dict[str, int] | None = None,
        context_plans: dict[str, ContextPlan] | None = None,
    ) -> dict[str, Any]:
        queue_lengths = queue_lengths or {}
        context_plans = context_plans or {}
        profiles = self.profiles()
        live_reconciled = self.sync_live_lmstudio_state(profiles)
        stopped = await self.stop_idle_instances(profiles, queue_lengths)
        plan = await self.plan(queue_lengths, context_plans)
        created: list[dict[str, Any]] = []
        reloaded: list[dict[str, Any]] = []

        for model_plan in plan["models"]:
            profile = profile_with_context_plan(
                profiles[model_plan["model"]],
                context_plans.get(model_plan["model"]),
            )
            profile = profile_with_lmstudio_estimate(profile)
            gpus = {gpu.id: gpu for gpu in await self.gpu_states()}
            for decision_payload in model_plan["decisions"]:
                if decision_payload["action"] == "reload":
                    reloaded.append(await self.reload_instance(profile, decision_payload, gpus))
                    continue
                if decision_payload["action"] != "start" or not decision_payload["gpu_id"]:
                    continue
                instance = self.start_instance(profile, decision_payload, gpus)
                self.registry.upsert(instance)
                instance = await self.initialize_instance(profile, instance)
                created.append(instance.to_dict())

        plan["created_instances"] = created
        plan["reloaded_instances"] = reloaded
        plan["stopped_instances"] = stopped
        plan["live_lmstudio_reconciled_instances"] = live_reconciled
        return plan

    def sync_live_lmstudio_state(
        self,
        profiles: dict[str, ModelProfile],
    ) -> list[dict[str, Any]]:
        loads_by_binary: dict[str, list[LmStudioLoad]] = {}
        synced: list[dict[str, Any]] = []

        for profile in profiles.values():
            if profile.runtime != "lmstudio":
                continue
            loads = loads_by_binary.setdefault(
                profile.lms_binary,
                loaded_models(profile.lms_binary),
            )
            synced.extend(self.sync_profile_live_lmstudio_load(profile, loads))

        return synced

    def sync_profile_live_lmstudio_load(
        self,
        profile: ModelProfile,
        loads: list[LmStudioLoad],
    ) -> list[dict[str, Any]]:
        synced: list[dict[str, Any]] = []
        matched_loads: set[str] = set()
        profile_loads = [
            load for load in loads if lmstudio_load_matches_profile(load, profile)
        ]

        for instance in list(self.registry.list()):
            if instance.runtime != "lmstudio" or instance.model != profile.public_name:
                continue
            load = matching_lmstudio_load(instance, profile_loads)
            if load is None:
                if instance.state == "external":
                    self.registry.remove(instance.instance_id)
                continue
            matched_loads.add(load.identifier)
            updated = instance_with_lmstudio_load(profile, instance, load)
            synced.append(self.registry.upsert(updated).to_dict())

        for load in profile_loads:
            if load.identifier in matched_loads:
                continue
            external = external_lmstudio_instance(profile, load)
            synced.append(self.registry.upsert(external).to_dict())

        return synced

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

    async def reload_instance(
        self,
        profile: ModelProfile,
        decision_payload: dict[str, Any],
        gpu_by_id: dict[str, GpuState],
    ) -> dict[str, Any]:
        instance_id = str(decision_payload.get("instance_id") or "")
        if not instance_id:
            return {"state": "skipped", "reason": "missing_instance_id"}

        instance = self.registry.get(instance_id)
        instance = instance_with_live_lmstudio_shape(profile, instance)
        if instance.active_requests > 0:
            draining = (
                instance
                if instance.state == "draining"
                else self.registry.mark_state(instance.instance_id, "draining")
            )
            return {
                "state": "draining",
                "reason": "active_requests_present",
                "instance": draining.to_dict(),
            }

        if (
            instance.runtime == "lmstudio"
            and not instance.dry_run
            and not instance.metadata.get("lmstudio_loaded_with_lms")
        ):
            return {
                "state": "skipped",
                "reason": "lmstudio_load_not_owned",
                "instance": instance.to_dict(),
            }

        stopped = await self.stop_instance(profile, instance)
        replacement = self.start_instance(profile, decision_payload, gpu_by_id)
        self.registry.upsert(replacement)
        replacement = await self.initialize_instance(profile, replacement)
        return {
            "state": "reloaded",
            "stopped_instance": stopped,
            "instance": replacement.to_dict(),
        }


def summarize_plan_explanations(explanations: list[dict[str, Any]]) -> dict[str, Any]:
    waiting = [
        explanation["model"]
        for explanation in explanations
        if explanation.get("waiting")
    ]
    blocked = [
        explanation["model"]
        for explanation in explanations
        if str(explanation.get("status", "")).startswith("blocked")
    ]
    return {
        "models": len(explanations),
        "waiting_models": waiting,
        "blocked_models": blocked,
    }


def explain_model_plan(
    model_plan: dict[str, Any],
    *,
    queue_length: int,
) -> dict[str, Any]:
    model = str(model_plan.get("model") or "")
    decisions = [
        decision
        for decision in model_plan.get("decisions", [])
        if isinstance(decision, dict)
    ]
    reasons = [explain_decision_reason(decision) for decision in decisions]
    next_actions = [explain_next_action(decision) for decision in decisions]
    status = explain_status(model_plan, decisions, queue_length)
    return {
        "model": model,
        "status": status,
        "waiting": status
        in {
            "blocked_oversized_task",
            "waiting_for_gpu",
            "starting_backend",
            "reload_required",
            "waiting_for_capacity",
        },
        "queue_length": queue_length,
        "ready_replicas": int(model_plan.get("ready_replicas") or 0),
        "active_replicas": int(model_plan.get("active_replicas") or 0),
        "desired_replicas": int(model_plan.get("desired_replicas") or 0),
        "desired_backend_shape": model_plan.get("desired_backend_shape"),
        "context_plan": model_plan.get("context_plan"),
        "reasons": reasons,
        "next_actions": next_actions,
    }


def explain_status(
    model_plan: dict[str, Any],
    decisions: list[dict[str, Any]],
    queue_length: int,
) -> str:
    decision = decisions[0] if decisions else {}
    action = str(decision.get("action") or "")
    reason = str(decision.get("reason") or "")
    ready_replicas = int(model_plan.get("ready_replicas") or 0)
    active_replicas = int(model_plan.get("active_replicas") or 0)

    if action == "reject_oversized":
        return "blocked_oversized_task"
    if reason == "no_gpu_with_enough_vram":
        return "waiting_for_gpu"
    if action == "start":
        return "starting_backend"
    if action == "reload":
        return "reload_required"
    if queue_length > 0 and ready_replicas > 0:
        if reason == "reload_hysteresis_min_dwell":
            return "capacity_degraded"
        return "ready"
    if queue_length > 0 and active_replicas > 0:
        return "waiting_for_capacity"
    if queue_length > 0:
        return "waiting_for_capacity"
    if active_replicas > 0:
        return "idle_ready"
    return "idle"


def explain_decision_reason(decision: dict[str, Any]) -> dict[str, Any]:
    reason = str(decision.get("reason") or "unknown")
    details: dict[str, Any] = {
        "type": reason,
        "message": DECISION_REASON_MESSAGES.get(reason, "See raw lifecycle decision."),
    }
    for key in (
        "action",
        "gpu_id",
        "instance_id",
        "required_vram_mb",
        "available_vram_mb",
        "lms_context_length",
        "lms_parallel",
        "current_lms_context_length",
        "current_lms_parallel",
    ):
        if decision.get(key) is not None:
            details[key] = decision[key]
    context_plan = decision.get("context_plan")
    if isinstance(context_plan, dict) and context_plan.get("oversized_tasks"):
        details["oversized_tasks"] = context_plan["oversized_tasks"]
    return details


def explain_next_action(decision: dict[str, Any]) -> dict[str, Any]:
    action = str(decision.get("action") or "noop")
    reason = str(decision.get("reason") or "")
    if action == "start":
        return {
            "type": "start_backend",
            "message": "Lifecycle can place and start a backend for this queue.",
            "gpu_id": decision.get("gpu_id"),
        }
    if action == "reload":
        return {
            "type": "drain_and_reload",
            "message": "Lifecycle should drain the backend and reload it with the planned LM Studio shape.",
            "instance_id": decision.get("instance_id"),
        }
    if action == "reject_oversized":
        return {
            "type": "reject_or_split_tasks",
            "message": "At least one task is larger than the model context cap.",
        }
    if reason == "no_gpu_with_enough_vram":
        return {
            "type": "wait_for_gpu_capacity",
            "message": "Free or add GPU memory, reduce the requested shape, or wait for cleanup.",
        }
    if reason == "reload_hysteresis_min_dwell":
        return {
            "type": "wait_for_reload_dwell",
            "message": "Keep using the current load until the minimum dwell window allows reload.",
        }
    return {
        "type": "none",
        "message": "No lifecycle action is required right now.",
    }


DECISION_REASON_MESSAGES = {
    "desired_replicas_satisfied": "The current active backend count satisfies policy.",
    "vram_available": "A GPU has enough free memory after current reservations.",
    "no_gpu_with_enough_vram": "No allowed GPU has enough free memory after current reservations.",
    "context_plan_has_oversized_tasks": "One or more queued tasks exceed the model context cap.",
    "backend_context_too_small_for_task": "The live LM Studio context is too small for at least one queued task.",
    "backend_context_unknown": "Lifecycle cannot read the live LM Studio context length.",
    "backend_parallel_unknown": "Lifecycle cannot read the live LM Studio parallel slot count.",
    "backend_context_below_desired_bucket": "The live LM Studio context is below the desired context bucket.",
    "backend_parallel_too_small": "The live LM Studio parallel slot count is below the desired shape.",
    "current_context_satisfies_required_tokens": "The live context fits the queued tasks, so bucket-only reload is skipped.",
    "reload_hysteresis_min_dwell": "Reload is delayed by the model profile minimum dwell window.",
}


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
    active_instances: list[BackendInstance],
    context_plan: ContextPlan | None,
) -> PlacementDecision | None:
    if context_plan is None or profile.runtime != "lmstudio":
        return None

    for instance in active_instances:
        if instance.state not in {"ready", "draining"}:
            continue
        assessment = assess_reload_need(instance, profile, context_plan)
        if not assessment.reload_needed:
            if assessment.noop_reason:
                return noop_shape_decision(profile, instance, context_plan, assessment)
            continue
        if not assessment.hard_context_mismatch and not reload_dwell_satisfied(profile, instance):
            return noop_shape_decision(
                profile,
                instance,
                context_plan,
                replace(
                    assessment,
                    noop_reason="reload_hysteresis_min_dwell",
                ),
            )
        current_context = lms_context_length_from_instance(instance)
        current_parallel = lms_parallel_from_instance(instance)
        return PlacementDecision(
            model=profile.public_name,
            action="reload",
            gpu_id=instance.gpu_ids[0] if instance.gpu_ids else None,
            reason=assessment.reload_reason or "backend_shape_mismatch",
            required_vram_mb=profile.estimated_vram_mb + profile.safety_margin_mb,
            instance_id=instance.instance_id,
            lms_context_length=profile.lms_context_length,
            lms_parallel=profile.lms_parallel,
            current_lms_context_length=current_context,
            current_lms_parallel=current_parallel,
            context_plan=context_plan.to_dict(),
        )

    return None


@dataclass(frozen=True)
class ReloadAssessment:
    reload_needed: bool
    hard_context_mismatch: bool = False
    reload_reason: str | None = None
    noop_reason: str | None = None


def assess_reload_need(
    instance: BackendInstance,
    profile: ModelProfile,
    context_plan: ContextPlan,
) -> ReloadAssessment:
    current_context = lms_context_length_from_instance(instance)
    current_parallel = lms_parallel_from_instance(instance)
    required_context = context_plan.max_required_context_tokens
    desired_context = profile.lms_context_length
    desired_parallel = profile.lms_parallel

    if desired_context and current_context is None:
        return ReloadAssessment(
            reload_needed=True,
            hard_context_mismatch=True,
            reload_reason="backend_context_unknown",
        )
    if desired_parallel and current_parallel is None:
        return ReloadAssessment(
            reload_needed=True,
            reload_reason="backend_parallel_unknown",
        )
    if current_context is not None and required_context > 0 and current_context < required_context:
        return ReloadAssessment(
            reload_needed=True,
            hard_context_mismatch=True,
            reload_reason="backend_context_too_small_for_task",
        )
    if (
        desired_context
        and current_context is not None
        and current_context < desired_context
        and not profile.reload_allow_bucket_only
    ):
        return ReloadAssessment(
            reload_needed=False,
            noop_reason="current_context_satisfies_required_tokens",
        )
    if desired_context and current_context is not None and current_context < desired_context:
        return ReloadAssessment(
            reload_needed=True,
            reload_reason="backend_context_below_desired_bucket",
        )
    if desired_parallel and current_parallel is not None and current_parallel < desired_parallel:
        return ReloadAssessment(
            reload_needed=True,
            reload_reason="backend_parallel_too_small",
        )
    return ReloadAssessment(reload_needed=False)


def noop_shape_decision(
    profile: ModelProfile,
    instance: BackendInstance,
    context_plan: ContextPlan,
    assessment: ReloadAssessment,
) -> PlacementDecision:
    return PlacementDecision(
        model=profile.public_name,
        action="noop",
        gpu_id=instance.gpu_ids[0] if instance.gpu_ids else None,
        reason=assessment.noop_reason or "desired_replicas_satisfied",
        required_vram_mb=profile.estimated_vram_mb + profile.safety_margin_mb,
        instance_id=instance.instance_id,
        lms_context_length=profile.lms_context_length,
        lms_parallel=profile.lms_parallel,
        current_lms_context_length=lms_context_length_from_instance(instance),
        current_lms_parallel=lms_parallel_from_instance(instance),
        context_plan=context_plan.to_dict(),
    )


def reload_dwell_satisfied(profile: ModelProfile, instance: BackendInstance) -> bool:
    if profile.reload_min_dwell_seconds <= 0:
        return True
    try:
        created_at = parse_iso(instance.created_at).astimezone(UTC)
    except Exception:
        return True
    age_seconds = (datetime.now(UTC) - created_at).total_seconds()
    return age_seconds >= profile.reload_min_dwell_seconds


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


def profile_with_lmstudio_estimate(profile: ModelProfile) -> ModelProfile:
    if profile.runtime != "lmstudio":
        return profile
    try:
        estimated_vram_mb = estimate_lmstudio_load_vram_mb(profile)
    except Exception:
        return profile
    if not estimated_vram_mb:
        return profile
    return replace(profile, estimated_vram_mb=estimated_vram_mb)


def instance_with_live_lmstudio_shape(
    profile: ModelProfile,
    instance: BackendInstance,
) -> BackendInstance:
    if profile.runtime != "lmstudio":
        return instance
    try:
        loaded = inspect_loaded_model(
            instance.backend_model,
            str(instance.metadata.get("lms_binary") or profile.lms_binary),
        )
    except Exception:
        return instance
    if loaded is None:
        return instance

    return instance_with_lmstudio_load(profile, instance, loaded)


def instance_with_lmstudio_load(
    profile: ModelProfile,
    instance: BackendInstance,
    loaded: LmStudioLoad,
) -> BackendInstance:
    copy = BackendInstance.from_dict(instance.to_dict())
    copy.metadata = dict(copy.metadata)
    copy.metadata["live_lmstudio_load"] = loaded.to_dict()
    copy.metadata["live_lmstudio_reconciled_at"] = datetime.now(UTC).isoformat()
    copy.metadata["lmstudio_identifier"] = loaded.identifier
    copy.metadata["lmstudio_ownership"] = (
        "owned" if copy.metadata.get("lmstudio_loaded_with_lms") else "external"
    )
    if loaded.context_length is not None:
        copy.metadata["lms_context_length"] = loaded.context_length
    if loaded.parallel is not None:
        copy.metadata["lms_parallel"] = loaded.parallel
    if loaded.gpu is not None:
        copy.metadata["lms_gpu"] = loaded.gpu
    if loaded.ttl_seconds is not None:
        copy.metadata["lms_ttl_seconds"] = loaded.ttl_seconds
    return copy


def lmstudio_load_matches_profile(load: LmStudioLoad, profile: ModelProfile) -> bool:
    candidates = {load.identifier, load.model_key}
    raw = load.raw
    for key in ("model", "modelKey", "selectedVariant", "indexedModelIdentifier"):
        if raw.get(key) is not None:
            candidates.add(str(raw[key]))
    return profile.backend_model in candidates or profile.public_name in candidates


def matching_lmstudio_load(
    instance: BackendInstance,
    loads: list[LmStudioLoad],
) -> LmStudioLoad | None:
    preferred_identifier = str(instance.metadata.get("lmstudio_identifier") or "")
    candidates = {instance.backend_model, instance.model}
    if preferred_identifier:
        candidates.add(preferred_identifier)
    for load in loads:
        if load.identifier in candidates or load.model_key in candidates:
            return load
    return None


def external_lmstudio_instance(
    profile: ModelProfile,
    load: LmStudioLoad,
) -> BackendInstance:
    instance = BackendInstance(
        instance_id=f"external-lmstudio-{stable_id(load.identifier)}",
        model=profile.public_name,
        backend_model=profile.backend_model,
        runtime="lmstudio",
        base_url=profile.base_url or "",
        gpu_ids=gpu_ids_from_lmstudio_load(load, profile),
        state="external",
        reserved_vram_mb=profile.estimated_vram_mb + profile.safety_margin_mb,
        dry_run=False,
        metadata={
            "lmstudio_identifier": load.identifier,
            "lmstudio_ownership": "external",
            "lmstudio_loaded_with_lms": False,
            "lms_binary": profile.lms_binary,
        },
    )
    return instance_with_lmstudio_load(profile, instance, load)


def gpu_ids_from_lmstudio_load(load: LmStudioLoad, profile: ModelProfile) -> list[str]:
    raw_gpu = (load.gpu or "").strip().lower()
    if raw_gpu.startswith("gpu"):
        return [raw_gpu]
    if raw_gpu.isdigit():
        return [f"gpu{raw_gpu}"]
    explicit = [
        gpu_id
        for gpu_id in profile.preferred_gpus
        if gpu_id not in {"auto", "max"} and gpu_id.startswith("gpu")
    ]
    return list(explicit[:1])


def stable_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{cleaned[:48] or 'load'}-{digest}"
