from __future__ import annotations

from dataclasses import replace
from typing import Any

from lifecycle.lmstudio import (
    compact_metadata as compact_lmstudio_metadata,
    estimate_vram_mb as estimate_vram_mb_from_lmstudio_metadata,
    metadata_for_model as lmstudio_metadata_for_model,
)
from lifecycle.models import ContextPlan, ModelProfile


def queue_lengths_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("queue_lengths", {})
    if not isinstance(raw, dict):
        return {}
    return {str(model): int(length) for model, length in raw.items()}


def context_plans_from_payload(payload: dict[str, Any]) -> dict[str, ContextPlan]:
    raw = payload.get("context_plans", {})
    if not isinstance(raw, dict):
        return {}
    plans: dict[str, ContextPlan] = {}
    for model, raw_plan in raw.items():
        if isinstance(raw_plan, dict):
            plans[str(model)] = ContextPlan.from_dict(raw_plan)
    return plans


def profile_with_context_plan(
    profile: ModelProfile,
    context_plan: ContextPlan | None,
) -> ModelProfile:
    if context_plan is None or context_plan.queued_tasks <= 0:
        return profile

    overrides: dict[str, Any] = {}
    if context_plan.recommended_lms_context_length:
        overrides["lms_context_length"] = context_plan.recommended_lms_context_length
    if context_plan.recommended_lms_parallel:
        overrides["lms_parallel"] = context_plan.recommended_lms_parallel
    if not overrides:
        return profile
    return replace(profile, **overrides)


def allocation_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    orchestration = payload.get("orchestration")
    if not isinstance(orchestration, dict):
        orchestration = {}

    lifecycle: dict[str, Any] = {}
    for source_key, target_key in (
        ("runtime", "runtime"),
        ("base_url", "base_url"),
        ("estimated_vram_gb", "estimated_vram_gb"),
        ("safety_margin_gb", "safety_margin_gb"),
        ("preferred_gpus", "preferred_gpus"),
        ("idle_ttl_seconds", "idle_ttl_seconds"),
        ("min_replicas", "min_replicas"),
        ("max_replicas", "max_replicas"),
        ("warmup_enabled", "warmup_enabled"),
        ("warmup_prompt", "warmup_prompt"),
        ("warmup_max_tokens", "warmup_max_tokens"),
        ("startup_timeout_seconds", "startup_timeout_seconds"),
        ("load_strategy", "load_strategy"),
        ("lms_gpu", "lms_gpu"),
        ("lms_context_length", "lms_context_length"),
        ("lms_parallel", "lms_parallel"),
        ("lms_ttl_seconds", "lms_ttl_seconds"),
    ):
        if source_key in orchestration:
            lifecycle[target_key] = orchestration[source_key]

    if "max_parallel" in orchestration and "lms_parallel" not in lifecycle:
        lifecycle["lms_parallel"] = orchestration["max_parallel"]

    if "gpu" in orchestration and "preferred_gpus" not in lifecycle:
        gpu = orchestration["gpu"]
        lifecycle["preferred_gpus"] = gpu if isinstance(gpu, list) else [gpu]

    return {"lifecycle": lifecycle} if lifecycle else {}


def allocation_has_vram_override(payload: dict[str, Any]) -> bool:
    orchestration = payload.get("orchestration")
    return isinstance(orchestration, dict) and "estimated_vram_gb" in orchestration


def enrich_profile_from_lmstudio_metadata(
    profile: ModelProfile,
    dynamic_config: dict[str, Any],
    allocation_payload: dict[str, Any],
) -> tuple[ModelProfile, dict[str, Any]]:
    if profile.runtime != "lmstudio":
        return profile, {}
    if allocation_has_vram_override(allocation_payload):
        return profile, {}
    if not bool(dynamic_config.get("auto_vram_from_lms", True)):
        return profile, {}

    metadata = lmstudio_metadata_for_model(
        profile.backend_model,
        str(dynamic_config.get("lms_binary") or profile.lms_binary),
    )
    if not metadata:
        return profile, {}

    estimated_vram_mb = estimate_vram_mb_from_lmstudio_metadata(
        metadata,
        fallback_mb=profile.estimated_vram_mb,
    )
    if estimated_vram_mb <= 0:
        return profile, {"lmstudio_metadata": metadata}
    return (
        replace(profile, estimated_vram_mb=estimated_vram_mb),
        {
            "lmstudio_metadata": compact_lmstudio_metadata(metadata),
            "auto_estimated_vram_mb": estimated_vram_mb,
        },
    )
