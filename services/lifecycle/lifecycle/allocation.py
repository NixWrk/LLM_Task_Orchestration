from __future__ import annotations

from typing import Any


def queue_lengths_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("queue_lengths", {})
    if not isinstance(raw, dict):
        return {}
    return {str(model): int(length) for model, length in raw.items()}


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
