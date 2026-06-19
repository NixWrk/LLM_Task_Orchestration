from __future__ import annotations

from lifecycle.models import GpuState, ModelProfile, PlacementDecision
from lifecycle.registry import BackendRegistry


def choose_gpu(
    profile: ModelProfile,
    gpus: list[GpuState],
    registry: BackendRegistry,
) -> PlacementDecision:
    required_vram_mb = profile.estimated_vram_mb + profile.safety_margin_mb
    preferred = set(profile.preferred_gpus)
    reserved = registry.reserved_vram_by_gpu()

    candidates: list[tuple[GpuState, int]] = []
    for gpu in gpus:
        if "auto" not in preferred and gpu.id not in preferred and str(gpu.index) not in preferred:
            continue
        available = gpu.memory_free_mb - reserved.get(gpu.id, 0)
        if available >= required_vram_mb:
            candidates.append((gpu, available))

    if not candidates:
        best_available = max(
            (gpu.memory_free_mb - reserved.get(gpu.id, 0) for gpu in gpus),
            default=None,
        )
        return PlacementDecision(
            model=profile.public_name,
            action="noop",
            gpu_id=None,
            reason="no_gpu_with_enough_vram",
            required_vram_mb=required_vram_mb,
            available_vram_mb=best_available,
            lms_context_length=profile.lms_context_length,
            lms_parallel=profile.lms_parallel,
        )

    selected_gpu, available = max(candidates, key=lambda item: item[1])
    return PlacementDecision(
        model=profile.public_name,
        action="start",
        gpu_id=selected_gpu.id,
        reason="vram_available",
        required_vram_mb=required_vram_mb,
        available_vram_mb=available,
        lms_context_length=profile.lms_context_length,
        lms_parallel=profile.lms_parallel,
    )


def desired_replicas(profile: ModelProfile, ready_replicas: int, queue_length: int = 0) -> int:
    desired = max(profile.min_replicas, ready_replicas)
    if queue_length > 0:
        desired += 1
    return min(profile.max_replicas, desired)
