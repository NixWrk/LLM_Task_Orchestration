from pathlib import Path

from lifecycle.models import BackendInstance
from lifecycle.registry import BackendRegistry


def test_adjust_active_requests_tracks_leases(tmp_path: Path) -> None:
    registry = BackendRegistry(str(tmp_path / "registry.json"))
    registry.upsert(
        BackendInstance(
            instance_id="backend-1",
            model="local-main",
            backend_model="local-main",
            runtime="vllm",
            base_url="http://backend:8000/v1",
            gpu_ids=["gpu0"],
            state="ready",
            reserved_vram_mb=1024,
        )
    )

    leased = registry.adjust_active_requests("backend-1", 1)
    leased_count = leased.active_requests
    leased_last_used_at = leased.last_used_at
    released = registry.adjust_active_requests("backend-1", -1)
    released_count = released.active_requests
    extra_release = registry.adjust_active_requests("backend-1", -1)

    assert leased_count == 1
    assert leased_last_used_at is not None
    assert released_count == 0
    assert extra_release.active_requests == 0
