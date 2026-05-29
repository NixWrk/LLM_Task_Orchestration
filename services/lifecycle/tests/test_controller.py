from datetime import UTC, datetime, timedelta
import asyncio
from pathlib import Path

from lifecycle.controller import LifecycleController, idle_seconds, openai_url
from lifecycle.models import BackendInstance, GpuState
from lifecycle.registry import BackendRegistry


def test_openai_url_does_not_duplicate_v1_prefix() -> None:
    assert openai_url("http://backend:8000/v1", "/v1/models") == (
        "http://backend:8000/v1/models"
    )
    assert openai_url("http://backend:8000/v1", "/chat/completions") == (
        "http://backend:8000/v1/chat/completions"
    )


def test_idle_seconds_uses_last_used_at_when_available() -> None:
    last_used = datetime.now(UTC) - timedelta(seconds=30)
    instance = BackendInstance(
        instance_id="backend-1",
        model="local-main",
        backend_model="local-main",
        runtime="vllm",
        base_url="http://backend:8000/v1",
        gpu_ids=["gpu0"],
        state="ready",
        reserved_vram_mb=1024,
        last_used_at=last_used.isoformat(),
    )

    assert idle_seconds(instance) >= 29


def test_allocate_dynamic_lmstudio_model_registers_ready_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "orchestrator.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                "  max_active_requests: 1",
                "dynamic_models:",
                "  enabled: true",
                "  lifecycle:",
                "    base_url: http://host.docker.internal:1234/v1",
                "    estimated_vram_gb: 8",
                "    safety_margin_gb: 1",
            ]
        ),
        encoding="utf-8",
    )
    registry = BackendRegistry(str(tmp_path / "registry.json"))
    controller = LifecycleController(
        config_path=str(config_path),
        registry=registry,
        gpu_inventory_url="http://gpu-inventory:4200",
        request_timeout_seconds=1,
        dry_run=True,
    )

    async def fake_gpu_states() -> list[GpuState]:
        return [GpuState("gpu0", 0, "gpu", 24_000, 1_000, 23_000)]

    async def fake_verify_model_available(_profile) -> None:
        return None

    async def fake_initialize(_profile, instance: BackendInstance) -> BackendInstance:
        return registry.mark_state(instance.instance_id, "ready")

    monkeypatch.setattr(controller, "gpu_states", fake_gpu_states)
    monkeypatch.setattr(controller, "verify_model_available", fake_verify_model_available)
    monkeypatch.setattr(controller, "initialize_instance", fake_initialize)

    result = asyncio.run(controller.allocate({"model": "qwen/qwen3.5-9b"}))

    assert result["created"] is True
    assert result["instance"]["model"] == "qwen/qwen3.5-9b"
    assert result["instance"]["base_url"] == "http://host.docker.internal:1234/v1"
    assert result["instance"]["state"] == "ready"
