from datetime import UTC, datetime, timedelta
import asyncio
from pathlib import Path

from lifecycle.allocation import allocation_overrides, enrich_profile_from_lmstudio_metadata
from lifecycle.controller import (
    LifecycleController,
    idle_seconds,
)
from lifecycle.dynamic_policy import dynamic_model_allowed
from lifecycle.lmstudio import estimate_vram_mb as estimate_vram_mb_from_lmstudio_metadata
from lifecycle.models import BackendInstance, GpuState, ModelProfile
from lifecycle.registry import BackendRegistry
from lifecycle.runtime import RuntimeLifecycleService, should_verify_before_start
from orchestrator_core.openai import openai_url


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
                "  auto_vram_from_lms: false",
                "  lifecycle:",
                "    base_url: http://host.docker.internal:1234/v1",
                "    load_strategy: none",
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


def test_dynamic_model_allowed_honors_allow_and_deny_patterns() -> None:
    config = {
        "enabled": True,
        "allowed_model_patterns": ["qwen*", "google/gemma-4-e2b"],
        "denied_model_patterns": ["*embedding*"],
    }

    assert dynamic_model_allowed("qwen/qwen3.5-9b", config) is True
    assert dynamic_model_allowed("google/gemma-4-e2b", config) is True
    assert dynamic_model_allowed("text-embedding-bge-m3", config) is False
    assert dynamic_model_allowed("mistralai/ministral-3-3b", config) is False


def test_allocation_overrides_map_max_parallel_to_lmstudio_parallel() -> None:
    overrides = allocation_overrides(
        {
            "model": "qwen",
            "orchestration": {
                "max_parallel": 2,
                "lms_gpu": "max",
                "lms_context_length": 8192,
                "startup_timeout_seconds": 1800,
            },
        }
    )

    lifecycle = overrides["lifecycle"]
    assert lifecycle["lms_parallel"] == 2
    assert lifecycle["lms_gpu"] == "max"
    assert lifecycle["lms_context_length"] == 8192
    assert lifecycle["startup_timeout_seconds"] == 1800


def test_allocate_dynamic_model_denied_by_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dynamic_models:",
                "  enabled: true",
                "  allowed_model_patterns: [qwen*]",
                "  denied_model_patterns: []",
            ]
        ),
        encoding="utf-8",
    )
    controller = LifecycleController(
        config_path=str(config_path),
        registry=BackendRegistry(str(tmp_path / "registry.json")),
        gpu_inventory_url="http://gpu-inventory:4200",
        request_timeout_seconds=1,
        dry_run=True,
    )

    try:
        asyncio.run(controller.allocate({"model": "mistralai/ministral-3-3b"}))
    except PermissionError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("Expected PermissionError")


def test_estimate_vram_from_lmstudio_metadata_uses_size_and_context() -> None:
    estimate = estimate_vram_mb_from_lmstudio_metadata(
        {
            "sizeBytes": 8 * 1024 * 1024 * 1024,
            "maxContextLength": 32768,
        },
        fallback_mb=1024,
    )

    assert estimate > 8 * 1024


def test_estimate_vram_from_lmstudio_metadata_can_replace_large_fallback() -> None:
    estimate = estimate_vram_mb_from_lmstudio_metadata(
        {
            "sizeBytes": 512 * 1024 * 1024,
            "maxContextLength": 8192,
        },
        fallback_mb=8 * 1024,
    )

    assert estimate < 8 * 1024
    assert estimate > 512


def test_lmstudio_metadata_enrichment_replaces_profile_vram(monkeypatch) -> None:
    monkeypatch.setattr(
        "lifecycle.allocation.lmstudio_metadata_for_model",
        lambda _model, _binary: {
            "modelKey": "qwen",
            "sizeBytes": 512 * 1024 * 1024,
            "maxContextLength": 8192,
        },
    )

    enriched, metadata = enrich_profile_from_lmstudio_metadata(
        lmstudio_profile(),
        {"auto_vram_from_lms": True, "lms_binary": "lms"},
        {"model": "qwen"},
    )

    assert enriched.estimated_vram_mb < 8 * 1024
    assert metadata["auto_estimated_vram_mb"] == enriched.estimated_vram_mb


def test_lmstudio_cli_load_strategy_skips_pre_start_openai_check() -> None:
    profile = lmstudio_profile(load_strategy="cli-if-available")

    assert should_verify_before_start(profile) is False
    assert should_verify_before_start(
        ModelProfile(**{**profile.__dict__, "load_strategy": "none"})
    ) is True


def test_initialize_failure_stops_loaded_lmstudio_instance(tmp_path: Path, monkeypatch) -> None:
    registry = BackendRegistry(str(tmp_path / "registry.json"))
    instance = BackendInstance(
        instance_id="lmstudio-1",
        model="qwen",
        backend_model="qwen",
        runtime="lmstudio",
        base_url="http://host.docker.internal:1234/v1",
        gpu_ids=["gpu0"],
        state="starting",
        reserved_vram_mb=1024,
        dry_run=False,
        metadata={"lmstudio_loaded_with_lms": True},
    )
    registry.upsert(instance)
    runtime = RuntimeLifecycleService(
        registry=registry,
        request_timeout_seconds=1,
        dry_run=False,
    )
    stop_calls: list[str] = []

    async def fake_wait_for_health(_profile, _instance) -> None:
        raise RuntimeError("boom")

    class FakeAdapter:
        def stop(self, stopped_instance: BackendInstance) -> None:
            stop_calls.append(stopped_instance.instance_id)

    monkeypatch.setattr(runtime, "wait_for_health", fake_wait_for_health)
    monkeypatch.setattr(
        "lifecycle.runtime.adapter_for",
        lambda *_args, **_kwargs: FakeAdapter(),
    )

    failed = asyncio.run(runtime.initialize_instance(lmstudio_profile(), instance))

    assert failed.state == "failed"
    assert stop_calls == ["lmstudio-1"]


def lmstudio_profile(load_strategy: str = "cli-if-available") -> ModelProfile:
    return ModelProfile(
        public_name="qwen",
        backend_model="qwen",
        runtime="lmstudio",
        artifact=None,
        runtime_image=None,
        host_port_start=8100,
        container_port=8000,
        public_host="host.docker.internal",
        base_url="http://host.docker.internal:1234/v1",
        docker_extra_args=(),
        runtime_extra_args=(),
        volume_mounts=(),
        environment=(),
        healthcheck_path="/v1/models",
        startup_timeout_seconds=120,
        healthcheck_interval_seconds=2,
        warmup_enabled=True,
        warmup_prompt="Return exactly: ok",
        warmup_max_tokens=8,
        estimated_vram_mb=8 * 1024,
        safety_margin_mb=1024,
        min_replicas=0,
        max_replicas=1,
        idle_ttl_seconds=900,
        load_strategy=load_strategy,
    )


def test_cleanup_stops_idle_dynamic_instance(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dynamic_models:",
                "  enabled: true",
                "  lifecycle:",
                "    base_url: http://host.docker.internal:1234/v1",
                "    idle_ttl_seconds: 1",
            ]
        ),
        encoding="utf-8",
    )
    registry = BackendRegistry(str(tmp_path / "registry.json"))
    old = datetime.now(UTC) - timedelta(seconds=30)
    registry.upsert(
        BackendInstance(
            instance_id="old",
            model="qwen/qwen3.5-9b",
            backend_model="qwen/qwen3.5-9b",
            runtime="lmstudio",
            base_url="http://host.docker.internal:1234/v1",
            gpu_ids=["gpu0"],
            state="ready",
            reserved_vram_mb=1024,
            last_used_at=old.isoformat(),
            dry_run=False,
            metadata={"idle_ttl_seconds": 1},
        )
    )
    controller = LifecycleController(
        config_path=str(config_path),
        registry=registry,
        gpu_inventory_url="http://gpu-inventory:4200",
        request_timeout_seconds=1,
        dry_run=True,
    )

    result = asyncio.run(controller.cleanup({}))

    assert result["stopped_instances"][0]["state"] == "stopped"


def test_cleanup_removes_stale_lmstudio_allocations(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dynamic_models:",
                "  enabled: true",
                "  registry_cleanup_ttl_seconds: 1",
                "  lifecycle:",
                "    base_url: http://host.docker.internal:1234/v1",
            ]
        ),
        encoding="utf-8",
    )
    registry = BackendRegistry(str(tmp_path / "registry.json"))
    old = datetime.now(UTC) - timedelta(seconds=30)
    registry.upsert(
        BackendInstance(
            instance_id="stale",
            model="qwen/qwen3.5-9b",
            backend_model="qwen/qwen3.5-9b",
            runtime="lmstudio",
            base_url="http://host.docker.internal:1234/v1",
            gpu_ids=["gpu0"],
            state="stopped",
            reserved_vram_mb=1024,
            updated_at=old.isoformat(),
            last_used_at=old.isoformat(),
        )
    )
    controller = LifecycleController(
        config_path=str(config_path),
        registry=registry,
        gpu_inventory_url="http://gpu-inventory:4200",
        request_timeout_seconds=1,
        dry_run=True,
    )

    result = asyncio.run(controller.cleanup({}))

    assert result["removed_instances"][0]["instance_id"] == "stale"
    assert registry.list() == []
