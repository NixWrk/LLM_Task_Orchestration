from pathlib import Path

from lifecycle.config import load_dynamic_model_profile, load_model_profiles
from lifecycle.models import BackendInstance, GpuState, ModelProfile
from lifecycle.registry import BackendRegistry
from lifecycle.scheduler import choose_gpu, desired_replicas


def profile() -> ModelProfile:
    return ModelProfile(
        public_name="local-main",
        backend_model="local-main",
        runtime="vllm",
        artifact="/models/local-main",
        runtime_image="vllm/vllm-openai:latest",
        host_port_start=8100,
        container_port=8000,
        public_host="host.docker.internal",
        base_url=None,
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
        min_replicas=1,
        max_replicas=2,
        idle_ttl_seconds=3600,
        preferred_gpus=("auto",),
    )


def test_choose_gpu_selects_highest_available_vram(tmp_path: Path) -> None:
    registry = BackendRegistry(str(tmp_path / "registry.json"))
    decision = choose_gpu(
        profile(),
        [
            GpuState("gpu0", 0, "small", 12000, 6000, 6000),
            GpuState("gpu1", 1, "big", 24000, 4000, 20000),
        ],
        registry,
    )

    assert decision.action == "start"
    assert decision.gpu_id == "gpu1"


def test_choose_gpu_accounts_for_reserved_vram(tmp_path: Path) -> None:
    registry = BackendRegistry(str(tmp_path / "registry.json"))
    registry.upsert(
        BackendInstance(
            instance_id="existing",
            model="other",
            backend_model="other",
            runtime="vllm",
            base_url="http://backend",
            gpu_ids=["gpu1"],
            state="ready",
            reserved_vram_mb=15000,
        )
    )

    decision = choose_gpu(
        profile(),
        [
            GpuState("gpu0", 0, "small", 16000, 2000, 14000),
            GpuState("gpu1", 1, "big", 24000, 4000, 20000),
        ],
        registry,
    )

    assert decision.action == "start"
    assert decision.gpu_id == "gpu0"


def test_choose_gpu_returns_noop_when_vram_is_insufficient(tmp_path: Path) -> None:
    registry = BackendRegistry(str(tmp_path / "registry.json"))
    decision = choose_gpu(
        profile(),
        [GpuState("gpu0", 0, "small", 12000, 8000, 4000)],
        registry,
    )

    assert decision.action == "noop"
    assert decision.reason == "no_gpu_with_enough_vram"


def test_desired_replicas_scales_up_with_queue_pressure() -> None:
    assert desired_replicas(profile(), ready_replicas=1, queue_length=3) == 2


def test_load_model_profiles_reads_lifecycle_config(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  qwen:",
                "    public_name: qwen",
                "    backend_model: qwen-backend",
                "    lifecycle:",
                "      runtime: vllm",
                "      artifact: /models/qwen",
                "      runtime_image: vllm/vllm-openai:latest",
                "      host_port_start: 8100",
                "      container_port: 8000",
                "      public_host: host.docker.internal",
                "      base_url: http://host.docker.internal:1234/v1",
                "      volumes:",
                "        - host_path: D:/models/qwen",
                "          container_path: /models/qwen",
                "          mode: ro",
                "      environment:",
                "        HF_HOME: /cache/hf",
                "      healthcheck_path: /v1/models",
                "      startup_timeout_seconds: 30",
                "      healthcheck_interval_seconds: 1",
                "      warmup_enabled: true",
                "      warmup_prompt: 'Return exactly: ok'",
                "      warmup_max_tokens: 8",
                "      estimated_vram_gb: 14",
                "      safety_margin_gb: 2",
                "      min_replicas: 1",
                "      max_replicas: 3",
                "      idle_ttl_seconds: 120",
                "      preferred_gpus: [gpu0, gpu1]",
            ]
        ),
        encoding="utf-8",
    )

    profiles = load_model_profiles(str(config_path))

    assert profiles["qwen"].estimated_vram_mb == 14 * 1024
    assert profiles["qwen"].safety_margin_mb == 2 * 1024
    assert profiles["qwen"].preferred_gpus == ("gpu0", "gpu1")
    assert profiles["qwen"].base_url == "http://host.docker.internal:1234/v1"
    assert profiles["qwen"].volume_mounts[0].host_path == "D:/models/qwen"
    assert profiles["qwen"].environment[0].name == "HF_HOME"


def test_load_dynamic_model_profile_uses_lmstudio_defaults(tmp_path: Path) -> None:
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
                "    estimated_vram_gb: 9",
                "    load_strategy: cli-if-available",
                "    lms_ttl_seconds: 900",
                "    preferred_gpus: [gpu1]",
            ]
        ),
        encoding="utf-8",
    )

    profile = load_dynamic_model_profile(str(config_path), "qwen/qwen3.5-9b")

    assert profile is not None
    assert profile.public_name == "qwen/qwen3.5-9b"
    assert profile.backend_model == "qwen/qwen3.5-9b"
    assert profile.runtime == "lmstudio"
    assert profile.base_url == "http://host.docker.internal:1234/v1"
    assert profile.estimated_vram_mb == 9 * 1024
    assert profile.load_strategy == "cli-if-available"
    assert profile.lms_ttl_seconds == 900
    assert profile.preferred_gpus == ("gpu1",)


def test_repository_zotero_html_translate_profile_preserves_baseline() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "orchestrator.yaml"

    profile = load_model_profiles(str(config_path))["zotero-html-translate"]

    assert profile.backend_model == "p6_google_gemma-4-26b-a4b@q6_k"
    assert profile.load_strategy == "cli-if-available"
    assert profile.lms_binary == "lms"
    assert profile.startup_timeout_seconds == 1800
    assert profile.estimated_vram_mb == 26 * 1024
    assert profile.safety_margin_mb == 2 * 1024
    assert profile.lms_gpu == "max"
    assert profile.lms_context_length == 32768
    assert profile.lms_parallel == 2
    assert profile.lms_ttl_seconds == 3600
