from lifecycle.adapters import DockerVllmAdapter, DryRunAdapter, docker_vllm_command
from lifecycle.models import EnvironmentVariable, ModelProfile, VolumeMount


def profile() -> ModelProfile:
    return ModelProfile(
        public_name="qwen",
        backend_model="qwen",
        runtime="vllm",
        artifact="/models/qwen",
        runtime_image="vllm/vllm-openai:latest",
        host_port_start=8100,
        container_port=8000,
        public_host="host.docker.internal",
        docker_extra_args=("--ipc=host",),
        runtime_extra_args=("--max-model-len", "8192"),
        volume_mounts=(),
        environment=(),
        healthcheck_path="/v1/models",
        startup_timeout_seconds=120,
        healthcheck_interval_seconds=2,
        warmup_enabled=True,
        warmup_prompt="Return exactly: ok",
        warmup_max_tokens=8,
        estimated_vram_mb=14 * 1024,
        safety_margin_mb=2 * 1024,
        min_replicas=1,
        max_replicas=2,
        idle_ttl_seconds=3600,
        preferred_gpus=("gpu0",),
    )


def test_docker_vllm_command_contains_gpu_port_and_model() -> None:
    command = docker_vllm_command(profile(), gpu_index=0, host_port=8100, instance_id="qwen-1")

    assert command[:4] == ["docker", "run", "-d", "--name"]
    assert "--gpus" in command
    assert "device=0" in command
    assert "8100:8000" in command
    assert "--model" in command
    assert "/models/qwen" in command
    assert "--max-model-len" in command


def test_docker_vllm_command_maps_host_model_path_to_container_path() -> None:
    mounted = ModelProfile(
        **{
            **profile().__dict__,
            "artifact": "D:/models/qwen",
            "volume_mounts": (VolumeMount("D:/models/qwen", "/models/qwen", "ro"),),
            "environment": (EnvironmentVariable("HF_HOME", "/cache/hf"),),
        }
    )

    command = docker_vllm_command(mounted, gpu_index=1, host_port=8101, instance_id="qwen-2")

    assert "-v" in command
    assert "D:/models/qwen:/models/qwen:ro" in command
    assert "-e" in command
    assert "HF_HOME=/cache/hf" in command
    assert "/models/qwen" in command
    assert "device=1" in command


def test_docker_vllm_adapter_dry_instance_base_url_shape(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)

    monkeypatch.setattr("subprocess.run", fake_run)
    adapter = DockerVllmAdapter()
    instance = adapter.start(
        profile(),
        gpu_id="gpu0",
        gpu_index=0,
        host_port=8100,
        instance_id="qwen-1",
        reserved_vram_mb=16 * 1024,
    )

    assert calls
    assert instance.base_url == "http://host.docker.internal:8100/v1"
    assert instance.state == "starting"


def test_dry_run_adapter_does_not_generate_vllm_command_for_external_runtime() -> None:
    external = profile()
    external = ModelProfile(
        **{
            **external.__dict__,
            "runtime": "external",
            "runtime_image": None,
            "artifact": None,
        }
    )

    instance = DryRunAdapter().start(
        external,
        gpu_id="gpu0",
        gpu_index=0,
        host_port=8100,
        instance_id="external-1",
        reserved_vram_mb=16 * 1024,
    )

    assert instance.runtime_command == ["dry-run", "external", "qwen"]
