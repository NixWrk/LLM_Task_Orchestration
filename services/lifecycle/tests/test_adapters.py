import subprocess

from lifecycle.adapters import (
    DockerVllmAdapter,
    DryRunAdapter,
    ExternalOpenAIAdapter,
    docker_vllm_command,
    lmstudio_identifier_already_exists,
    lmstudio_load_command,
    lmstudio_unload_command,
)
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
        base_url=None,
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


def test_external_openai_adapter_uses_configured_base_url() -> None:
    external = ModelProfile(
        **{
            **profile().__dict__,
            "runtime": "lmstudio",
            "runtime_image": None,
            "artifact": None,
            "base_url": "http://host.docker.internal:1234/v1",
        }
    )

    instance = ExternalOpenAIAdapter().start(
        external,
        gpu_id="gpu0",
        gpu_index=0,
        host_port=8100,
        instance_id="lmstudio-1",
        reserved_vram_mb=16 * 1024,
    )

    assert instance.base_url == "http://host.docker.internal:1234/v1"
    assert instance.state == "starting"
    assert instance.dry_run is False
    assert instance.runtime_command == ["external-openai", "http://host.docker.internal:1234/v1"]


def test_lmstudio_load_command_contains_identifier_and_runtime_options() -> None:
    external = ModelProfile(
        **{
            **profile().__dict__,
            "runtime": "lmstudio",
            "base_url": "http://host.docker.internal:1234/v1",
            "load_strategy": "cli",
            "lms_binary": "lms",
            "lms_gpu": "max",
            "lms_context_length": 8192,
            "lms_parallel": 2,
            "lms_ttl_seconds": 900,
        }
    )

    command = lmstudio_load_command(external)

    assert command[:3] == ["lms", "load", "qwen"]
    assert ["--identifier", "qwen"] == command[3:5]
    assert "--gpu" in command
    assert "max" in command
    assert "--context-length" in command
    assert "--parallel" in command
    assert "--ttl" in command


def test_lmstudio_unload_command_uses_loaded_identifier() -> None:
    assert lmstudio_unload_command("lms", "qwen") == ["lms", "unload", "qwen"]


def test_external_openai_adapter_loads_and_unloads_lmstudio_with_cli(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)

    monkeypatch.setattr("subprocess.run", fake_run)
    external = ModelProfile(
        **{
            **profile().__dict__,
            "runtime": "lmstudio",
            "runtime_image": None,
            "artifact": None,
            "base_url": "http://host.docker.internal:1234/v1",
            "load_strategy": "cli",
            "lms_binary": "lms",
            "lms_ttl_seconds": 30,
        }
    )

    adapter = ExternalOpenAIAdapter()
    instance = adapter.start(
        external,
        gpu_id="gpu0",
        gpu_index=0,
        host_port=8100,
        instance_id="lmstudio-1",
        reserved_vram_mb=16 * 1024,
    )
    adapter.stop(instance)

    assert calls[0][:3] == ["lms", "load", "qwen"]
    assert calls[-1] == ["lms", "unload", "qwen"]
    assert instance.metadata["lmstudio_loaded_with_lms"] is True


def test_external_openai_adapter_skips_lmstudio_load_when_cli_is_unavailable(
    monkeypatch,
) -> None:
    def fake_run(_command, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("subprocess.run", fake_run)
    external = ModelProfile(
        **{
            **profile().__dict__,
            "runtime": "lmstudio",
            "runtime_image": None,
            "artifact": None,
            "base_url": "http://host.docker.internal:1234/v1",
            "load_strategy": "cli-if-available",
            "lms_binary": "missing-lms",
        }
    )

    instance = ExternalOpenAIAdapter().start(
        external,
        gpu_id="gpu0",
        gpu_index=0,
        host_port=8100,
        instance_id="lmstudio-1",
        reserved_vram_mb=16 * 1024,
    )

    assert instance.runtime_command[0] == "lmstudio-cli-unavailable"
    assert instance.metadata["lmstudio_loaded_with_lms"] is False


def test_lmstudio_identifier_already_exists_detection() -> None:
    exc = subprocess.CalledProcessError(
        1,
        ["lms", "load", "qwen"],
        stderr="Error: A model with identifier qwen already exists.",
    )

    assert lmstudio_identifier_already_exists(exc) is True
