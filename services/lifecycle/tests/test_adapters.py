from lifecycle.adapters import DockerVllmAdapter, DryRunAdapter, docker_vllm_command
from lifecycle.models import ModelProfile


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
