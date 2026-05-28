from __future__ import annotations

import subprocess
from typing import Protocol

from lifecycle.models import BackendInstance, ModelProfile, now_iso


class RuntimeAdapter(Protocol):
    def start(
        self,
        profile: ModelProfile,
        gpu_id: str,
        gpu_index: int,
        host_port: int,
        instance_id: str,
        reserved_vram_mb: int,
    ) -> BackendInstance:
        ...

    def stop(self, instance: BackendInstance) -> None:
        ...


class DryRunAdapter:
    def start(
        self,
        profile: ModelProfile,
        gpu_id: str,
        gpu_index: int,
        host_port: int,
        instance_id: str,
        reserved_vram_mb: int,
    ) -> BackendInstance:
        command = (
            docker_vllm_command(profile, gpu_index, host_port, instance_id)
            if profile.runtime == "vllm"
            else ["dry-run", profile.runtime, profile.backend_model]
        )
        return BackendInstance(
            instance_id=instance_id,
            model=profile.public_name,
            backend_model=profile.backend_model,
            runtime=profile.runtime,
            base_url=f"dry-run://{profile.public_name}/{instance_id}",
            gpu_ids=[gpu_id],
            state="ready",
            reserved_vram_mb=reserved_vram_mb,
            host_port=host_port,
            container_name=container_name(instance_id),
            runtime_command=command,
            dry_run=True,
        )

    def stop(self, _instance: BackendInstance) -> None:
        return None


class DockerVllmAdapter:
    def __init__(self, docker_binary: str = "docker") -> None:
        self.docker_binary = docker_binary

    def start(
        self,
        profile: ModelProfile,
        gpu_id: str,
        gpu_index: int,
        host_port: int,
        instance_id: str,
        reserved_vram_mb: int,
    ) -> BackendInstance:
        command = docker_vllm_command(
            profile,
            gpu_index,
            host_port,
            instance_id,
            docker_binary=self.docker_binary,
        )
        subprocess.run(command, check=True, capture_output=True, text=True)
        return BackendInstance(
            instance_id=instance_id,
            model=profile.public_name,
            backend_model=profile.backend_model,
            runtime=profile.runtime,
            base_url=f"http://{profile.public_host}:{host_port}/v1",
            gpu_ids=[gpu_id],
            state="starting",
            reserved_vram_mb=reserved_vram_mb,
            host_port=host_port,
            container_name=container_name(instance_id),
            runtime_command=command,
            dry_run=False,
        )

    def stop(self, instance: BackendInstance) -> None:
        if not instance.container_name:
            return
        subprocess.run(
            [self.docker_binary, "stop", instance.container_name],
            check=True,
            capture_output=True,
            text=True,
        )


def adapter_for(profile: ModelProfile, dry_run: bool, docker_binary: str) -> RuntimeAdapter:
    if dry_run:
        return DryRunAdapter()
    if profile.runtime == "vllm":
        return DockerVllmAdapter(docker_binary)
    return DryRunAdapter()


def docker_vllm_command(
    profile: ModelProfile,
    gpu_index: int,
    host_port: int,
    instance_id: str,
    docker_binary: str = "docker",
) -> list[str]:
    image = profile.runtime_image or "vllm/vllm-openai:latest"
    model = profile.artifact or profile.backend_model
    command = [
        docker_binary,
        "run",
        "-d",
        "--name",
        container_name(instance_id),
        "--gpus",
        f"device={gpu_index}",
        "-p",
        f"{host_port}:{profile.container_port}",
        *profile.docker_extra_args,
        image,
        "--model",
        model,
        "--served-model-name",
        profile.backend_model,
        "--host",
        "0.0.0.0",
        "--port",
        str(profile.container_port),
        *profile.runtime_extra_args,
    ]
    return command


def container_name(instance_id: str) -> str:
    return f"llm-{instance_id}".replace("_", "-").replace("/", "-").lower()
