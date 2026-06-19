from __future__ import annotations

import subprocess
from typing import Protocol

from lifecycle.lmstudio import parse_estimated_gpu_memory_mb
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


class ExternalOpenAIAdapter:
    def start(
        self,
        profile: ModelProfile,
        gpu_id: str,
        gpu_index: int,
        host_port: int,
        instance_id: str,
        reserved_vram_mb: int,
    ) -> BackendInstance:
        if not profile.base_url:
            raise ValueError(
                f"Model {profile.public_name} uses runtime {profile.runtime} but has no lifecycle.base_url"
            )

        command = ["external-openai", profile.base_url]
        loaded_with_lms = False
        if profile.runtime == "lmstudio":
            command, loaded_with_lms = maybe_load_lmstudio(profile)

        return BackendInstance(
            instance_id=instance_id,
            model=profile.public_name,
            backend_model=profile.backend_model,
            runtime=profile.runtime,
            base_url=profile.base_url,
            gpu_ids=[gpu_id],
            state="starting",
            reserved_vram_mb=reserved_vram_mb,
            host_port=host_port,
            container_name=None,
            runtime_command=command,
            dry_run=False,
            metadata={
                "lmstudio_loaded_with_lms": loaded_with_lms,
                "lms_binary": profile.lms_binary,
            },
        )

    def stop(self, instance: BackendInstance) -> None:
        if instance.runtime != "lmstudio":
            return
        if not instance.metadata.get("lmstudio_loaded_with_lms"):
            return
        lms_binary = str(
            instance.metadata.get("lms_binary")
            or (instance.runtime_command[0] if instance.runtime_command else "lms")
        )
        command = lmstudio_unload_command(lms_binary, instance.backend_model)
        subprocess.run(command, check=True, capture_output=True, text=True)


def adapter_for(profile: ModelProfile, dry_run: bool, docker_binary: str) -> RuntimeAdapter:
    if dry_run:
        return DryRunAdapter()
    if profile.runtime in {"external", "lmstudio", "openai-compatible"}:
        return ExternalOpenAIAdapter()
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
    model = container_model_path(profile)
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
        *docker_volume_args(profile),
        *docker_environment_args(profile),
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


def maybe_load_lmstudio(profile: ModelProfile) -> tuple[list[str], bool]:
    command = lmstudio_load_command(profile)
    strategy = profile.load_strategy.lower()
    if strategy in {"none", "api", "external"}:
        return ["external-openai", profile.base_url or ""], False

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=profile.startup_timeout_seconds,
        )
        return command, True
    except subprocess.CalledProcessError as exc:
        if lmstudio_identifier_already_exists(exc):
            return ["lmstudio-preexisting", *command], False
        raise
    except FileNotFoundError:
        if strategy in {"cli-if-available", "if-available", "auto"}:
            return ["lmstudio-cli-unavailable", *command], False
        raise


def lmstudio_load_command(profile: ModelProfile) -> list[str]:
    command = [
        profile.lms_binary,
        "load",
        profile.backend_model,
        "--identifier",
        profile.backend_model,
        "--yes",
    ]
    if profile.lms_gpu:
        command.extend(["--gpu", profile.lms_gpu])
    if profile.lms_context_length is not None:
        command.extend(["--context-length", str(profile.lms_context_length)])
    if profile.lms_parallel is not None:
        command.extend(["--parallel", str(profile.lms_parallel)])
    if profile.lms_ttl_seconds is not None:
        command.extend(["--ttl", str(profile.lms_ttl_seconds)])
    return command


def lmstudio_estimate_command(profile: ModelProfile) -> list[str]:
    return [*lmstudio_load_command(profile), "--estimate-only"]


def estimate_lmstudio_load_vram_mb(profile: ModelProfile) -> int | None:
    completed = subprocess.run(
        lmstudio_estimate_command(profile),
        check=True,
        capture_output=True,
        text=True,
        timeout=profile.startup_timeout_seconds,
    )
    return parse_estimated_gpu_memory_mb(completed.stdout)


def lmstudio_unload_command(lms_binary: str, identifier: str) -> list[str]:
    return [lms_binary, "unload", identifier]


def lmstudio_identifier_already_exists(exc: subprocess.CalledProcessError) -> bool:
    output = f"{exc.stdout or ''}\n{exc.stderr or ''}".lower()
    return "identifier" in output and "already exists" in output


def container_name(instance_id: str) -> str:
    return f"llm-{instance_id}".replace("_", "-").replace("/", "-").lower()


def docker_volume_args(profile: ModelProfile) -> list[str]:
    args: list[str] = []
    for mount in profile.volume_mounts:
        args.extend(["-v", f"{mount.host_path}:{mount.container_path}:{mount.mode}"])
    return args


def docker_environment_args(profile: ModelProfile) -> list[str]:
    args: list[str] = []
    for item in profile.environment:
        args.extend(["-e", f"{item.name}={item.value}"])
    return args


def container_model_path(profile: ModelProfile) -> str:
    if not profile.artifact:
        return profile.backend_model

    for mount in profile.volume_mounts:
        normalized_host = mount.host_path.rstrip("/\\")
        normalized_artifact = profile.artifact.rstrip("/\\")
        if normalized_artifact == normalized_host:
            return mount.container_path
        if normalized_artifact.startswith(f"{normalized_host}/") or normalized_artifact.startswith(
            f"{normalized_host}\\"
        ):
            suffix = normalized_artifact[len(normalized_host) :].lstrip("/\\")
            return f"{mount.container_path.rstrip('/')}/{suffix.replace(chr(92), '/')}"

    return profile.artifact
