from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lifecycle.models import EnvironmentVariable, ModelProfile, VolumeMount


def load_model_profiles(path: str) -> dict[str, ModelProfile]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    defaults = raw.get("defaults") or {}
    models = raw.get("models") or {}
    profiles: dict[str, ModelProfile] = {}

    for model_key, model_config in models.items():
        model_data = {**defaults, **(model_config or {})}
        lifecycle_data = model_data.get("lifecycle") or {}
        public_name = str(model_data.get("public_name") or model_key)
        estimated_vram_mb = vram_mb(
            lifecycle_data.get("estimated_vram_gb", model_data.get("estimated_vram_gb", 0))
        )
        safety_margin_mb = vram_mb(
            lifecycle_data.get("safety_margin_gb", model_data.get("safety_margin_gb", 1))
        )
        preferred_gpus = tuple(
            str(gpu_id) for gpu_id in lifecycle_data.get("preferred_gpus", ["auto"])
        )
        profiles[public_name] = ModelProfile(
            public_name=public_name,
            backend_model=str(model_data.get("backend_model") or public_name),
            runtime=str(lifecycle_data.get("runtime", model_data.get("runtime", "external"))),
            artifact=optional_str(lifecycle_data.get("artifact", model_data.get("artifact"))),
            runtime_image=optional_str(
                lifecycle_data.get("runtime_image", model_data.get("runtime_image"))
            ),
            host_port_start=int(lifecycle_data.get("host_port_start", 8100)),
            container_port=int(lifecycle_data.get("container_port", 8000)),
            public_host=str(lifecycle_data.get("public_host", "host.docker.internal")),
            base_url=optional_str(lifecycle_data.get("base_url", model_data.get("base_url"))),
            docker_extra_args=string_tuple(lifecycle_data.get("docker_extra_args", [])),
            runtime_extra_args=string_tuple(lifecycle_data.get("runtime_extra_args", [])),
            volume_mounts=volume_mounts(lifecycle_data.get("volumes", [])),
            environment=environment_variables(lifecycle_data.get("environment", {})),
            healthcheck_path=str(lifecycle_data.get("healthcheck_path", "/v1/models")),
            startup_timeout_seconds=float(lifecycle_data.get("startup_timeout_seconds", 120)),
            healthcheck_interval_seconds=float(
                lifecycle_data.get("healthcheck_interval_seconds", 2)
            ),
            warmup_enabled=bool(lifecycle_data.get("warmup_enabled", True)),
            warmup_prompt=str(lifecycle_data.get("warmup_prompt", "Return exactly: ok")),
            warmup_max_tokens=int(lifecycle_data.get("warmup_max_tokens", 8)),
            estimated_vram_mb=estimated_vram_mb,
            safety_margin_mb=safety_margin_mb,
            min_replicas=int(lifecycle_data.get("min_replicas", 0)),
            max_replicas=int(lifecycle_data.get("max_replicas", 1)),
            idle_ttl_seconds=int(lifecycle_data.get("idle_ttl_seconds", 3600)),
            preferred_gpus=preferred_gpus,
        )

    return profiles


def vram_mb(value: Any) -> int:
    if value is None:
        return 0
    return int(float(value) * 1024)


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def volume_mounts(value: Any) -> tuple[VolumeMount, ...]:
    if not isinstance(value, list):
        return ()
    mounts: list[VolumeMount] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        host_path = optional_str(item.get("host_path") or item.get("host"))
        container_path = optional_str(item.get("container_path") or item.get("container"))
        if not host_path or not container_path:
            continue
        mounts.append(
            VolumeMount(
                host_path=host_path,
                container_path=container_path,
                mode=str(item.get("mode") or "ro"),
            )
        )
    return tuple(mounts)


def environment_variables(value: Any) -> tuple[EnvironmentVariable, ...]:
    if not isinstance(value, dict):
        return ()
    return tuple(
        EnvironmentVariable(name=str(name), value=str(raw_value))
        for name, raw_value in value.items()
    )
