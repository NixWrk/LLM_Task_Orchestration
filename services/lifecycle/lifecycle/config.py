from __future__ import annotations

from typing import Any

from orchestrator_core.config import load_orchestrator_config

from lifecycle.models import EnvironmentVariable, ModelProfile, VolumeMount


def load_model_profiles(path: str) -> dict[str, ModelProfile]:
    config = load_orchestrator_config(path)
    profiles: dict[str, ModelProfile] = {}

    for model_key, model_config in config.models.items():
        profile = model_profile_from_data(model_key, model_config or {}, config.defaults)
        profiles[profile.public_name] = profile

    return profiles


def load_dynamic_models_config(path: str) -> dict[str, Any]:
    return load_orchestrator_config(path).dynamic_models


def load_dynamic_model_profile(
    path: str,
    model: str,
    overrides: dict[str, Any] | None = None,
) -> ModelProfile | None:
    config = load_orchestrator_config(path)
    dynamic_models = config.dynamic_models
    if not bool(dynamic_models.get("enabled", False)):
        return None

    model_config = {
        **dynamic_models,
        **(overrides or {}),
        "public_name": model,
        "backend_model": model,
    }
    lifecycle_defaults = dynamic_models.get("lifecycle") or {}
    lifecycle_overrides = (overrides or {}).get("lifecycle") or {}
    model_config["lifecycle"] = {
        "runtime": "lmstudio",
        "base_url": "http://host.docker.internal:1234/v1",
        "estimated_vram_gb": 8,
        "safety_margin_gb": 1,
        "min_replicas": 0,
        "max_replicas": 1,
        "idle_ttl_seconds": 900,
        "preferred_gpus": ["auto"],
        "load_strategy": "cli-if-available",
        "lms_binary": "lms",
        **lifecycle_defaults,
        **lifecycle_overrides,
    }
    return model_profile_from_data(model, model_config, config.defaults)


def model_profile_from_data(
    model_key: str,
    model_config: dict[str, Any],
    defaults: dict[str, Any],
) -> ModelProfile:
    model_data = {**defaults, **model_config}
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
    return ModelProfile(
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
        healthcheck_interval_seconds=float(lifecycle_data.get("healthcheck_interval_seconds", 2)),
        warmup_enabled=bool(lifecycle_data.get("warmup_enabled", True)),
        warmup_prompt=str(lifecycle_data.get("warmup_prompt", "Return exactly: ok")),
        warmup_max_tokens=int(lifecycle_data.get("warmup_max_tokens", 8)),
        estimated_vram_mb=estimated_vram_mb,
        safety_margin_mb=safety_margin_mb,
        min_replicas=int(lifecycle_data.get("min_replicas", 0)),
        max_replicas=int(lifecycle_data.get("max_replicas", 1)),
        idle_ttl_seconds=int(lifecycle_data.get("idle_ttl_seconds", 3600)),
        load_strategy=str(lifecycle_data.get("load_strategy", model_data.get("load_strategy", "none"))),
        lms_binary=str(lifecycle_data.get("lms_binary", model_data.get("lms_binary", "lms"))),
        lms_gpu=optional_str(lifecycle_data.get("lms_gpu", model_data.get("lms_gpu"))),
        lms_context_length=optional_int(
            lifecycle_data.get("lms_context_length", model_data.get("lms_context_length"))
        ),
        lms_parallel=optional_int(
            lifecycle_data.get("lms_parallel", model_data.get("lms_parallel"))
        ),
        lms_ttl_seconds=optional_int(
            lifecycle_data.get("lms_ttl_seconds", model_data.get("lms_ttl_seconds"))
        ),
        preferred_gpus=preferred_gpus,
    )


def vram_mb(value: Any) -> int:
    if value is None:
        return 0
    return int(float(value) * 1024)


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


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
