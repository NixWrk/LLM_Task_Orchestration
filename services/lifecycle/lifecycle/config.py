from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lifecycle.models import ModelProfile


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
            docker_extra_args=string_tuple(lifecycle_data.get("docker_extra_args", [])),
            runtime_extra_args=string_tuple(lifecycle_data.get("runtime_extra_args", [])),
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
