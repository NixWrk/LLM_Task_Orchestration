from __future__ import annotations

import hashlib
import json
import math
import subprocess
from contextlib import suppress
from dataclasses import replace
from fnmatch import fnmatchcase
from time import monotonic
from typing import Any

import httpx

from lifecycle.adapters import adapter_for
from lifecycle.config import (
    load_dynamic_model_profile,
    load_dynamic_models_config,
    load_model_profiles,
)
from lifecycle.models import (
    BackendInstance,
    GpuState,
    ModelProfile,
    PlacementDecision,
    now_iso,
    parse_iso,
)
from lifecycle.registry import BackendRegistry
from lifecycle.scheduler import choose_gpu, desired_replicas


class LifecycleController:
    def __init__(
        self,
        config_path: str,
        registry: BackendRegistry,
        gpu_inventory_url: str,
        request_timeout_seconds: float,
        dry_run: bool,
        docker_binary: str = "docker",
    ) -> None:
        self.config_path = config_path
        self.registry = registry
        self.gpu_inventory_url = gpu_inventory_url.rstrip("/")
        self.request_timeout_seconds = request_timeout_seconds
        self.dry_run = dry_run
        self.docker_binary = docker_binary
        self.allocation_results: dict[tuple[str, str], int] = {}

    def profiles(self) -> dict[str, ModelProfile]:
        return load_model_profiles(self.config_path)

    def profile_for_model(
        self,
        model: str,
        overrides: dict[str, Any] | None = None,
    ) -> ModelProfile | None:
        profiles = self.profiles()
        if model in profiles:
            return profiles[model]
        return load_dynamic_model_profile(self.config_path, model, overrides)

    async def catalog(self) -> dict[str, Any]:
        profiles = self.profiles()
        dynamic_config = load_dynamic_models_config(self.config_path)
        dynamic_enabled = bool(dynamic_config.get("enabled", False))
        dynamic_models: list[dict[str, Any]] = []

        if dynamic_enabled:
            probe = load_dynamic_model_profile(self.config_path, "__catalog_probe__")
            if probe and probe.base_url:
                for model_id in await self.list_openai_model_ids(probe.base_url):
                    dynamic_models.append(
                        {
                            "id": model_id,
                            "allowed": dynamic_model_allowed(model_id, dynamic_config),
                            "source": dynamic_config.get("source", "lmstudio"),
                        }
                    )

        return {
            "configured_models": [
                {
                    "id": profile.public_name,
                    "backend_model": profile.backend_model,
                    "runtime": profile.runtime,
                }
                for profile in profiles.values()
            ],
            "dynamic_models_enabled": dynamic_enabled,
            "dynamic_models": dynamic_models,
        }

    async def gpu_states(self) -> list[GpuState]:
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.get(f"{self.gpu_inventory_url}/gpus")
            response.raise_for_status()
            payload = response.json()
        return [
            GpuState(
                id=str(item["id"]),
                index=int(item["index"]),
                name=str(item["name"]),
                memory_total_mb=int(item["memory_total_mb"]),
                memory_used_mb=int(item["memory_used_mb"]),
                memory_free_mb=int(item["memory_free_mb"]),
            )
            for item in payload.get("gpus", [])
        ]

    async def plan(self, queue_lengths: dict[str, int] | None = None) -> dict[str, Any]:
        queue_lengths = queue_lengths or {}
        profiles = self.profiles()
        gpus = await self.gpu_states()
        plans: list[dict[str, Any]] = []

        for profile in profiles.values():
            ready_count = len(self.registry.ready_for_model(profile.public_name))
            active_count = len(self.registry.active_for_model(profile.public_name))
            desired_count = desired_replicas(
                profile,
                ready_count,
                queue_lengths.get(profile.public_name, 0),
            )
            # Scale one replica per reconcile cycle so placement can account for each
            # newly reserved backend before making the next decision.
            missing = 1 if desired_count > active_count else 0

            decisions: list[PlacementDecision] = []
            for _ in range(missing):
                decisions.append(choose_gpu(profile, gpus, self.registry))

            if not decisions and desired_count == active_count:
                decisions.append(
                    PlacementDecision(
                        model=profile.public_name,
                        action="noop",
                        gpu_id=None,
                        reason="desired_replicas_satisfied",
                        required_vram_mb=profile.estimated_vram_mb + profile.safety_margin_mb,
                    )
                )

            plans.append(
                {
                    "model": profile.public_name,
                    "ready_replicas": ready_count,
                    "active_replicas": active_count,
                    "desired_replicas": desired_count,
                    "decisions": [decision.to_dict() for decision in decisions],
                }
            )

        return {
            "dry_run": self.dry_run,
            "gpu_count": len(gpus),
            "models": plans,
        }

    async def reconcile(self, queue_lengths: dict[str, int] | None = None) -> dict[str, Any]:
        queue_lengths = queue_lengths or {}
        profiles = self.profiles()
        stopped = await self.stop_idle_instances(profiles, queue_lengths)
        plan = await self.plan(queue_lengths)
        created: list[dict[str, Any]] = []

        for model_plan in plan["models"]:
            profile = profiles[model_plan["model"]]
            gpus = {gpu.id: gpu for gpu in await self.gpu_states()}
            for decision_payload in model_plan["decisions"]:
                if decision_payload["action"] != "start" or not decision_payload["gpu_id"]:
                    continue
                instance = self.start_instance(profile, decision_payload, gpus)
                self.registry.upsert(instance)
                instance = await self.initialize_instance(profile, instance)
                created.append(instance.to_dict())

        plan["created_instances"] = created
        plan["stopped_instances"] = stopped
        return plan

    async def allocate(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = str(payload.get("model") or "").strip()
        if not model:
            self.record_allocation("unknown", "invalid_request")
            raise ValueError("Allocation requires a non-empty model.")

        overrides = allocation_overrides(payload)
        configured_profiles = self.profiles()
        profile = configured_profiles.get(model)
        dynamic_config: dict[str, Any] = {}
        if profile is None:
            dynamic_config = load_dynamic_models_config(self.config_path)
            if not dynamic_model_allowed(model, dynamic_config):
                self.record_allocation(model, "denied")
                raise PermissionError(f"Model {model} is not allowed by dynamic model policy.")
            profile = load_dynamic_model_profile(self.config_path, model, overrides)
        if profile is None:
            self.record_allocation(model, "not_configured")
            raise LookupError(f"Model {model} is not configured and dynamic models are disabled.")

        ready_instances = self.registry.ready_for_model(profile.public_name)
        if ready_instances:
            instance = min(ready_instances, key=lambda item: item.active_requests)
            self.record_allocation(profile.public_name, "reused")
            return {
                "model": profile.public_name,
                "created": False,
                "instance": instance.to_dict(),
            }

        profile, model_metadata = self.enrich_profile_from_lmstudio_metadata(
            profile,
            dynamic_config,
            payload,
        )
        if should_verify_before_start(profile):
            await self.verify_model_available(profile)
        gpus = await self.gpu_states()
        decision = choose_gpu(profile, gpus, self.registry)
        if decision.action != "start" or not decision.gpu_id:
            self.record_allocation(profile.public_name, "insufficient_vram")
            return {
                "model": profile.public_name,
                "created": False,
                "decision": decision.to_dict(),
                "instance": None,
            }

        gpu_by_id = {gpu.id: gpu for gpu in gpus}
        try:
            instance = self.start_instance(profile, decision.to_dict(), gpu_by_id)
        except Exception:
            self.record_allocation(profile.public_name, "failed")
            raise
        instance.metadata.update(model_metadata)
        self.registry.upsert(instance)
        instance = await self.initialize_instance(profile, instance)
        self.record_allocation(
            profile.public_name,
            "created" if instance.state == "ready" else "failed",
        )
        return {
            "model": profile.public_name,
            "created": instance.state == "ready",
            "decision": decision.to_dict(),
            "instance": instance.to_dict(),
        }

    def record_allocation(self, model: str, result: str) -> None:
        key = (model, result)
        self.allocation_results[key] = self.allocation_results.get(key, 0) + 1

    def start_instance(
        self,
        profile: ModelProfile,
        decision_payload: dict[str, Any],
        gpu_by_id: dict[str, GpuState],
    ) -> BackendInstance:
        gpu_id = str(decision_payload["gpu_id"])
        gpu = gpu_by_id[gpu_id]
        instance_id = instance_id_for(profile.public_name, gpu_id, now_iso())
        host_port = self.registry.next_host_port(profile.public_name, profile.host_port_start)
        reserved_vram_mb = int(decision_payload["required_vram_mb"])
        adapter = adapter_for(profile, dry_run=self.dry_run, docker_binary=self.docker_binary)
        instance = adapter.start(
            profile,
            gpu_id,
            gpu.index,
            host_port,
            instance_id,
            reserved_vram_mb,
        )
        if instance.dry_run:
            instance.base_url = f"dry-run://{profile.public_name}/{instance_id}"
        elif profile.runtime == "vllm":
            instance.base_url = f"http://{profile.public_host}:{host_port}/v1"
        instance.metadata.update(
            {
                "idle_ttl_seconds": profile.idle_ttl_seconds,
                "estimated_vram_mb": profile.estimated_vram_mb,
                "safety_margin_mb": profile.safety_margin_mb,
            }
        )
        return instance

    async def initialize_instance(
        self,
        profile: ModelProfile,
        instance: BackendInstance,
    ) -> BackendInstance:
        if instance.dry_run:
            return self.registry.mark_state(instance.instance_id, "ready")

        try:
            await self.wait_for_health(profile, instance)
            if not should_verify_before_start(profile):
                await self.verify_model_available(profile)
            self.registry.mark_state(instance.instance_id, "warming")
            if profile.warmup_enabled:
                await self.warmup(profile, instance)
            return self.registry.mark_state(instance.instance_id, "ready")
        except Exception as exc:
            with suppress(Exception):
                adapter_for(
                    profile,
                    dry_run=instance.dry_run or self.dry_run,
                    docker_binary=self.docker_binary,
                ).stop(instance)
            return self.registry.mark_failed(instance.instance_id, type(exc).__name__)

    async def wait_for_health(
        self,
        profile: ModelProfile,
        instance: BackendInstance,
    ) -> None:
        deadline = monotonic() + profile.startup_timeout_seconds
        last_error: Exception | None = None
        while monotonic() < deadline:
            try:
                async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                    response = await client.get(openai_url(instance.base_url, profile.healthcheck_path))
                    if 200 <= response.status_code < 300:
                        return
            except httpx.HTTPError as exc:
                last_error = exc
            await async_sleep(profile.healthcheck_interval_seconds)
        if last_error:
            raise last_error
        raise TimeoutError(f"Backend {instance.instance_id} did not become healthy.")

    async def warmup(self, profile: ModelProfile, instance: BackendInstance) -> None:
        payload = {
            "model": profile.backend_model,
            "messages": [{"role": "user", "content": profile.warmup_prompt}],
            "temperature": 0,
            "max_tokens": profile.warmup_max_tokens,
        }
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.post(
                openai_url(instance.base_url, "/chat/completions"),
                json=payload,
            )
            response.raise_for_status()

    async def verify_model_available(self, profile: ModelProfile) -> None:
        if profile.runtime not in {"lmstudio", "openai-compatible", "external"}:
            return
        if not profile.base_url:
            raise ValueError(f"Model {profile.public_name} has no base_url.")

        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.get(openai_url(profile.base_url, "/v1/models"))
            response.raise_for_status()
            payload = response.json()

        model_ids = model_ids_from_openai_payload(payload)
        if profile.backend_model not in model_ids:
            raise LookupError(f"Model {profile.backend_model} is not visible at {profile.base_url}.")

    async def list_openai_model_ids(self, base_url: str) -> list[str]:
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.get(openai_url(base_url, "/v1/models"))
            response.raise_for_status()
            payload = response.json()

        return sorted(model_ids_from_openai_payload(payload))

    def enrich_profile_from_lmstudio_metadata(
        self,
        profile: ModelProfile,
        dynamic_config: dict[str, Any],
        allocation_payload: dict[str, Any],
    ) -> tuple[ModelProfile, dict[str, Any]]:
        if profile.runtime != "lmstudio":
            return profile, {}
        if allocation_has_vram_override(allocation_payload):
            return profile, {}
        if not bool(dynamic_config.get("auto_vram_from_lms", True)):
            return profile, {}

        metadata = lmstudio_metadata_for_model(
            profile.backend_model,
            str(dynamic_config.get("lms_binary") or profile.lms_binary),
        )
        if not metadata:
            return profile, {}

        estimated_vram_mb = estimate_vram_mb_from_lmstudio_metadata(
            metadata,
            fallback_mb=profile.estimated_vram_mb,
        )
        if estimated_vram_mb <= 0:
            return profile, {"lmstudio_metadata": metadata}
        return (
            replace(profile, estimated_vram_mb=estimated_vram_mb),
            {
                "lmstudio_metadata": compact_lmstudio_metadata(metadata),
                "auto_estimated_vram_mb": estimated_vram_mb,
            },
        )

    async def stop_idle_instances(
        self,
        profiles: dict[str, ModelProfile],
        queue_lengths: dict[str, int],
    ) -> list[dict[str, Any]]:
        stopped: list[dict[str, Any]] = []
        for profile in profiles.values():
            if queue_lengths.get(profile.public_name, 0) > 0:
                continue
            ready_instances = self.registry.ready_for_model(profile.public_name)
            candidates = [
                instance
                for instance in ready_instances
                if instance.active_requests == 0
                and idle_seconds(instance) >= idle_ttl_seconds_for(instance, profile)
            ]
            candidates.sort(key=lambda instance: idle_reference(instance))

            removable_count = max(0, len(ready_instances) - profile.min_replicas)
            for instance in candidates[:removable_count]:
                stopped.append(await self.stop_instance(profile, instance))
        return stopped

    async def cleanup(self, queue_lengths: dict[str, int] | None = None) -> dict[str, Any]:
        profiles = self.profiles()
        for instance in self.registry.list():
            if instance.model not in profiles and instance.state == "ready":
                profile = self.profile_for_model(instance.model)
                if profile is not None:
                    profiles[profile.public_name] = profile
        dynamic_config = load_dynamic_models_config(self.config_path)
        stopped = await self.stop_idle_instances(profiles, queue_lengths or {})
        removed = self.purge_stale_instances(registry_cleanup_ttl_seconds(dynamic_config))
        return {
            "stopped_instances": stopped,
            "removed_instances": removed,
            "remaining_instances": [instance.to_dict() for instance in self.registry.list()],
        }

    def purge_stale_instances(self, ttl_seconds: int) -> list[dict[str, Any]]:
        if ttl_seconds < 0:
            return []
        removed: list[dict[str, Any]] = []
        for instance in list(self.registry.list()):
            if instance.runtime != "lmstudio" or instance.state not in {"stopped", "failed"}:
                continue
            if idle_seconds(instance) < ttl_seconds:
                continue
            removed.append(instance.to_dict())
            self.registry.remove(instance.instance_id)
        return removed

    async def stop_instance(
        self,
        profile: ModelProfile,
        instance: BackendInstance,
    ) -> dict[str, Any]:
        self.registry.mark_state(instance.instance_id, "draining")
        try:
            adapter = adapter_for(
                profile,
                dry_run=instance.dry_run or self.dry_run,
                docker_binary=self.docker_binary,
            )
            adapter.stop(instance)
            stopped = self.registry.mark_state(instance.instance_id, "stopped")
            return stopped.to_dict()
        except Exception as exc:
            failed = self.registry.mark_failed(instance.instance_id, type(exc).__name__)
            return failed.to_dict()


def instance_id_for(model: str, gpu_id: str, seed: str) -> str:
    digest = hashlib.sha1(f"{model}:{gpu_id}:{seed}".encode("utf-8")).hexdigest()[:8]
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"{safe_model}-{gpu_id}-{digest}"


def queue_lengths_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("queue_lengths", {})
    if not isinstance(raw, dict):
        return {}
    return {str(model): int(length) for model, length in raw.items()}


def allocation_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    orchestration = payload.get("orchestration")
    if not isinstance(orchestration, dict):
        orchestration = {}

    lifecycle: dict[str, Any] = {}
    for source_key, target_key in (
        ("runtime", "runtime"),
        ("base_url", "base_url"),
        ("estimated_vram_gb", "estimated_vram_gb"),
        ("safety_margin_gb", "safety_margin_gb"),
        ("preferred_gpus", "preferred_gpus"),
        ("idle_ttl_seconds", "idle_ttl_seconds"),
        ("min_replicas", "min_replicas"),
        ("max_replicas", "max_replicas"),
        ("warmup_enabled", "warmup_enabled"),
        ("warmup_prompt", "warmup_prompt"),
        ("warmup_max_tokens", "warmup_max_tokens"),
        ("load_strategy", "load_strategy"),
        ("lms_gpu", "lms_gpu"),
        ("lms_context_length", "lms_context_length"),
        ("lms_parallel", "lms_parallel"),
        ("lms_ttl_seconds", "lms_ttl_seconds"),
    ):
        if source_key in orchestration:
            lifecycle[target_key] = orchestration[source_key]

    if "max_parallel" in orchestration and "lms_parallel" not in lifecycle:
        lifecycle["lms_parallel"] = orchestration["max_parallel"]

    if "gpu" in orchestration and "preferred_gpus" not in lifecycle:
        gpu = orchestration["gpu"]
        lifecycle["preferred_gpus"] = gpu if isinstance(gpu, list) else [gpu]

    return {"lifecycle": lifecycle} if lifecycle else {}


def model_ids_from_openai_payload(payload: dict[str, Any]) -> set[str]:
    raw_models = payload.get("data", [])
    if not isinstance(raw_models, list):
        return set()
    return {
        str(item.get("id"))
        for item in raw_models
        if isinstance(item, dict) and item.get("id") is not None
    }


def dynamic_model_allowed(model: str, config: dict[str, Any]) -> bool:
    if not bool(config.get("enabled", False)):
        return False

    denied = pattern_list(config, "denied_models", "deny_models", "denied_model_patterns")
    if any(pattern_matches(model, pattern) for pattern in denied):
        return False

    allowed = pattern_list(config, "allowed_models", "allow_models", "allowed_model_patterns")
    if not allowed:
        return True
    return any(pattern_matches(model, pattern) for pattern in allowed)


def pattern_list(config: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = config.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
        elif raw not in (None, ""):
            values.append(str(raw))
    return values


def pattern_matches(value: str, pattern: str) -> bool:
    return value == pattern or fnmatchcase(value, pattern)


def allocation_has_vram_override(payload: dict[str, Any]) -> bool:
    orchestration = payload.get("orchestration")
    return isinstance(orchestration, dict) and "estimated_vram_gb" in orchestration


def should_verify_before_start(profile: ModelProfile) -> bool:
    if profile.runtime != "lmstudio":
        return True
    return profile.load_strategy.lower() in {"none", "api", "external"}


def lmstudio_metadata_for_model(model: str, lms_binary: str = "lms") -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [lms_binary, "ls", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}

    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, list):
        return {}

    for item in payload:
        if not isinstance(item, dict):
            continue
        candidates = {
            str(item.get("modelKey") or ""),
            str(item.get("selectedVariant") or ""),
            str(item.get("indexedModelIdentifier") or ""),
        }
        variants = item.get("variants")
        if isinstance(variants, list):
            candidates.update(str(variant) for variant in variants)
        if model in candidates:
            return item
    return {}


def estimate_vram_mb_from_lmstudio_metadata(
    metadata: dict[str, Any],
    fallback_mb: int,
) -> int:
    raw_size = metadata.get("sizeBytes")
    if raw_size in (None, ""):
        return fallback_mb

    size_mb = int(math.ceil(float(raw_size) / (1024 * 1024)))
    context_length = int(metadata.get("maxContextLength") or 0)
    context_overhead_mb = min(4096, max(512, math.ceil(context_length / 4096) * 64))
    return int(math.ceil(size_mb * 1.15)) + context_overhead_mb


def compact_lmstudio_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "type",
        "modelKey",
        "displayName",
        "path",
        "sizeBytes",
        "paramsString",
        "architecture",
        "quantization",
        "maxContextLength",
        "vision",
        "trainedForToolUse",
        "selectedVariant",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def idle_ttl_seconds_for(instance: BackendInstance, profile: ModelProfile) -> int:
    raw_value = instance.metadata.get("idle_ttl_seconds")
    if raw_value in (None, ""):
        return profile.idle_ttl_seconds
    return int(raw_value)


def registry_cleanup_ttl_seconds(config: dict[str, Any]) -> int:
    raw_value = config.get("registry_cleanup_ttl_seconds", 3600)
    return int(raw_value)


async def async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def openai_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    normalized_path = path.lstrip("/")
    if base.endswith("/v1") and normalized_path.startswith("v1/"):
        normalized_path = normalized_path[3:]
    return f"{base}/{normalized_path}"


def idle_reference(instance: BackendInstance) -> str:
    return instance.last_used_at or instance.updated_at or instance.created_at


def idle_seconds(instance: BackendInstance) -> float:
    reference = parse_iso(idle_reference(instance))
    elapsed = parse_iso(now_iso()) - reference
    return max(0.0, elapsed.total_seconds())
