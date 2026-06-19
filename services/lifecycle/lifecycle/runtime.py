from __future__ import annotations

import hashlib
from contextlib import suppress
from time import monotonic
from typing import Any

import httpx
from orchestrator_core.openai import openai_url

from lifecycle.adapters import adapter_for
from lifecycle.models import BackendInstance, GpuState, ModelProfile, now_iso
from lifecycle.registry import BackendRegistry


class RuntimeLifecycleService:
    def __init__(
        self,
        registry: BackendRegistry,
        request_timeout_seconds: float,
        dry_run: bool,
        docker_binary: str = "docker",
    ) -> None:
        self.registry = registry
        self.request_timeout_seconds = request_timeout_seconds
        self.dry_run = dry_run
        self.docker_binary = docker_binary

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
                "lms_context_length": profile.lms_context_length,
                "lms_parallel": profile.lms_parallel,
                "lms_gpu": profile.lms_gpu,
                "lms_ttl_seconds": profile.lms_ttl_seconds,
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
                    response = await client.get(
                        openai_url(instance.base_url, profile.healthcheck_path)
                    )
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


def model_ids_from_openai_payload(payload: dict[str, Any]) -> set[str]:
    raw_models = payload.get("data", [])
    if not isinstance(raw_models, list):
        return set()
    return {
        str(item.get("id"))
        for item in raw_models
        if isinstance(item, dict) and item.get("id") is not None
    }


def should_verify_before_start(profile: ModelProfile) -> bool:
    if profile.runtime != "lmstudio":
        return True
    return profile.load_strategy.lower() in {"none", "api", "external"}


async def async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
