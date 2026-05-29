from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from lifecycle.models import BackendInstance, ModelProfile, now_iso, parse_iso
from lifecycle.registry import BackendRegistry

StopInstance = Callable[[ModelProfile, BackendInstance], Awaitable[dict[str, Any]]]
ProfileResolver = Callable[[str], ModelProfile | None]


class CleanupService:
    def __init__(
        self,
        registry: BackendRegistry,
        stop_instance: StopInstance,
    ) -> None:
        self.registry = registry
        self.stop_instance = stop_instance

    async def cleanup(
        self,
        profiles: dict[str, ModelProfile],
        queue_lengths: dict[str, int],
        dynamic_config: dict[str, Any],
        profile_for_model: ProfileResolver,
    ) -> dict[str, Any]:
        profiles = dict(profiles)
        for instance in self.registry.list():
            if instance.model not in profiles and instance.state == "ready":
                profile = profile_for_model(instance.model)
                if profile is not None:
                    profiles[profile.public_name] = profile

        stopped = await self.stop_idle_instances(profiles, queue_lengths)
        removed = self.purge_stale_instances(registry_cleanup_ttl_seconds(dynamic_config))
        return {
            "stopped_instances": stopped,
            "removed_instances": removed,
            "remaining_instances": [instance.to_dict() for instance in self.registry.list()],
        }

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


def idle_ttl_seconds_for(instance: BackendInstance, profile: ModelProfile) -> int:
    raw_value = instance.metadata.get("idle_ttl_seconds")
    if raw_value in (None, ""):
        return profile.idle_ttl_seconds
    return int(raw_value)


def registry_cleanup_ttl_seconds(config: dict[str, Any]) -> int:
    raw_value = config.get("registry_cleanup_ttl_seconds", 3600)
    return int(raw_value)


def idle_reference(instance: BackendInstance) -> str:
    return instance.last_used_at or instance.updated_at or instance.created_at


def idle_seconds(instance: BackendInstance) -> float:
    reference = parse_iso(idle_reference(instance))
    elapsed = parse_iso(now_iso()) - reference
    return max(0.0, elapsed.total_seconds())
