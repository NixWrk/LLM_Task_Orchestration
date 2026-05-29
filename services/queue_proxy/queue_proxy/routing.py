from __future__ import annotations

import logging
from typing import Any

import httpx

from queue_proxy.backend_registry import BackendRegistryClient
from queue_proxy.settings import Settings


class AllocationUnavailable(Exception):
    pass


class BackendResolver:
    def __init__(
        self,
        settings: Settings,
        backend_registry_client: BackendRegistryClient | None,
        logger: logging.Logger,
    ) -> None:
        self.settings = settings
        self.backend_registry_client = backend_registry_client
        self.logger = logger

    async def resolve(
        self,
        model: str,
        orchestration: dict[str, Any] | None = None,
    ) -> tuple[str | None, str | None]:
        if (
            not self.settings.enable_backend_registry_routing
            or self.backend_registry_client is None
        ):
            return self.settings.upstream_base_url, None

        try:
            backend = await self.backend_registry_client.choose_backend(model)
        except httpx.HTTPError as exc:
            self.logger.warning(
                "backend_registry_lookup_failed error_type=%s",
                type(exc).__name__,
            )
            if self.settings.require_backend_registry_backend:
                return None, None
            return self.settings.upstream_base_url, None

        if backend is None:
            try:
                backend = await self.ensure_allocation(model, orchestration)
            except AllocationUnavailable:
                return None, None
            if backend is None or not backend.is_ready:
                if self.settings.require_backend_registry_backend:
                    return None, None
                return self.settings.upstream_base_url, None

        try:
            await self.backend_registry_client.lease_backend(backend.instance_id)
        except httpx.HTTPError as exc:
            self.logger.warning(
                "backend_registry_lease_failed error_type=%s",
                type(exc).__name__,
            )
            if self.settings.require_backend_registry_backend:
                return None, None
            return self.settings.upstream_base_url, None

        return backend.base_url, backend.instance_id

    async def ensure_allocation(
        self,
        model: str,
        orchestration: dict[str, Any] | None,
    ) -> Any:
        if self.backend_registry_client is None:
            return None
        try:
            return await self.backend_registry_client.ensure_allocation(model, orchestration)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {403, 404, 409}:
                raise AllocationUnavailable from exc
            self.logger.warning(
                "backend_allocation_failed status=%s",
                exc.response.status_code,
            )
            if self.settings.require_backend_registry_backend:
                return None
            return None
        except httpx.HTTPError as exc:
            self.logger.warning(
                "backend_allocation_failed error_type=%s",
                type(exc).__name__,
            )
            if self.settings.require_backend_registry_backend:
                return None
            return None

    async def release(self, instance_id: str | None) -> None:
        if instance_id is None or self.backend_registry_client is None:
            return
        try:
            await self.backend_registry_client.release_backend(instance_id)
        except httpx.HTTPError as exc:
            self.logger.warning(
                "backend_registry_release_failed error_type=%s",
                type(exc).__name__,
            )
