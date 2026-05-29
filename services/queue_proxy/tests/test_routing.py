import asyncio
import logging

import httpx

from queue_proxy.routing import BackendResolver
from queue_proxy.settings import Settings


def test_backend_resolver_denied_allocation_does_not_fallback() -> None:
    resolver = BackendResolver(
        Settings(
            UPSTREAM_LITELLM_BASE_URL="http://static:4000",
            enable_backend_registry_routing=True,
            require_backend_registry_backend=False,
        ),
        DeniedAllocationClient(),
        logging.getLogger("test"),
    )

    upstream_url, instance_id = asyncio.run(resolver.resolve("denied-model"))

    assert upstream_url is None
    assert instance_id is None


def test_backend_resolver_lookup_failure_can_fallback_to_static_upstream() -> None:
    resolver = BackendResolver(
        Settings(
            UPSTREAM_LITELLM_BASE_URL="http://static:4000",
            enable_backend_registry_routing=True,
            require_backend_registry_backend=False,
        ),
        LookupFailureClient(),
        logging.getLogger("test"),
    )

    upstream_url, instance_id = asyncio.run(resolver.resolve("local-main"))

    assert upstream_url == "http://static:4000"
    assert instance_id is None


class DeniedAllocationClient:
    async def choose_backend(self, _model):
        return None

    async def ensure_allocation(self, _model, _orchestration):
        request = httpx.Request("POST", "http://registry/allocations")
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError("denied", request=request, response=response)


class LookupFailureClient:
    async def choose_backend(self, _model):
        raise httpx.ConnectError("down")
