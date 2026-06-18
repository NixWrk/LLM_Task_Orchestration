from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import httpx
from fastapi import Request, Response, status
from fastapi.responses import StreamingResponse

from queue_proxy.http_proxy import (
    response_headers,
    upstream_headers as build_upstream_headers,
    upstream_url,
)
from queue_proxy.responses import error_response

StreamFinishedCallback = Callable[[int], Awaitable[None]]


class ClientDisconnectedError(Exception):
    pass


class UpstreamForwarder:
    def __init__(self, request_timeout_seconds: float, upstream_api_key: str) -> None:
        self.request_timeout_seconds = request_timeout_seconds
        self.upstream_api_key = upstream_api_key

    async def forward_without_limiter(
        self,
        path: str,
        request: Request,
        body: bytes,
        upstream_base_url: str,
    ) -> Response:
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
                upstream_response = await client.request(
                    request.method,
                    upstream_url(upstream_base_url, path),
                    headers=self.upstream_headers(request),
                    content=body,
                    params=request.query_params,
                )
        except httpx.HTTPError:
            return error_response(
                status.HTTP_502_BAD_GATEWAY,
                "upstream_request_failed",
                "Upstream LLM gateway request failed.",
            )

        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers(upstream_response.headers),
            media_type=upstream_response.headers.get("content-type"),
        )

    async def stream_response(
        self,
        *,
        path: str,
        request: Request,
        body: bytes,
        upstream_base_url: str,
        on_finished: StreamFinishedCallback,
    ) -> StreamingResponse:
        client = httpx.AsyncClient(timeout=self.request_timeout_seconds)
        upstream_request = client.build_request(
            request.method,
            upstream_url(upstream_base_url, path),
            headers=self.upstream_headers(request),
            content=body,
            params=request.query_params,
        )
        send_task = asyncio.create_task(client.send(upstream_request, stream=True))
        disconnect_task = asyncio.create_task(wait_for_client_disconnect(request))
        try:
            done, _pending = await asyncio.wait(
                {send_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                send_task.cancel()
                with suppress(BaseException):
                    await send_task
                await client.aclose()
                raise ClientDisconnectedError

            disconnect_task.cancel()
            with suppress(BaseException):
                await disconnect_task
            upstream_response = send_task.result()
        except Exception:
            disconnect_task.cancel()
            send_task.cancel()
            with suppress(BaseException):
                await disconnect_task
            with suppress(BaseException):
                await send_task
            await client.aclose()
            raise

        async def response_body() -> Any:
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                status_code = upstream_response.status_code
                await upstream_response.aclose()
                await client.aclose()
                await on_finished(status_code)

        return StreamingResponse(
            response_body(),
            status_code=upstream_response.status_code,
            headers=response_headers(upstream_response.headers),
            media_type=upstream_response.headers.get("content-type"),
        )

    def upstream_headers(self, request: Request) -> dict[str, str]:
        return build_upstream_headers(request.headers, self.upstream_api_key)


async def wait_for_client_disconnect(request: Request) -> None:
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(0.05)
