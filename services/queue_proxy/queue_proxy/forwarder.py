from __future__ import annotations

from collections.abc import Awaitable, Callable
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
        try:
            upstream_response = await client.send(upstream_request, stream=True)
        except Exception:
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
