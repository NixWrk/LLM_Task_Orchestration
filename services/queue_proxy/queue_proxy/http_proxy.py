from __future__ import annotations

from collections.abc import Mapping

import httpx
from orchestrator_core.openai import openai_url

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def upstream_url(base_url: str, path: str) -> str:
    return openai_url(base_url, path, ensure_v1=True)


def upstream_headers(headers: Mapping[str, str], upstream_api_key: str = "") -> dict[str, str]:
    forwarded = {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }
    if upstream_api_key:
        forwarded["authorization"] = f"Bearer {upstream_api_key}"
    return forwarded


def response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }
