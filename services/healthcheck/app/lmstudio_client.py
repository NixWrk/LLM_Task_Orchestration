from __future__ import annotations

from typing import Any

import httpx


def join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def auth_headers(api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


class OpenAICompatibleClient:
    def __init__(self, timeout_seconds: float) -> None:
        self._timeout = httpx.Timeout(timeout_seconds)

    async def list_models(self, base_url: str, api_key: str) -> list[str]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                join_url(base_url, "/models"),
                headers=auth_headers(api_key),
            )
            response.raise_for_status()
            payload = response.json()

        data = payload.get("data", [])
        if not isinstance(data, list):
            return []

        model_ids: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id") or item.get("model") or item.get("name")
            if model_id:
                model_ids.append(str(model_id))
        return model_ids

    async def chat_completion(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                join_url(base_url, "/v1/chat/completions"),
                headers={
                    **auth_headers(api_key),
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices", [])
        content = ""
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            message = choice.get("message")
            if isinstance(message, dict):
                content = str(message.get("content") or "")
            else:
                content = str(choice.get("text") or "")

        return {
            "id": data.get("id"),
            "model": data.get("model"),
            "content": content,
        }
