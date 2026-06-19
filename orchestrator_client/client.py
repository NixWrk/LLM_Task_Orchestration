from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_QUEUE_PROXY_URL = "http://localhost:4100"
DEFAULT_LIFECYCLE_URL = "http://localhost:4300"


class OrchestratorClient:
    def __init__(
        self,
        *,
        queue_url: str = DEFAULT_QUEUE_PROXY_URL,
        lifecycle_url: str = DEFAULT_LIFECYCLE_URL,
        api_key: str | None = None,
        timeout_seconds: float = 240,
    ) -> None:
        self.queue_url = queue_url
        self.lifecycle_url = lifecycle_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def models(self) -> Any:
        return request_json(
            "GET",
            join_url(self.lifecycle_url, "/catalog/models"),
            timeout_seconds=self.timeout_seconds,
        )

    def registry(self) -> Any:
        return request_json(
            "GET",
            join_url(self.lifecycle_url, "/registry"),
            timeout_seconds=self.timeout_seconds,
        )

    def cleanup(self) -> Any:
        return request_json(
            "POST",
            join_url(self.lifecycle_url, "/cleanup"),
            {},
            timeout_seconds=self.timeout_seconds,
        )

    def explain_plan(
        self,
        *,
        queue_lengths: dict[str, int] | None = None,
        context_plans: dict[str, dict[str, Any]] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {"queue_lengths": queue_lengths or {}}
        if context_plans is not None:
            payload["context_plans"] = context_plans
        return request_json(
            "POST",
            join_url(self.lifecycle_url, "/explain-plan"),
            payload,
            timeout_seconds=self.timeout_seconds,
        )

    def metrics(self) -> str:
        return request_text(
            "GET",
            join_url(self.lifecycle_url, "/metrics"),
            timeout_seconds=self.timeout_seconds,
        )

    def allocate(self, model: str, orchestration: dict[str, Any] | None = None) -> Any:
        payload: dict[str, Any] = {"model": model}
        if orchestration is not None:
            payload["orchestration"] = orchestration
        return request_json(
            "POST",
            join_url(self.lifecycle_url, "/allocations"),
            payload,
            timeout_seconds=self.timeout_seconds,
        )

    def submit_task_queue(
        self,
        *,
        model: str,
        orchestration: dict[str, Any],
        tasks: list[dict[str, Any]],
        endpoint: str = "/v1/chat/completions",
        payload_template: dict[str, Any] | None = None,
        template_vars: dict[str, Any] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": model,
            "endpoint": endpoint,
            "orchestration": orchestration,
            "tasks": tasks,
        }
        if payload_template is not None:
            payload["payload_template"] = payload_template
        if template_vars is not None:
            payload["template_vars"] = template_vars
        return request_json(
            "POST",
            join_url(self.queue_url, "/tasks/queue"),
            payload,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )

    def list_tasks(
        self,
        *,
        tenant: str,
        state: str | None = None,
        model: str | None = None,
        limit: int = 100,
    ) -> Any:
        params: dict[str, str] = {"tenant": tenant, "limit": str(limit)}
        if state:
            params["state"] = state
        if model:
            params["model"] = model
        return request_json(
            "GET",
            join_url(self.queue_url, f"/tasks?{urlencode(params)}"),
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )

    def explain_tasks(
        self,
        *,
        tenant: str,
        model: str | None = None,
        limit: int = 500,
    ) -> Any:
        params: dict[str, str] = {"tenant": tenant, "limit": str(limit)}
        if model:
            params["model"] = model
        return request_json(
            "GET",
            join_url(self.queue_url, f"/tasks/explain?{urlencode(params)}"),
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )

    def get_task(self, task_id: str, *, tenant: str) -> Any:
        return request_json(
            "GET",
            join_url(self.queue_url, f"/tasks/{task_id}?{urlencode({'tenant': tenant})}"),
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )

    def cancel_task(self, task_id: str, *, tenant: str) -> Any:
        return request_json(
            "DELETE",
            join_url(self.queue_url, f"/tasks/{task_id}?{urlencode({'tenant': tenant})}"),
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 64,
        stream: bool = False,
        orchestration: dict[str, Any] | None = None,
    ) -> Any:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if orchestration is not None:
            payload["orchestration"] = orchestration
        url = join_url(self.queue_url, "/v1/chat/completions")
        if stream:
            return request_text(
                "POST",
                url,
                payload,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
            )
        return request_json(
            "POST",
            url,
            payload,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )

    def embeddings(
        self,
        model: str,
        text: str,
        orchestration: dict[str, Any] | None = None,
    ) -> Any:
        payload = {"model": model, "input": text}
        if orchestration is not None:
            payload["orchestration"] = orchestration
        return request_json(
            "POST",
            join_url(self.queue_url, "/v1/embeddings"),
            payload,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 240,
) -> Any:
    with open_url(method, url, payload, api_key, timeout_seconds) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def request_text(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 60,
) -> str:
    with open_url(method, url, payload, api_key, timeout_seconds) as response:
        return response.read().decode("utf-8")


def open_url(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    api_key: str | None,
    timeout_seconds: float,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if payload is not None else {}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, method=method, headers=headers)
    return urlopen(request, timeout=timeout_seconds)


def join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
