from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from tests.integration.support import (
    running_fake_backend,
    running_fake_registry,
    running_lifecycle,
    running_queue_proxy,
    wait_for_active_request,
    write_policy_config,
)


def test_chat_completion_through_proxy_clamps_output_tokens(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    config_path = write_policy_config(
        tmp_path,
        max_active_requests=1,
        max_queued_requests=1,
        max_output_tokens=8,
        max_total_tokens=200,
    )

    with running_fake_backend(fake_port), running_queue_proxy(
        proxy_port,
        fake_port,
        config_path,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json={
                "model": "local-main",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 999,
            },
            timeout=5,
        )

    assert response.status_code == 200
    assert response.headers["x-llm-output-tokens-capped"] == "true"
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def test_streaming_chat_completion_through_proxy(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path)

    with running_fake_backend(fake_port), running_queue_proxy(
        proxy_port,
        fake_port,
        config_path,
    ):
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json={
                "model": "local-main",
                "messages": [{"role": "user", "content": "stream please"}],
                "stream": True,
            },
            timeout=5,
        ) as response:
            body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data:" in body
    assert "[DONE]" in body


def test_token_budget_excess_returns_413(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path, max_input_tokens=2, max_total_tokens=8)

    with running_fake_backend(fake_port), running_queue_proxy(
        proxy_port,
        fake_port,
        config_path,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json={
                "model": "local-main",
                "messages": [{"role": "user", "content": "x" * 200}],
            },
            timeout=5,
        )

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "token_budget_exceeded"


def test_queue_overflow_returns_429(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    config_path = write_policy_config(
        tmp_path,
        max_active_requests=1,
        max_queued_requests=0,
        queue_timeout_seconds=1,
    )

    async def scenario() -> None:
        async with httpx.AsyncClient(timeout=5) as client:
            first = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                    headers={"x-fake-delay-ms": "500"},
                    json={
                        "model": "local-main",
                        "messages": [{"role": "user", "content": "hold slot"}],
                    },
                )
            )
            await wait_for_active_request(client, proxy_port)
            second = await client.post(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json={
                    "model": "local-main",
                    "messages": [{"role": "user", "content": "overflow"}],
                },
            )
            first_response = await first

        assert first_response.status_code == 200
        assert second.status_code == 429
        assert second.json()["error"]["type"] == "queue_full"

    with running_fake_backend(fake_port), running_queue_proxy(
        proxy_port,
        fake_port,
        config_path,
    ):
        asyncio.run(scenario())


def test_queue_timeout_returns_429(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    config_path = write_policy_config(
        tmp_path,
        max_active_requests=1,
        max_queued_requests=1,
        queue_timeout_seconds=0.1,
    )

    async def scenario() -> None:
        async with httpx.AsyncClient(timeout=5) as client:
            first = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                    headers={"x-fake-delay-ms": "500"},
                    json={
                        "model": "local-main",
                        "messages": [{"role": "user", "content": "hold slot"}],
                    },
                )
            )
            await wait_for_active_request(client, proxy_port)
            second = await client.post(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json={
                    "model": "local-main",
                    "messages": [{"role": "user", "content": "timeout"}],
                },
            )
            first_response = await first

        assert first_response.status_code == 200
        assert second.status_code == 429
        assert second.json()["error"]["type"] == "queue_timeout"

    with running_fake_backend(fake_port), running_queue_proxy(
        proxy_port,
        fake_port,
        config_path,
    ):
        asyncio.run(scenario())


def test_upstream_unavailable_returns_502_and_releases_slot(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    unused_upstream_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path, max_active_requests=1, max_queued_requests=0)

    with running_queue_proxy(
        proxy_port,
        unused_upstream_port,
        config_path,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json={
                "model": "local-main",
                "messages": [{"role": "user", "content": "backend down"}],
            },
            timeout=5,
        )
        status_response = httpx.get(
            f"http://127.0.0.1:{proxy_port}/status",
            timeout=5,
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_request_failed"
    assert status_response.json()["models"][0]["active_requests"] == 0


def test_queue_proxy_routes_through_backend_registry(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    registry_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path, backend_model="actual-main")

    with running_fake_backend(fake_port), running_fake_registry(
        registry_port,
        f"http://127.0.0.1:{fake_port}/v1",
    ), running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
        registry_port=registry_port,
        require_registry_backend=True,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json={
                "model": "local-main",
                "messages": [{"role": "user", "content": "registry route"}],
            },
            timeout=5,
        )
        registry_response = httpx.get(
            f"http://127.0.0.1:{registry_port}/registry",
            timeout=5,
        )

    assert response.status_code == 200
    assert response.json()["model"] == "actual-main"
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert registry_response.json()["instances"][0]["active_requests"] == 0


def test_task_queue_submission_reconciles_capacity(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    registry_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    unused_backend_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path)

    with running_fake_registry(
        registry_port,
        f"http://127.0.0.1:{unused_backend_port}/v1",
    ), running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
        registry_port=registry_port,
        require_registry_backend=True,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/tasks/queue",
            json={
                "model": "local-main",
                "orchestration": {
                    "schema_version": "llmo.task.v1",
                    "tenant": "elvis",
                    "project": "zotero",
                    "service": "zotero-html-translate-worker",
                    "task": "html_translate",
                    "priority": "batch",
                    "gpu": "auto",
                    "lms_gpu": "max",
                    "lms_context_length": 32768,
                    "max_parallel": 4,
                },
                "tasks": [
                    {
                        "job_id": "zotero:item:ABCD1234:source-html:ru",
                        "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                        "tokens": {
                            "estimated_input_tokens": 5200,
                            "max_output_tokens": 1200,
                        },
                        "artifacts": {
                            "input_ref": "file:///data/zotero/ABCD1234/02.en.polish.html",
                            "output_ref": "file:///data/zotero/ABCD1234/03.ru.translate.html",
                        },
                    },
                    {
                        "job_id": "zotero:item:EFGH5678:source-html:ru",
                        "idempotency_key": "zotero:item:EFGH5678:source-html:ru:v1",
                        "tokens": {
                            "estimated_input_tokens": 9200,
                            "max_output_tokens": 1800,
                        },
                        "artifacts": {
                            "input_ref": "file:///data/zotero/EFGH5678/02.en.polish.html",
                            "output_ref": "file:///data/zotero/EFGH5678/03.ru.translate.html",
                        },
                    },
                ],
            },
            timeout=5,
        )

    assert response.status_code == 202
    body = response.json()
    assert body["accepted_tasks"] == 2
    assert body["reused_tasks"] == 0
    assert body["queue_lengths"] == {"local-main": 2}
    assert body["context_plans"]["local-main"]["recommended_lms_context_length"] == 16384
    assert body["context_plans"]["local-main"]["recommended_lms_parallel"] == 2
    assert body["context_plans"]["local-main"]["total_slot_context_tokens"] == 32768
    assert body["capacity"]["state"] == "reconciled"
    assert body["capacity"]["result"]["queue_lengths"] == {"local-main": 2}
    assert body["capacity"]["result"]["context_plans"]["local-main"] == body[
        "context_plans"
    ]["local-main"]
    assert body["capacity"]["result"]["models"][0]["decisions"][0]["gpu_id"] == "gpu0"


def test_task_queue_submission_survives_queue_proxy_restart(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    registry_port = unused_tcp_port_factory()
    first_proxy_port = unused_tcp_port_factory()
    second_proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    unused_backend_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path)
    task_store_path = tmp_path / "task-store.json"
    payload = {
        "model": "local-main",
        "orchestration": {
            "schema_version": "llmo.task.v1",
            "tenant": "elvis",
            "project": "zotero",
            "service": "zotero-html-translate-worker",
            "task": "html_translate",
            "priority": "batch",
            "max_parallel": 4,
        },
        "tasks": [
            {
                "job_id": "zotero:item:ABCD1234:source-html:ru",
                "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                "tokens": {
                    "estimated_input_tokens": 5200,
                    "max_output_tokens": 1200,
                },
            },
            {
                "job_id": "zotero:item:EFGH5678:source-html:ru",
                "idempotency_key": "zotero:item:EFGH5678:source-html:ru:v1",
                "tokens": {
                    "estimated_input_tokens": 9200,
                    "max_output_tokens": 1800,
                },
            },
        ],
    }

    with running_fake_registry(
        registry_port,
        f"http://127.0.0.1:{unused_backend_port}/v1",
    ):
        with running_queue_proxy(
            first_proxy_port,
            unused_static_upstream_port,
            config_path,
            registry_port=registry_port,
            require_registry_backend=True,
            task_store_path=task_store_path,
        ):
            first_response = httpx.post(
                f"http://127.0.0.1:{first_proxy_port}/tasks/queue",
                json=payload,
                timeout=5,
            )

        with running_queue_proxy(
            second_proxy_port,
            unused_static_upstream_port,
            config_path,
            registry_port=registry_port,
            require_registry_backend=True,
            task_store_path=task_store_path,
        ):
            second_response = httpx.post(
                f"http://127.0.0.1:{second_proxy_port}/tasks/queue",
                json=payload,
                timeout=5,
            )

    assert first_response.status_code == 202
    first_body = first_response.json()
    assert first_body["accepted_tasks"] == 2
    assert first_body["reused_tasks"] == 0

    assert second_response.status_code == 202
    second_body = second_response.json()
    assert second_body["accepted_tasks"] == 0
    assert second_body["reused_tasks"] == 2
    assert second_body["queue_lengths"] == {"local-main": 2}
    assert second_body["context_plans"]["local-main"]["queued_tasks"] == 2
    assert [
        task["task_id"] for task in second_body["tasks"]
    ] == [task["task_id"] for task in first_body["tasks"]]


def test_task_status_api_is_tenant_scoped(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path)

    with running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/tasks/queue",
            json={
                "model": "local-main",
                "orchestration": {
                    "schema_version": "llmo.task.v1",
                    "tenant": "elvis",
                    "project": "zotero",
                    "service": "zotero-html-translate-worker",
                    "task": "html_translate",
                    "priority": "batch",
                },
                "tasks": [
                    {
                        "job_id": "zotero:item:ABCD1234:source-html:ru",
                        "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                    }
                ],
            },
            timeout=5,
        )
        task_id = response.json()["tasks"][0]["task_id"]
        list_response = httpx.get(
            f"http://127.0.0.1:{proxy_port}/tasks",
            params={"tenant": "elvis"},
            timeout=5,
        )
        other_tenant_response = httpx.get(
            f"http://127.0.0.1:{proxy_port}/tasks/{task_id}",
            params={"tenant": "other"},
            timeout=5,
        )
        cancel_response = httpx.delete(
            f"http://127.0.0.1:{proxy_port}/tasks/{task_id}",
            headers={"x-tenant-id": "elvis"},
            timeout=5,
        )

    assert response.status_code == 202
    assert list_response.status_code == 200
    assert list_response.json()["tasks"][0]["task_id"] == task_id
    assert other_tenant_response.status_code == 404
    assert other_tenant_response.json()["error"]["type"] == "task_not_found"
    assert cancel_response.status_code == 200
    assert cancel_response.json()["state"] == "cancelled"


def test_task_executor_runs_payload_and_records_result(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path)
    task_store_path = tmp_path / "task-store.json"

    with running_fake_backend(fake_port), running_queue_proxy(
        proxy_port,
        fake_port,
        config_path,
        task_store_path=task_store_path,
        task_executor_enabled=True,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/tasks/queue",
            json={
                "model": "local-main",
                "orchestration": {
                    "schema_version": "llmo.task.v1",
                    "tenant": "elvis",
                    "project": "zotero",
                    "service": "zotero-html-translate-worker",
                    "task": "html_translate",
                    "priority": "batch",
                },
                "tasks": [
                    {
                        "job_id": "zotero:item:ABCD1234:source-html:ru",
                        "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
                        "payload": {
                            "model": "local-main",
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 8,
                        },
                    }
                ],
            },
            timeout=5,
        )
        task_id = response.json()["tasks"][0]["task_id"]
        task_status = wait_for_task_state(proxy_port, task_id, "elvis", "succeeded")
        metrics_response = httpx.get(
            f"http://127.0.0.1:{proxy_port}/metrics",
            timeout=5,
        )

    assert response.status_code == 202
    assert task_status["result"]["status_code"] == 200
    assert task_status["result"]["body"]["choices"][0]["message"]["content"] == "ok"
    metrics_text = metrics_response.text
    assert "llmo_task_events_total" in metrics_text
    assert 'event="succeeded"' in metrics_text
    assert "llmo_tasks_by_state" in metrics_text
    assert 'state="succeeded"' in metrics_text
    assert "llmo_task_execution_seconds_count" in metrics_text


def test_task_executor_runs_employer_payload_template(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path)
    task_store_path = tmp_path / "task-store.json"

    with running_fake_backend(fake_port), running_queue_proxy(
        proxy_port,
        fake_port,
        config_path,
        task_store_path=task_store_path,
        task_executor_enabled=True,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/tasks/queue",
            json={
                "model": "local-main",
                "endpoint": "/v1/chat/completions",
                "payload_template": {
                    "model": "{{model}}",
                    "messages": [
                        {"role": "system", "content": "{{system_prompt}}"},
                        {"role": "user", "content": "Translate: {{text}}"},
                    ],
                    "max_tokens": "{{max_tokens}}",
                },
                "template_vars": {
                    "system_prompt": "Translate scientific HTML to Russian.",
                },
                "orchestration": {
                    "schema_version": "llmo.task.v1",
                    "tenant": "elvis",
                    "project": "zotero",
                    "service": "zotero-html-translate-worker",
                    "task": "html_translate",
                    "priority": "batch",
                },
                "tasks": [
                    {
                        "job_id": "zotero:item:TEMPLATE:source-html:ru",
                        "idempotency_key": "zotero:item:TEMPLATE:source-html:ru:v1",
                        "template_vars": {
                            "text": "<p>Hello.</p>",
                            "max_tokens": 8,
                        },
                    }
                ],
            },
            timeout=5,
        )
        task_id = response.json()["tasks"][0]["task_id"]
        task_status = wait_for_task_state(proxy_port, task_id, "elvis", "succeeded")

    assert response.status_code == 202
    assert task_status["payload"]["messages"][1]["content"] == "Translate: <p>Hello.</p>"
    assert task_status["payload"]["max_tokens"] == 8
    assert task_status["result"]["body"]["choices"][0]["message"]["content"] == "ok"


def test_task_executor_retries_until_transient_backend_is_available(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path)
    task_store_path = tmp_path / "task-store.json"

    with running_queue_proxy(
        proxy_port,
        fake_port,
        config_path,
        request_timeout_seconds=0.2,
        task_store_path=task_store_path,
        task_executor_enabled=True,
        task_executor_max_attempts=5,
        task_executor_retry_base_seconds=0.3,
        task_executor_retry_max_seconds=0.3,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/tasks/queue",
            json={
                "model": "local-main",
                "orchestration": {
                    "schema_version": "llmo.task.v1",
                    "tenant": "elvis",
                    "project": "zotero",
                    "service": "zotero-html-translate-worker",
                    "task": "html_translate",
                    "priority": "batch",
                },
                "tasks": [
                    {
                        "job_id": "zotero:item:RETRY:source-html:ru",
                        "idempotency_key": "zotero:item:RETRY:source-html:ru:v1",
                        "payload": {
                            "model": "local-main",
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 8,
                        },
                    }
                ],
            },
            timeout=5,
        )
        task_id = response.json()["tasks"][0]["task_id"]
        retry_status = wait_for_task_attempt(proxy_port, task_id, "elvis", 1, "queued")

        with running_fake_backend(fake_port):
            task_status = wait_for_task_state(proxy_port, task_id, "elvis", "succeeded")

    assert response.status_code == 202
    assert retry_status["error"]["retryable"] is True
    assert retry_status["error"]["type"] == "upstream_request_failed"
    assert task_status["attempt_count"] >= 2
    assert task_status["result"]["body"]["choices"][0]["message"]["content"] == "ok"


def test_task_executor_does_not_retry_missing_payload(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path)
    task_store_path = tmp_path / "task-store.json"

    with running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
        task_store_path=task_store_path,
        task_executor_enabled=True,
        task_executor_max_attempts=5,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/tasks/queue",
            json={
                "model": "local-main",
                "orchestration": {
                    "schema_version": "llmo.task.v1",
                    "tenant": "elvis",
                    "project": "zotero",
                    "service": "zotero-html-translate-worker",
                    "task": "html_translate",
                    "priority": "batch",
                },
                "tasks": [
                    {
                        "job_id": "zotero:item:NO_PAYLOAD:source-html:ru",
                        "idempotency_key": "zotero:item:NO_PAYLOAD:source-html:ru:v1",
                    }
                ],
            },
            timeout=5,
        )
        task_id = response.json()["tasks"][0]["task_id"]
        task_status = wait_for_task_state(proxy_port, task_id, "elvis", "failed")

    assert response.status_code == 202
    assert task_status["attempt_count"] == 1
    assert task_status["error"]["retryable"] is False
    assert task_status["error"]["type"] == "missing_task_payload"


def test_task_executor_rejects_non_openai_task_payload_without_retry(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path)
    task_store_path = tmp_path / "task-store.json"

    with running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
        task_store_path=task_store_path,
        task_executor_enabled=True,
        task_executor_max_attempts=5,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/tasks/queue",
            json={
                "model": "local-main",
                "endpoint": "/v1/chat/completions",
                "orchestration": {
                    "schema_version": "llmo.task.v1",
                    "tenant": "elvis",
                    "project": "zotero",
                    "service": "zotero-html-translate-worker",
                    "task": "html_translate",
                    "priority": "batch",
                },
                "tasks": [
                    {
                        "job_id": "zotero:item:WORKER_PAYLOAD:source-html:ru",
                        "idempotency_key": "zotero:item:WORKER_PAYLOAD:source-html:ru:v1",
                        "payload": {
                            "run_name": "planned",
                            "source_stage": "02.en.polish.html",
                            "target_stage": "03.ru.translate.html",
                        },
                    }
                ],
            },
            timeout=5,
        )
        task_id = response.json()["tasks"][0]["task_id"]
        task_status = wait_for_task_state(proxy_port, task_id, "elvis", "failed")

    assert response.status_code == 202
    assert task_status["attempt_count"] == 1
    assert task_status["error"]["retryable"] is False
    assert task_status["error"]["type"] == "invalid_task_payload"
    assert task_status["error"]["endpoint"] == "/v1/chat/completions"


def test_client_timeout_before_upstream_headers_releases_registry_lease(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    registry_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path, backend_model="actual-main")

    async def scenario() -> None:
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.ReadTimeout):
                await client.post(
                    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                    headers={"x-fake-delay-ms": "5000"},
                    json={
                        "model": "local-main",
                        "messages": [{"role": "user", "content": "slow registry route"}],
                    },
                    timeout=0.2,
                )

            deadline = time.monotonic() + 3
            last_registry_active = None
            last_proxy_active = None
            while time.monotonic() < deadline:
                registry_response = await client.get(
                    f"http://127.0.0.1:{registry_port}/registry",
                    timeout=1,
                )
                proxy_status = await client.get(
                    f"http://127.0.0.1:{proxy_port}/status",
                    timeout=1,
                )
                last_registry_active = registry_response.json()["instances"][0][
                    "active_requests"
                ]
                proxy_models = proxy_status.json().get("models", [])
                last_proxy_active = proxy_models[0]["active_requests"] if proxy_models else 0
                if last_registry_active == 0 and last_proxy_active == 0:
                    return
                await asyncio.sleep(0.05)

            raise AssertionError(
                "request lease was not released after client timeout: "
                f"registry={last_registry_active} proxy={last_proxy_active}"
            )

    with running_fake_backend(fake_port), running_fake_registry(
        registry_port,
        f"http://127.0.0.1:{fake_port}/v1",
    ), running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
        registry_port=registry_port,
        require_registry_backend=True,
        request_timeout_seconds=10,
    ):
        asyncio.run(scenario())


def test_queue_proxy_allocates_dynamic_model_through_backend_registry(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    registry_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path, dynamic_models_enabled=True)

    with running_fake_backend(fake_port), running_fake_registry(
        registry_port,
        f"http://127.0.0.1:{fake_port}",
    ), running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
        registry_port=registry_port,
        require_registry_backend=True,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json={
                "model": "qwen/qwen3.5-9b",
                "messages": [{"role": "user", "content": "dynamic registry route"}],
                "orchestration": {"gpu": "auto"},
            },
            timeout=5,
        )
        registry_response = httpx.get(
            f"http://127.0.0.1:{registry_port}/registry",
            timeout=5,
        )

    assert response.status_code == 200
    assert response.json()["model"] == "qwen/qwen3.5-9b"
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert registry_response.json()["instances"][0]["model"] == "qwen/qwen3.5-9b"
    assert registry_response.json()["instances"][0]["active_requests"] == 0


def test_queue_proxy_streams_dynamic_model_through_backend_registry(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    registry_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path, dynamic_models_enabled=True)

    with running_fake_backend(fake_port), running_fake_registry(
        registry_port,
        f"http://127.0.0.1:{fake_port}/v1",
    ), running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
        registry_port=registry_port,
        require_registry_backend=True,
    ):
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json={
                "model": "qwen/qwen3.5-9b",
                "messages": [{"role": "user", "content": "stream dynamic"}],
                "stream": True,
                "orchestration": {"gpu": "auto"},
            },
            timeout=5,
        ) as response:
            body = response.read().decode("utf-8")
        registry_response = httpx.get(
            f"http://127.0.0.1:{registry_port}/registry",
            timeout=5,
        )

    assert response.status_code == 200
    assert "data:" in body
    assert "[DONE]" in body
    assert registry_response.json()["instances"][0]["active_requests"] == 0


def test_queue_proxy_routes_dynamic_embeddings_through_backend_registry(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    registry_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path, dynamic_models_enabled=True)

    with running_fake_backend(fake_port), running_fake_registry(
        registry_port,
        f"http://127.0.0.1:{fake_port}/v1",
    ), running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
        registry_port=registry_port,
        require_registry_backend=True,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/embeddings",
            json={
                "model": "text-embedding-bge-m3",
                "input": "embed this",
                "orchestration": {"gpu": "auto"},
            },
            timeout=5,
        )

    assert response.status_code == 200
    assert response.json()["model"] == "text-embedding-bge-m3"
    assert response.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_queue_proxy_returns_503_when_dynamic_allocation_is_denied(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    fake_port = unused_tcp_port_factory()
    registry_port = unused_tcp_port_factory()
    proxy_port = unused_tcp_port_factory()
    unused_static_upstream_port = unused_tcp_port_factory()
    config_path = write_policy_config(tmp_path, dynamic_models_enabled=True)

    with running_fake_backend(fake_port), running_fake_registry(
        registry_port,
        f"http://127.0.0.1:{fake_port}/v1",
        denied_model="denied-model",
    ), running_queue_proxy(
        proxy_port,
        unused_static_upstream_port,
        config_path,
        registry_port=registry_port,
        require_registry_backend=True,
    ):
        response = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json={
                "model": "denied-model",
                "messages": [{"role": "user", "content": "blocked"}],
            },
            timeout=5,
        )

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "no_ready_backend"


def test_lifecycle_cleanup_drains_idle_dynamic_lmstudio_allocation(
    tmp_path: Path,
    unused_tcp_port_factory: object,
) -> None:
    lifecycle_port = unused_tcp_port_factory()
    config_path = tmp_path / "orchestrator.yaml"
    registry_path = tmp_path / "registry.json"
    old_timestamp = "2026-01-01T00:00:00+00:00"
    config_path.write_text(
        "\n".join(
            [
                "dynamic_models:",
                "  enabled: true",
                "  registry_cleanup_ttl_seconds: 3600",
                "  lifecycle:",
                "    runtime: lmstudio",
                "    base_url: http://host.docker.internal:1234/v1",
                "    idle_ttl_seconds: 1",
                "    min_replicas: 0",
            ]
        ),
        encoding="utf-8",
    )
    registry_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "instance_id": "idle-dynamic",
                        "model": "qwen/qwen3.5-9b",
                        "backend_model": "qwen/qwen3.5-9b",
                        "runtime": "lmstudio",
                        "base_url": "http://host.docker.internal:1234/v1",
                        "gpu_ids": ["gpu0"],
                        "state": "ready",
                        "reserved_vram_mb": 1024,
                        "active_requests": 0,
                        "created_at": old_timestamp,
                        "updated_at": old_timestamp,
                        "last_used_at": old_timestamp,
                        "dry_run": True,
                        "metadata": {"idle_ttl_seconds": 1},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with running_lifecycle(lifecycle_port, config_path, registry_path):
        response = httpx.post(
            f"http://127.0.0.1:{lifecycle_port}/cleanup",
            json={},
            timeout=5,
        )

    assert response.status_code == 200
    assert response.json()["stopped_instances"][0]["instance_id"] == "idle-dynamic"
    assert response.json()["stopped_instances"][0]["state"] == "stopped"


def wait_for_task_state(
    proxy_port: int,
    task_id: str,
    tenant: str,
    state: str,
) -> dict[str, object]:
    deadline = time.monotonic() + 5
    last_body: dict[str, object] | None = None
    while time.monotonic() < deadline:
        response = httpx.get(
            f"http://127.0.0.1:{proxy_port}/tasks/{task_id}",
            params={"tenant": tenant},
            timeout=5,
        )
        response.raise_for_status()
        last_body = response.json()
        if last_body.get("state") == state:
            return last_body
        time.sleep(0.05)
    raise AssertionError(f"Task {task_id} did not reach {state}: {last_body}")


def wait_for_task_attempt(
    proxy_port: int,
    task_id: str,
    tenant: str,
    min_attempt_count: int,
    state: str | None = None,
) -> dict[str, object]:
    deadline = time.monotonic() + 5
    last_body: dict[str, object] | None = None
    while time.monotonic() < deadline:
        response = httpx.get(
            f"http://127.0.0.1:{proxy_port}/tasks/{task_id}",
            params={"tenant": tenant},
            timeout=5,
        )
        response.raise_for_status()
        last_body = response.json()
        state_matches = state is None or last_body.get("state") == state
        if int(last_body.get("attempt_count", 0)) >= min_attempt_count and state_matches:
            return last_body
        time.sleep(0.05)
    raise AssertionError(
        f"Task {task_id} did not reach attempt {min_attempt_count}: {last_body}"
    )
