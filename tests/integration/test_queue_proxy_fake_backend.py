from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

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
