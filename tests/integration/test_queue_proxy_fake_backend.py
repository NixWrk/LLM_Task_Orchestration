from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


class RunningProcess:
    def __init__(self, process: subprocess.Popen[bytes], name: str) -> None:
        self.process = process
        self.name = name

    def stop(self) -> None:
        if self.process.poll() is not None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


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


@contextmanager
def running_fake_backend(port: int) -> Iterator[None]:
    env = service_env("services/fake_openai_backend")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "fake_openai_backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    runner = RunningProcess(process, "fake-openai-backend")
    try:
        wait_for_http(f"http://127.0.0.1:{port}/health", process, runner.name)
        yield
    finally:
        runner.stop()


@contextmanager
def running_queue_proxy(
    port: int,
    upstream_port: int,
    config_path: Path,
    registry_port: int | None = None,
    require_registry_backend: bool = False,
) -> Iterator[None]:
    env = service_env("services/queue_proxy")
    env["QUEUE_PROXY_CONFIG_PATH"] = str(config_path)
    env["UPSTREAM_LITELLM_BASE_URL"] = f"http://127.0.0.1:{upstream_port}"
    env["LITELLM_MASTER_KEY"] = "test-key"
    env["REQUEST_TIMEOUT_SECONDS"] = "1"
    if registry_port is not None:
        env["BACKEND_REGISTRY_URL"] = f"http://127.0.0.1:{registry_port}"
        env["ENABLE_BACKEND_REGISTRY_ROUTING"] = "true"
        env["REQUIRE_BACKEND_REGISTRY_BACKEND"] = str(require_registry_backend).lower()

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "queue_proxy.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    runner = RunningProcess(process, "queue-proxy")
    try:
        wait_for_http(f"http://127.0.0.1:{port}/health", process, runner.name)
        yield
    finally:
        runner.stop()


@contextmanager
def running_fake_registry(
    port: int,
    backend_url: str,
    denied_model: str | None = None,
) -> Iterator[None]:
    env = service_env("services/queue_proxy")
    env["FAKE_REGISTRY_BACKEND_URL"] = backend_url
    if denied_model:
        env["FAKE_REGISTRY_DENIED_MODEL"] = denied_model

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.support.fake_registry_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    runner = RunningProcess(process, "fake-registry")
    try:
        wait_for_http(f"http://127.0.0.1:{port}/health", process, runner.name)
        yield
    finally:
        runner.stop()


@contextmanager
def running_lifecycle(
    port: int,
    config_path: Path,
    registry_path: Path,
) -> Iterator[None]:
    env = service_env("services/lifecycle")
    env["LIFECYCLE_CONFIG_PATH"] = str(config_path)
    env["BACKEND_REGISTRY_PATH"] = str(registry_path)
    env["LIFECYCLE_DRY_RUN"] = "true"
    env["LIFECYCLE_ENABLE_RECONCILE_LOOP"] = "false"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "lifecycle.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    runner = RunningProcess(process, "lifecycle")
    try:
        wait_for_http(f"http://127.0.0.1:{port}/health", process, runner.name)
        yield
    finally:
        runner.stop()


def write_policy_config(
    tmp_path: Path,
    *,
    max_active_requests: int = 1,
    max_queued_requests: int = 1,
    queue_timeout_seconds: float = 1,
    default_max_output_tokens: int = 16,
    max_input_tokens: int = 128,
    max_output_tokens: int = 32,
    max_total_tokens: int = 256,
    backend_model: str = "local-main",
    dynamic_models_enabled: bool = False,
) -> Path:
    config_path = tmp_path / "orchestrator.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                f"  max_active_requests: {max_active_requests}",
                f"  max_queued_requests: {max_queued_requests}",
                f"  queue_timeout_seconds: {queue_timeout_seconds}",
                f"  default_max_output_tokens: {default_max_output_tokens}",
                f"  max_input_tokens: {max_input_tokens}",
                f"  max_output_tokens: {max_output_tokens}",
                f"  max_total_tokens: {max_total_tokens}",
                "  token_estimate_chars_per_token: 4",
                "  output_over_limit_behavior: clamp",
                "dynamic_models:",
                f"  enabled: {str(dynamic_models_enabled).lower()}",
                "models:",
                "  local-main:",
                "    public_name: local-main",
                f"    backend_model: {backend_model}",
                "    aliases:",
                "      - local-main",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def service_env(*relative_paths: str) -> dict[str, str]:
    env = os.environ.copy()
    service_paths = [str(ROOT), *(str(ROOT / relative_path) for relative_path in relative_paths)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        service_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(service_paths)
    return env


def wait_for_http(url: str, process: subprocess.Popen[bytes], name: str) -> None:
    deadline = time.monotonic() + 10
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise RuntimeError(
                f"{name} exited early with code {process.returncode}.\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            )

        try:
            response = httpx.get(url, timeout=0.25)
            if response.status_code < 500:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.05)

    raise TimeoutError(f"Timed out waiting for {name} at {url}: {last_error}")


async def wait_for_active_request(client: httpx.AsyncClient, proxy_port: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = await client.get(f"http://127.0.0.1:{proxy_port}/status")
        models = response.json().get("models", [])
        if models and models[0].get("active_requests") == 1:
            return
        await asyncio.sleep(0.02)

    raise TimeoutError("Timed out waiting for an active queue-proxy request.")
