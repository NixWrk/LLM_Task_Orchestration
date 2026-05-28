from __future__ import annotations

import asyncio
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
    config_path = write_policy_config(tmp_path)

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
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert registry_response.json()["instances"][0]["active_requests"] == 0


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
) -> Iterator[None]:
    env = service_env("services/queue_proxy")
    env["FAKE_REGISTRY_BACKEND_URL"] = backend_url

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
                "models:",
                "  local-main:",
                "    public_name: local-main",
                "    backend_model: local-main",
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
