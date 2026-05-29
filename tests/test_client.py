from typing import Any

from orchestrator_client import OrchestratorClient, join_url


def test_client_chat_stream_uses_queue_url_and_api_key(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None, str | None, float]] = []

    def fake_request_text(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 60,
    ) -> str:
        calls.append((method, url, payload, api_key, timeout_seconds))
        return "data: ok"

    monkeypatch.setattr("orchestrator_client.client.request_text", fake_request_text)

    result = OrchestratorClient(
        queue_url="http://queue/",
        api_key="sk-test",
        timeout_seconds=12,
    ).chat("qwen", "hello", stream=True, orchestration={"gpu": "auto"})

    assert result == "data: ok"
    assert calls[0][0] == "POST"
    assert calls[0][1] == "http://queue/v1/chat/completions"
    assert calls[0][2]["orchestration"] == {"gpu": "auto"}
    assert calls[0][3] == "sk-test"
    assert calls[0][4] == 12


def test_join_url_normalizes_slashes() -> None:
    assert join_url("http://localhost:4100/", "/v1/models") == (
        "http://localhost:4100/v1/models"
    )
