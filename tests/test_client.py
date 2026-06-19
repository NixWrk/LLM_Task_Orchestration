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


def test_client_submit_task_queue_uses_queue_endpoint(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None, str | None, float]] = []

    def fake_request_json(
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 240,
    ) -> dict[str, str]:
        calls.append((method, url, payload, api_key, timeout_seconds))
        return {"status": "accepted"}

    monkeypatch.setattr("orchestrator_client.client.request_json", fake_request_json)

    result = OrchestratorClient(
        queue_url="http://queue/",
        api_key="sk-test",
        timeout_seconds=12,
    ).submit_task_queue(
        model="zotero-html-translate",
        orchestration={
            "schema_version": "llmo.task.v1",
            "tenant": "elvis",
            "project": "zotero",
            "service": "zotero-html-translate-worker",
            "task": "html_translate",
            "priority": "batch",
        },
        tasks=[
            {
                "job_id": "zotero:item:ABCD1234:source-html:ru",
                "idempotency_key": "zotero:item:ABCD1234:source-html:ru:v1",
            }
        ],
    )

    assert result == {"status": "accepted"}
    assert calls[0][0] == "POST"
    assert calls[0][1] == "http://queue/tasks/queue"
    assert calls[0][2]["model"] == "zotero-html-translate"
    assert calls[0][2]["tasks"][0]["job_id"] == "zotero:item:ABCD1234:source-html:ru"
    assert calls[0][3] == "sk-test"
    assert calls[0][4] == 12


def test_join_url_normalizes_slashes() -> None:
    assert join_url("http://localhost:4100/", "/v1/models") == (
        "http://localhost:4100/v1/models"
    )
