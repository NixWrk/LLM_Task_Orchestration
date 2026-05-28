from datetime import UTC, datetime, timedelta

from lifecycle.controller import idle_seconds, openai_url
from lifecycle.models import BackendInstance


def test_openai_url_does_not_duplicate_v1_prefix() -> None:
    assert openai_url("http://backend:8000/v1", "/v1/models") == (
        "http://backend:8000/v1/models"
    )
    assert openai_url("http://backend:8000/v1", "/chat/completions") == (
        "http://backend:8000/v1/chat/completions"
    )


def test_idle_seconds_uses_last_used_at_when_available() -> None:
    last_used = datetime.now(UTC) - timedelta(seconds=30)
    instance = BackendInstance(
        instance_id="backend-1",
        model="local-main",
        backend_model="local-main",
        runtime="vllm",
        base_url="http://backend:8000/v1",
        gpu_ids=["gpu0"],
        state="ready",
        reserved_vram_mb=1024,
        last_used_at=last_used.isoformat(),
    )

    assert idle_seconds(instance) >= 29
