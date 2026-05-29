from orchestrator_core.openai import openai_url
from queue_proxy.http_proxy import upstream_headers, upstream_url


def test_openai_url_deduplicates_v1_prefix() -> None:
    assert openai_url("http://backend:8000/v1", "/v1/models") == (
        "http://backend:8000/v1/models"
    )


def test_upstream_url_adds_v1_when_missing() -> None:
    assert upstream_url("http://litellm:4000", "chat/completions") == (
        "http://litellm:4000/v1/chat/completions"
    )


def test_upstream_headers_drop_hop_by_hop_and_apply_api_key() -> None:
    headers = upstream_headers(
        {
            "host": "localhost",
            "content-type": "application/json",
            "connection": "keep-alive",
        },
        upstream_api_key="sk-test",
    )

    assert headers == {
        "content-type": "application/json",
        "authorization": "Bearer sk-test",
    }
