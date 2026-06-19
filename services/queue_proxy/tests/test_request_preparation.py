import pytest

from queue_proxy.policy import ModelPolicy, PolicyError, PolicyRegistry
from queue_proxy.request_preparation import RequestPreparationService, should_stream_response


def test_request_preparer_applies_token_policy() -> None:
    payload, metadata, policy = RequestPreparationService(policy_registry()).prepare(
        "chat/completions",
        "POST",
        {"content-type": "application/json"},
        b'{"model":"local-main","messages":[{"role":"user","content":"hello"}]}',
    )

    assert payload["max_tokens"] == 64
    assert metadata["estimated_input_tokens"] == 1
    assert policy.public_name == "local-main"


def test_request_preparer_skips_non_generation_endpoint() -> None:
    payload, metadata, policy = RequestPreparationService(policy_registry()).prepare(
        "models",
        "GET",
        {},
        b"",
    )

    assert payload is None
    assert metadata == {}
    assert policy is None


def test_request_preparer_rejects_invalid_json() -> None:
    with pytest.raises(PolicyError):
        RequestPreparationService(policy_registry()).prepare(
            "chat/completions",
            "POST",
            {"content-type": "application/json"},
            b"{",
        )


def test_request_preparer_rejects_malformed_v1_orchestration() -> None:
    with pytest.raises(PolicyError) as exc:
        RequestPreparationService(policy_registry()).prepare(
            "chat/completions",
            "POST",
            {"content-type": "application/json"},
            (
                b'{"model":"local-main","messages":[{"role":"user","content":"hello"}],'
                b'"orchestration":{"schema_version":"llmo.task.v1","priority":"batch"}}'
            ),
        )

    assert exc.value.error_type == "invalid_task_protocol"


def test_should_stream_response_requires_explicit_true() -> None:
    assert should_stream_response({"stream": True}) is True
    assert should_stream_response({"stream": False}) is False
    assert should_stream_response({}) is False
    assert should_stream_response({"stream": "true"}) is False


def policy_registry() -> PolicyRegistry:
    policy = ModelPolicy(
        public_name="local-main",
        backend_model="local-main",
        aliases=(),
        max_active_requests=1,
        max_queued_requests=2,
        queue_timeout_seconds=1,
        default_max_output_tokens=64,
        max_input_tokens=100,
        max_output_tokens=128,
        max_total_tokens=180,
        token_estimate_chars_per_token=4,
    )
    return PolicyRegistry({"local-main": policy}, policy)
