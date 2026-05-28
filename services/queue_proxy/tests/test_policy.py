import pytest

from queue_proxy.policy import (
    ModelPolicy,
    TokenBudgetExceeded,
    apply_token_policy,
    extract_model,
    strip_internal_fields,
)


def policy() -> ModelPolicy:
    return ModelPolicy(
        public_name="local-main",
        backend_model="local-main",
        aliases=("main",),
        max_active_requests=1,
        max_queued_requests=2,
        queue_timeout_seconds=1,
        default_max_output_tokens=64,
        max_input_tokens=100,
        max_output_tokens=128,
        max_total_tokens=180,
        token_estimate_chars_per_token=4,
    )


def test_extract_model_from_payload() -> None:
    assert extract_model({"model": "local-main"}) == "local-main"


def test_apply_token_policy_sets_default_output_tokens() -> None:
    payload = apply_token_policy(
        {"model": "local-main", "messages": [{"role": "user", "content": "hello"}]},
        policy(),
    )

    assert payload["max_tokens"] == 64
    assert payload["_orchestrator"]["estimated_input_tokens"] == 1


def test_apply_token_policy_clamps_output_tokens() -> None:
    payload = apply_token_policy(
        {
            "model": "local-main",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 999,
        },
        policy(),
    )

    assert payload["max_tokens"] == 128
    assert payload["_orchestrator"]["output_tokens_capped"] is True


def test_apply_token_policy_rejects_too_large_input() -> None:
    with pytest.raises(TokenBudgetExceeded):
        apply_token_policy(
            {
                "model": "local-main",
                "messages": [{"role": "user", "content": "x" * 1000}],
            },
            policy(),
        )


def test_strip_internal_fields_removes_orchestrator_metadata() -> None:
    assert strip_internal_fields({"model": "local-main", "_orchestrator": {}}) == {
        "model": "local-main"
    }
