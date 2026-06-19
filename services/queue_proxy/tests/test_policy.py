from pathlib import Path

import pytest

from queue_proxy.policy import (
    ModelPolicy,
    PolicyError,
    PolicyRegistry,
    TokenBudgetExceeded,
    apply_orchestration_overrides,
    apply_token_policy,
    extract_model,
    load_policy_registry,
    strip_internal_fields,
    validate_orchestration_contract,
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
    assert strip_internal_fields(
        {"model": "local-main", "_orchestrator": {}, "orchestration": {"gpu": "auto"}}
    ) == {
        "model": "local-main"
    }


def test_policy_registry_builds_dynamic_policy_for_requested_model() -> None:
    registry = PolicyRegistry({"local-main": policy()}, policy(), dynamic_models_enabled=True)

    resolved = registry.resolve("qwen/qwen3.5-9b")

    assert resolved.public_name == "qwen/qwen3.5-9b"
    assert resolved.backend_model == "qwen/qwen3.5-9b"
    assert resolved.max_active_requests == policy().max_active_requests


def test_apply_orchestration_overrides_can_lower_request_limits() -> None:
    resolved = apply_orchestration_overrides(
        policy(),
        {
            "max_parallel": 3,
            "max_queued_requests": 1,
            "tokens": {"max_output_tokens": 32},
        },
    )

    assert resolved.max_active_requests == 1
    assert resolved.max_queued_requests == 1
    assert resolved.max_output_tokens == 32


def test_validate_orchestration_contract_accepts_strict_v1_envelope() -> None:
    validate_orchestration_contract(
        {
            "schema_version": "llmo.task.v1",
            "tenant": "elvis",
            "project": "zotero",
            "service": "worker",
            "task": "html_translate",
            "job_id": "job-1",
            "priority": "batch",
            "gpu": ["gpu0", "gpu1"],
            "max_parallel": 1,
            "tokens": {"max_output_tokens": 128},
            "artifacts": {"input_ref": "file:///tmp/input.html"},
            "labels": {"domain": "scientific_html"},
        }
    )


def test_validate_orchestration_contract_rejects_malformed_v1_envelope() -> None:
    with pytest.raises(PolicyError, match="orchestration.tenant"):
        validate_orchestration_contract(
            {
                "schema_version": "llmo.task.v1",
                "project": "zotero",
                "service": "worker",
                "task": "html_translate",
                "job_id": "job-1",
                "priority": "batch",
            }
        )

    with pytest.raises(PolicyError, match="Unsupported orchestration.schema_version"):
        validate_orchestration_contract({"schema_version": "llmo.task.v2"})

    with pytest.raises(PolicyError, match="orchestration.priority"):
        validate_orchestration_contract(
            {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "worker",
                "task": "html_translate",
                "job_id": "job-1",
                "priority": "right-now",
            }
        )

    with pytest.raises(PolicyError, match="orchestration.max_parallel"):
        validate_orchestration_contract(
            {
                "schema_version": "llmo.task.v1",
                "tenant": "elvis",
                "project": "zotero",
                "service": "worker",
                "task": "html_translate",
                "job_id": "job-1",
                "priority": "batch",
                "max_parallel": 0,
            }
        )


def test_repository_zotero_html_translate_policy_allows_two_active_requests() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "orchestrator.yaml"

    policy = load_policy_registry(str(config_path)).resolve("zotero-html-translate")

    assert policy.max_active_requests == 2
    assert policy.max_queued_requests == 64
    assert policy.max_input_tokens == 32768
    assert policy.max_output_tokens == 8192
    assert policy.max_total_tokens == 40960
