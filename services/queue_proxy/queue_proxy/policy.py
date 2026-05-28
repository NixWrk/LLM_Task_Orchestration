from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class PolicyError(Exception):
    status_code = 400

    def __init__(self, message: str, error_type: str = "policy_error") -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type


class TokenBudgetExceeded(PolicyError):
    status_code = 413

    def __init__(self, message: str) -> None:
        super().__init__(message, "token_budget_exceeded")


@dataclass(frozen=True)
class ModelPolicy:
    public_name: str
    backend_model: str
    aliases: tuple[str, ...]
    max_active_requests: int
    max_queued_requests: int
    queue_timeout_seconds: float
    default_max_output_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    token_estimate_chars_per_token: float
    output_over_limit_behavior: str = "clamp"

    def matches(self, model: str) -> bool:
        return model == self.public_name or model in self.aliases


class PolicyRegistry:
    def __init__(self, policies: dict[str, ModelPolicy], default_policy: ModelPolicy) -> None:
        self._policies = policies
        self._default_policy = default_policy

    @property
    def policies(self) -> dict[str, ModelPolicy]:
        return self._policies

    @property
    def default_policy(self) -> ModelPolicy:
        return self._default_policy

    def resolve(self, model: str | None) -> ModelPolicy:
        if model:
            for policy in self._policies.values():
                if policy.matches(model):
                    return policy
        return self._default_policy


def load_policy_registry(path: str) -> PolicyRegistry:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    defaults = raw.get("defaults") or {}
    models = raw.get("models") or {}
    policies: dict[str, ModelPolicy] = {}

    for model_key, model_config in models.items():
        model_data = {**defaults, **(model_config or {})}
        public_name = str(model_data.get("public_name") or model_key)
        aliases = tuple(str(alias) for alias in model_data.get("aliases", []))
        policies[public_name] = ModelPolicy(
            public_name=public_name,
            backend_model=str(model_data.get("backend_model") or public_name),
            aliases=aliases,
            max_active_requests=int(model_data.get("max_active_requests", 1)),
            max_queued_requests=int(model_data.get("max_queued_requests", 16)),
            queue_timeout_seconds=float(model_data.get("queue_timeout_seconds", 30)),
            default_max_output_tokens=int(model_data.get("default_max_output_tokens", 512)),
            max_input_tokens=int(model_data.get("max_input_tokens", 8192)),
            max_output_tokens=int(model_data.get("max_output_tokens", 1024)),
            max_total_tokens=int(model_data.get("max_total_tokens", 9216)),
            token_estimate_chars_per_token=float(
                model_data.get("token_estimate_chars_per_token", 4)
            ),
            output_over_limit_behavior=str(
                model_data.get("output_over_limit_behavior", "clamp")
            ),
        )

    if not policies:
        default_policy = ModelPolicy(
            public_name="default",
            backend_model="default",
            aliases=(),
            max_active_requests=int(defaults.get("max_active_requests", 1)),
            max_queued_requests=int(defaults.get("max_queued_requests", 16)),
            queue_timeout_seconds=float(defaults.get("queue_timeout_seconds", 30)),
            default_max_output_tokens=int(defaults.get("default_max_output_tokens", 512)),
            max_input_tokens=int(defaults.get("max_input_tokens", 8192)),
            max_output_tokens=int(defaults.get("max_output_tokens", 1024)),
            max_total_tokens=int(defaults.get("max_total_tokens", 9216)),
            token_estimate_chars_per_token=float(
                defaults.get("token_estimate_chars_per_token", 4)
            ),
        )
        return PolicyRegistry({"default": default_policy}, default_policy)

    first_policy = next(iter(policies.values()))
    return PolicyRegistry(policies, first_policy)


def extract_model(payload: Any) -> str | None:
    if isinstance(payload, dict):
        model = payload.get("model")
        if model is not None:
            return str(model)
    return None


def apply_token_policy(payload: dict[str, Any], policy: ModelPolicy) -> dict[str, Any]:
    updated = dict(payload)
    input_tokens = estimate_input_tokens(updated, policy)
    output_key = output_token_key(updated)
    capped_output = False

    if input_tokens > policy.max_input_tokens:
        raise TokenBudgetExceeded(
            f"Estimated input tokens {input_tokens} exceed max_input_tokens "
            f"{policy.max_input_tokens} for model {policy.public_name}."
        )

    requested_output = updated.get(output_key)
    if requested_output is None:
        requested_output = policy.default_max_output_tokens
        updated[output_key] = requested_output
    else:
        requested_output = int(requested_output)

    if requested_output > policy.max_output_tokens:
        if policy.output_over_limit_behavior == "reject":
            raise TokenBudgetExceeded(
                f"Requested output tokens {requested_output} exceed max_output_tokens "
                f"{policy.max_output_tokens} for model {policy.public_name}."
            )
        requested_output = policy.max_output_tokens
        updated[output_key] = requested_output
        capped_output = True

    remaining_output = policy.max_total_tokens - input_tokens
    if remaining_output < 1:
        raise TokenBudgetExceeded(
            f"Estimated input tokens {input_tokens} leave no room within max_total_tokens "
            f"{policy.max_total_tokens} for model {policy.public_name}."
        )

    if requested_output > remaining_output:
        requested_output = remaining_output
        updated[output_key] = requested_output
        capped_output = True

    updated["_orchestrator"] = {
        "estimated_input_tokens": input_tokens,
        "effective_output_tokens": requested_output,
        "output_tokens_capped": capped_output,
    }
    return updated


def strip_internal_fields(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    cleaned.pop("_orchestrator", None)
    return cleaned


def output_token_key(payload: dict[str, Any]) -> str:
    if "max_completion_tokens" in payload:
        return "max_completion_tokens"
    if "max_output_tokens" in payload:
        return "max_output_tokens"
    return "max_tokens"


def estimate_input_tokens(payload: dict[str, Any], policy: ModelPolicy) -> int:
    text = collect_input_text(payload)
    if not text:
        return 0
    return max(1, round(len(text) / policy.token_estimate_chars_per_token))


def collect_input_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(collect_input_text(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("messages", "input", "prompt", "content", "text"):
            if key in value:
                parts.append(collect_input_text(value[key]))
        if parts:
            return "\n".join(parts)
        return ""
    return ""
