from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from orchestrator_core.config import load_orchestrator_config


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
    def __init__(
        self,
        policies: dict[str, ModelPolicy],
        default_policy: ModelPolicy,
        dynamic_models_enabled: bool = False,
    ) -> None:
        self._policies = policies
        self._default_policy = default_policy
        self._dynamic_models_enabled = dynamic_models_enabled

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
            if self._dynamic_models_enabled:
                return dynamic_policy(model, self._default_policy)
        return self._default_policy


def load_policy_registry(path: str) -> PolicyRegistry:
    config = load_orchestrator_config(path)
    policies: dict[str, ModelPolicy] = {}

    for model_key, model_config in config.models.items():
        model_data = {**config.defaults, **(model_config or {})}
        policy = model_policy_from_data(model_key, model_data)
        policies[policy.public_name] = policy

    if not policies:
        default_policy = ModelPolicy(
            public_name="default",
            backend_model="default",
            aliases=(),
            max_active_requests=int(config.defaults.get("max_active_requests", 1)),
            max_queued_requests=int(config.defaults.get("max_queued_requests", 16)),
            queue_timeout_seconds=float(config.defaults.get("queue_timeout_seconds", 30)),
            default_max_output_tokens=int(
                config.defaults.get("default_max_output_tokens", 512)
            ),
            max_input_tokens=int(config.defaults.get("max_input_tokens", 8192)),
            max_output_tokens=int(config.defaults.get("max_output_tokens", 1024)),
            max_total_tokens=int(config.defaults.get("max_total_tokens", 9216)),
            token_estimate_chars_per_token=float(
                config.defaults.get("token_estimate_chars_per_token", 4)
            ),
        )
        return PolicyRegistry(
            {"default": default_policy},
            default_policy,
            dynamic_models_enabled=bool(config.dynamic_models.get("enabled", False)),
        )

    default_policy = model_policy_from_data("default", config.defaults)
    return PolicyRegistry(
        policies,
        default_policy,
        dynamic_models_enabled=bool(config.dynamic_models.get("enabled", False)),
    )


def model_policy_from_data(model_key: str, model_data: dict[str, Any]) -> ModelPolicy:
    public_name = str(model_data.get("public_name") or model_key)
    aliases = tuple(str(alias) for alias in model_data.get("aliases", []))
    return ModelPolicy(
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
        output_over_limit_behavior=str(model_data.get("output_over_limit_behavior", "clamp")),
    )


def dynamic_policy(model: str, default_policy: ModelPolicy) -> ModelPolicy:
    return ModelPolicy(
        public_name=model,
        backend_model=model,
        aliases=(),
        max_active_requests=default_policy.max_active_requests,
        max_queued_requests=default_policy.max_queued_requests,
        queue_timeout_seconds=default_policy.queue_timeout_seconds,
        default_max_output_tokens=default_policy.default_max_output_tokens,
        max_input_tokens=default_policy.max_input_tokens,
        max_output_tokens=default_policy.max_output_tokens,
        max_total_tokens=default_policy.max_total_tokens,
        token_estimate_chars_per_token=default_policy.token_estimate_chars_per_token,
        output_over_limit_behavior=default_policy.output_over_limit_behavior,
    )


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


def apply_orchestration_overrides(
    policy: ModelPolicy,
    orchestration: Any,
) -> ModelPolicy:
    if not isinstance(orchestration, dict):
        return policy

    token_overrides = orchestration.get("tokens")
    if not isinstance(token_overrides, dict):
        token_overrides = {}

    changes: dict[str, Any] = {}
    bounded_int_override(
        changes,
        "max_active_requests",
        orchestration.get("max_parallel", orchestration.get("max_active_requests")),
        policy.max_active_requests,
    )
    bounded_int_override(
        changes,
        "max_queued_requests",
        orchestration.get("max_queued_requests"),
        policy.max_queued_requests,
    )
    bounded_float_override(
        changes,
        "queue_timeout_seconds",
        orchestration.get("queue_timeout_seconds"),
        policy.queue_timeout_seconds,
    )
    bounded_int_override(
        changes,
        "default_max_output_tokens",
        token_overrides.get(
            "default_max_output_tokens",
            orchestration.get("default_max_output_tokens"),
        ),
        policy.default_max_output_tokens,
    )
    bounded_int_override(
        changes,
        "max_input_tokens",
        token_overrides.get("max_input_tokens", orchestration.get("max_input_tokens")),
        policy.max_input_tokens,
    )
    bounded_int_override(
        changes,
        "max_output_tokens",
        token_overrides.get("max_output_tokens", orchestration.get("max_output_tokens")),
        policy.max_output_tokens,
    )
    bounded_int_override(
        changes,
        "max_total_tokens",
        token_overrides.get("max_total_tokens", orchestration.get("max_total_tokens")),
        policy.max_total_tokens,
    )
    return replace(policy, **changes) if changes else policy


def strip_internal_fields(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    cleaned.pop("_orchestrator", None)
    cleaned.pop("orchestration", None)
    return cleaned


def bounded_int_override(
    changes: dict[str, Any],
    field: str,
    raw_value: Any,
    upper_bound: int,
) -> None:
    if raw_value in (None, ""):
        return
    value = max(1, int(raw_value))
    changes[field] = min(value, upper_bound)


def bounded_float_override(
    changes: dict[str, Any],
    field: str,
    raw_value: Any,
    upper_bound: float,
) -> None:
    if raw_value in (None, ""):
        return
    value = max(0.001, float(raw_value))
    changes[field] = min(value, upper_bound)


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
