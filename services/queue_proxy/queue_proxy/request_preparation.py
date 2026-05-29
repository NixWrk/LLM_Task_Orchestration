from __future__ import annotations

import json
from typing import Any, Mapping

from queue_proxy.policy import (
    ModelPolicy,
    PolicyError,
    PolicyRegistry,
    apply_orchestration_overrides,
    apply_token_policy,
    extract_model,
)


class RequestPreparationService:
    def __init__(self, policy_registry: PolicyRegistry) -> None:
        self.policy_registry = policy_registry

    def prepare(
        self,
        path: str,
        method: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> tuple[dict[str, Any] | None, dict[str, Any], ModelPolicy | None]:
        if method.upper() != "POST":
            return None, {}, None

        content_type = headers.get("content-type", "")
        if "application/json" not in content_type:
            return None, {}, None

        if not is_llm_generation_endpoint(path):
            return None, {}, None

        try:
            raw_payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise PolicyError(f"Invalid JSON request body: {exc.msg}", "invalid_json") from exc

        if not isinstance(raw_payload, dict):
            raise PolicyError("JSON request body must be an object.", "invalid_json")

        policy = self.policy_registry.resolve(extract_model(raw_payload))
        policy = apply_orchestration_overrides(policy, raw_payload.get("orchestration"))
        payload = apply_token_policy(raw_payload, policy)
        metadata = dict(payload.get("_orchestrator") or {})
        return payload, metadata, policy


def is_llm_generation_endpoint(path: str) -> bool:
    normalized = path.strip("/")
    return normalized in {
        "chat/completions",
        "responses",
        "completions",
        "embeddings",
    }
