from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any


def dynamic_model_allowed(model: str, config: dict[str, Any]) -> bool:
    if not bool(config.get("enabled", False)):
        return False

    denied = pattern_list(config, "denied_models", "deny_models", "denied_model_patterns")
    if any(pattern_matches(model, pattern) for pattern in denied):
        return False

    allowed = pattern_list(config, "allowed_models", "allow_models", "allowed_model_patterns")
    if not allowed:
        return True
    return any(pattern_matches(model, pattern) for pattern in allowed)


def pattern_list(config: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = config.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
        elif raw not in (None, ""):
            values.append(str(raw))
    return values


def pattern_matches(value: str, pattern: str) -> bool:
    return value == pattern or fnmatchcase(value, pattern)
