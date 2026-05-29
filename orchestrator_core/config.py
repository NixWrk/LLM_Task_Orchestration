from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    defaults: dict[str, Any] = Field(default_factory=dict)
    dynamic_models: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, dict[str, Any] | None] = Field(default_factory=dict)

    @field_validator("defaults", "dynamic_models", "models", mode="before")
    @classmethod
    def empty_or_invalid_mapping_to_dict(cls, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}


def load_orchestrator_config(path: str) -> OrchestratorConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raw = {}
    return OrchestratorConfig.model_validate(raw)
