from __future__ import annotations

import json
import math
import subprocess
from typing import Any


def metadata_for_model(model: str, lms_binary: str = "lms") -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [lms_binary, "ls", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}

    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, list):
        return {}

    for item in payload:
        if not isinstance(item, dict):
            continue
        candidates = {
            str(item.get("modelKey") or ""),
            str(item.get("selectedVariant") or ""),
            str(item.get("indexedModelIdentifier") or ""),
        }
        variants = item.get("variants")
        if isinstance(variants, list):
            candidates.update(str(variant) for variant in variants)
        if model in candidates:
            return item
    return {}


def estimate_vram_mb(metadata: dict[str, Any], fallback_mb: int) -> int:
    raw_size = metadata.get("sizeBytes")
    if raw_size in (None, ""):
        return fallback_mb

    size_mb = int(math.ceil(float(raw_size) / (1024 * 1024)))
    context_length = int(metadata.get("maxContextLength") or 0)
    context_overhead_mb = min(4096, max(512, math.ceil(context_length / 4096) * 64))
    return int(math.ceil(size_mb * 1.15)) + context_overhead_mb


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "type",
        "modelKey",
        "displayName",
        "path",
        "sizeBytes",
        "paramsString",
        "architecture",
        "quantization",
        "maxContextLength",
        "vision",
        "trainedForToolUse",
        "selectedVariant",
    )
    return {key: metadata[key] for key in keys if key in metadata}
