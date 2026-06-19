from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LmStudioLoad:
    identifier: str
    model_key: str
    status: str | None
    context_length: int | None
    parallel: int | None
    gpu: str | None
    ttl_seconds: int | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def loaded_models(lms_binary: str = "lms") -> list[LmStudioLoad]:
    try:
        completed = subprocess.run(
            [lms_binary, "ps", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    try:
        payload = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return parse_loaded_models(payload)


def inspect_loaded_model(model: str, lms_binary: str = "lms") -> LmStudioLoad | None:
    for loaded in loaded_models(lms_binary):
        candidates = {loaded.identifier, loaded.model_key}
        raw = loaded.raw
        for key in ("model", "modelKey", "selectedVariant", "indexedModelIdentifier"):
            if raw.get(key) is not None:
                candidates.add(str(raw[key]))
        if model in candidates:
            return loaded
    return None


def parse_loaded_models(payload: Any) -> list[LmStudioLoad]:
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = (
            payload.get("models")
            or payload.get("loadedModels")
            or payload.get("data")
            or []
        )
    else:
        raw_items = []

    loads: list[LmStudioLoad] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        identifier = string_from_keys(
            item,
            "identifier",
            "id",
            "modelIdentifier",
            "modelKey",
            "model",
        )
        model_key = string_from_keys(
            item,
            "modelKey",
            "model",
            "selectedVariant",
            "indexedModelIdentifier",
            "identifier",
        )
        if not identifier and not model_key:
            continue
        loads.append(
            LmStudioLoad(
                identifier=identifier or model_key,
                model_key=model_key or identifier,
                status=optional_string_from_keys(item, "status", "state"),
                context_length=int_from_keys(
                    item,
                    "contextLength",
                    "context_length",
                    "maxContextLength",
                    "context",
                ),
                parallel=int_from_keys(
                    item,
                    "parallel",
                    "parallelism",
                    "maxPredictions",
                    "predictions",
                ),
                gpu=optional_string_from_keys(
                    item,
                    "gpu",
                    "gpuOffload",
                    "device",
                    "devices",
                ),
                ttl_seconds=int_from_keys(item, "ttlSeconds", "ttl", "ttl_seconds"),
                raw=item,
            )
        )
    return loads


def estimate_vram_mb(metadata: dict[str, Any], fallback_mb: int) -> int:
    raw_size = metadata.get("sizeBytes")
    if raw_size in (None, ""):
        return fallback_mb

    size_mb = int(math.ceil(float(raw_size) / (1024 * 1024)))
    context_length = int(metadata.get("maxContextLength") or 0)
    context_overhead_mb = min(4096, max(512, math.ceil(context_length / 4096) * 64))
    return int(math.ceil(size_mb * 1.15)) + context_overhead_mb


def parse_estimated_gpu_memory_mb(output: str) -> int | None:
    match = re.search(
        r"Estimated GPU Memory:\s*([\d.,]+)\s*([GM]i?B)",
        output.replace("\u00a0", " "),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower()
    if unit in {"gib", "gb"}:
        return int(math.ceil(value * 1024))
    return int(math.ceil(value))


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


def string_from_keys(payload: dict[str, Any], *keys: str) -> str:
    value = optional_string_from_keys(payload, *keys)
    return value or ""


def optional_string_from_keys(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, list):
            return ",".join(str(item) for item in value)
        return str(value)
    return None


def int_from_keys(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        parsed = optional_int(value)
        if parsed is not None:
            return parsed
    return None


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).replace("\u00a0", "").replace(" ", "")
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))
