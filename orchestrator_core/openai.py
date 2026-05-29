from __future__ import annotations


def openai_url(base_url: str, path: str, *, ensure_v1: bool = False) -> str:
    base = base_url.rstrip("/")
    normalized_path = path.lstrip("/")

    if base.endswith("/v1") and normalized_path.startswith("v1/"):
        normalized_path = normalized_path[3:]
    elif ensure_v1 and not base.endswith("/v1"):
        normalized_path = f"v1/{normalized_path.removeprefix('v1/')}"

    return f"{base}/{normalized_path}"
