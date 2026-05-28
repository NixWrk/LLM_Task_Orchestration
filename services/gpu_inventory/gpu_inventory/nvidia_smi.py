from __future__ import annotations

import csv
import json
import subprocess
from io import StringIO
from typing import Any

from gpu_inventory.models import GpuState, InventorySnapshot

QUERY_FIELDS = [
    "index",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
]


class GpuInventoryError(Exception):
    pass


def collect_inventory(
    nvidia_smi_path: str,
    timeout_seconds: float,
    fake_inventory_json: str = "",
) -> InventorySnapshot:
    if fake_inventory_json:
        return parse_fake_inventory(fake_inventory_json)

    command = [
        nvidia_smi_path,
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise GpuInventoryError(str(exc)) from exc

    return InventorySnapshot(
        gpus=parse_nvidia_smi_csv(result.stdout),
        source="nvidia-smi",
    )


def parse_nvidia_smi_csv(output: str) -> list[GpuState]:
    rows = csv.reader(StringIO(output.strip()))
    gpus: list[GpuState] = []
    for row in rows:
        if not row:
            continue
        cleaned = [value.strip() for value in row]
        if len(cleaned) < 4:
            raise GpuInventoryError(f"Unexpected nvidia-smi row: {row}")

        index = int(cleaned[0])
        gpus.append(
            GpuState(
                id=f"gpu{index}",
                index=index,
                name=cleaned[1],
                memory_total_mb=int(cleaned[2]),
                memory_used_mb=int(cleaned[3]),
                utilization_gpu_percent=parse_optional_int(cleaned, 4),
                temperature_gpu_celsius=parse_optional_int(cleaned, 5),
                source="nvidia-smi",
            )
        )
    return gpus


def parse_fake_inventory(raw_json: str) -> InventorySnapshot:
    payload = json.loads(raw_json)
    raw_gpus = payload.get("gpus", payload if isinstance(payload, list) else [])
    if not isinstance(raw_gpus, list):
        raise GpuInventoryError("Fake GPU inventory must be a list or object with 'gpus'.")

    gpus: list[GpuState] = []
    for fallback_index, item in enumerate(raw_gpus):
        if not isinstance(item, dict):
            raise GpuInventoryError("Fake GPU entries must be objects.")
        index = int(item.get("index", fallback_index))
        memory_total_mb = int(
            item.get("memory_total_mb", item.get("memory_total", item.get("total_mb", 0)))
        )
        memory_used_mb = int(
            item.get("memory_used_mb", item.get("memory_used", item.get("used_mb", 0)))
        )
        gpus.append(
            GpuState(
                id=str(item.get("id") or f"gpu{index}"),
                index=index,
                name=str(item.get("name") or "fake-gpu"),
                memory_total_mb=memory_total_mb,
                memory_used_mb=memory_used_mb,
                utilization_gpu_percent=optional_int(item.get("utilization_gpu_percent")),
                temperature_gpu_celsius=optional_int(item.get("temperature_gpu_celsius")),
                source="fake",
            )
        )

    return InventorySnapshot(gpus=gpus, source="fake")


def parse_optional_int(values: list[str], index: int) -> int | None:
    if len(values) <= index:
        return None
    return optional_int(values[index])


def optional_int(value: Any) -> int | None:
    if value in (None, "", "N/A", "[N/A]"):
        return None
    return int(value)
