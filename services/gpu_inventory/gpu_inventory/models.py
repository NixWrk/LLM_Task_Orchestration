from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GpuState:
    id: str
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_gpu_percent: int | None = None
    temperature_gpu_celsius: int | None = None
    source: str = "nvidia-smi"

    @property
    def memory_free_mb(self) -> int:
        return max(0, self.memory_total_mb - self.memory_used_mb)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["memory_free_mb"] = self.memory_free_mb
        return payload


@dataclass(frozen=True)
class InventorySnapshot:
    gpus: list[GpuState]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "gpu_count": len(self.gpus),
            "gpus": [gpu.to_dict() for gpu in self.gpus],
        }
