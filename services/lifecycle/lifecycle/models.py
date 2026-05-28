from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

BackendState = Literal[
    "starting",
    "warming",
    "ready",
    "draining",
    "stopping",
    "stopped",
    "failed",
]

ScaleActionType = Literal["start", "stop", "noop"]


@dataclass(frozen=True)
class GpuState:
    id: str
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int


@dataclass(frozen=True)
class ModelProfile:
    public_name: str
    backend_model: str
    runtime: str
    artifact: str | None
    runtime_image: str | None
    host_port_start: int
    container_port: int
    public_host: str
    docker_extra_args: tuple[str, ...]
    runtime_extra_args: tuple[str, ...]
    volume_mounts: tuple[VolumeMount, ...]
    environment: tuple[EnvironmentVariable, ...]
    healthcheck_path: str
    startup_timeout_seconds: float
    healthcheck_interval_seconds: float
    warmup_enabled: bool
    warmup_prompt: str
    warmup_max_tokens: int
    estimated_vram_mb: int
    safety_margin_mb: int
    min_replicas: int
    max_replicas: int
    idle_ttl_seconds: int
    preferred_gpus: tuple[str, ...] = ("auto",)


@dataclass
class BackendInstance:
    instance_id: str
    model: str
    backend_model: str
    runtime: str
    base_url: str
    gpu_ids: list[str]
    state: BackendState
    reserved_vram_mb: int
    host_port: int | None = None
    container_name: str | None = None
    runtime_command: list[str] = field(default_factory=list)
    active_requests: int = 0
    failure_reason: str | None = None
    created_at: str = field(default_factory=lambda: now_iso())
    updated_at: str = field(default_factory=lambda: now_iso())
    last_used_at: str | None = None
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BackendInstance:
        return cls(
            instance_id=str(payload["instance_id"]),
            model=str(payload["model"]),
            backend_model=str(payload.get("backend_model") or payload["model"]),
            runtime=str(payload.get("runtime") or "external"),
            base_url=str(payload.get("base_url") or ""),
            gpu_ids=[str(gpu_id) for gpu_id in payload.get("gpu_ids", [])],
            state=payload.get("state", "ready"),
            reserved_vram_mb=int(payload.get("reserved_vram_mb", 0)),
            host_port=optional_int(payload.get("host_port")),
            container_name=payload.get("container_name"),
            runtime_command=[str(part) for part in payload.get("runtime_command", [])],
            active_requests=int(payload.get("active_requests", 0)),
            failure_reason=payload.get("failure_reason"),
            created_at=str(payload.get("created_at") or now_iso()),
            updated_at=str(payload.get("updated_at") or now_iso()),
            last_used_at=payload.get("last_used_at"),
            dry_run=bool(payload.get("dry_run", True)),
        )


@dataclass(frozen=True)
class PlacementDecision:
    model: str
    action: ScaleActionType
    gpu_id: str | None
    reason: str
    required_vram_mb: int
    available_vram_mb: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VolumeMount:
    host_path: str
    container_path: str
    mode: str = "ro"


@dataclass(frozen=True)
class EnvironmentVariable:
    name: str
    value: str


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
