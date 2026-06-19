from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = "llmo.task.v1"
ACTIVE_STATES = {"submitted", "queued", "allocating", "starting", "warming", "running"}
PRIORITIES = {"interactive", "foreground", "batch", "maintenance"}


class TaskProtocolError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class QueueTask:
    tenant: str
    project: str
    service: str
    task: str
    job_id: str
    idempotency_key: str
    priority: str
    model: str
    endpoint: str
    payload: dict[str, Any]
    orchestration: dict[str, Any]
    artifacts: dict[str, Any]
    labels: dict[str, Any]


@dataclass
class StoredTask:
    task_id: str
    tenant: str
    project: str
    service: str
    task: str
    job_id: str
    idempotency_key: str
    priority: str
    model: str
    endpoint: str
    state: str = "queued"
    payload: dict[str, Any] = field(default_factory=dict)
    orchestration: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)

    def to_summary(self, *, reused: bool = False) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tenant": self.tenant,
            "project": self.project,
            "service": self.service,
            "task": self.task,
            "job_id": self.job_id,
            "idempotency_key": self.idempotency_key,
            "priority": self.priority,
            "model": self.model,
            "endpoint": self.endpoint,
            "state": self.state,
            "reused": reused,
        }


class InMemoryTaskStore:
    def __init__(self) -> None:
        self._tasks_by_id: dict[str, StoredTask] = {}
        self._idempotency_index: dict[tuple[str, str], str] = {}

    def submit_many(self, tasks: list[QueueTask]) -> tuple[list[StoredTask], list[StoredTask]]:
        accepted: list[StoredTask] = []
        reused: list[StoredTask] = []
        for task in tasks:
            existing_id = self._idempotency_index.get((task.tenant, task.idempotency_key))
            if existing_id is not None:
                reused.append(self._tasks_by_id[existing_id])
                continue

            task_id = f"task_{uuid4().hex}"
            stored = StoredTask(
                task_id=task_id,
                tenant=task.tenant,
                project=task.project,
                service=task.service,
                task=task.task,
                job_id=task.job_id,
                idempotency_key=task.idempotency_key,
                priority=task.priority,
                model=task.model,
                endpoint=task.endpoint,
                payload=task.payload,
                orchestration=task.orchestration,
                artifacts=task.artifacts,
                labels=task.labels,
            )
            self._tasks_by_id[task_id] = stored
            self._idempotency_index[(task.tenant, task.idempotency_key)] = task_id
            accepted.append(stored)
        return accepted, reused

    def queue_lengths_by_model(self) -> dict[str, int]:
        queue_lengths: dict[str, int] = {}
        for task in self._tasks_by_id.values():
            if task.state in ACTIVE_STATES:
                queue_lengths[task.model] = queue_lengths.get(task.model, 0) + 1
        return queue_lengths


def parse_task_queue_payload(payload: Any) -> list[QueueTask]:
    if not isinstance(payload, dict):
        raise TaskProtocolError("Task queue request body must be a JSON object.")

    common_orchestration = require_mapping(payload.get("orchestration"), "orchestration")
    validate_common_orchestration(common_orchestration)

    common_model = required_string(payload, "model")
    common_endpoint = string_value(payload.get("endpoint") or "/v1/chat/completions", "endpoint")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise TaskProtocolError("Task queue request requires a non-empty tasks array.")

    tasks: list[QueueTask] = []
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict):
            raise TaskProtocolError(f"tasks[{index}] must be a JSON object.")
        tasks.append(parse_queue_task(raw_task, common_orchestration, common_model, common_endpoint, index))
    return tasks


def parse_queue_task(
    raw_task: dict[str, Any],
    common_orchestration: dict[str, Any],
    common_model: str,
    common_endpoint: str,
    index: int,
) -> QueueTask:
    orchestration = dict(common_orchestration)
    for key in ("job_id", "idempotency_key", "labels", "artifacts"):
        if key in raw_task:
            orchestration[key] = raw_task[key]

    job_id = string_value(orchestration.get("job_id"), f"tasks[{index}].job_id")
    idempotency_key = string_value(
        orchestration.get("idempotency_key"),
        f"tasks[{index}].idempotency_key",
    )
    artifacts = optional_mapping(orchestration.get("artifacts"), f"tasks[{index}].artifacts")
    labels = optional_mapping(orchestration.get("labels"), f"tasks[{index}].labels")
    payload = optional_mapping(raw_task.get("payload"), f"tasks[{index}].payload")

    return QueueTask(
        tenant=string_value(common_orchestration.get("tenant"), "orchestration.tenant"),
        project=string_value(common_orchestration.get("project"), "orchestration.project"),
        service=string_value(common_orchestration.get("service"), "orchestration.service"),
        task=string_value(common_orchestration.get("task"), "orchestration.task"),
        job_id=job_id,
        idempotency_key=idempotency_key,
        priority=string_value(common_orchestration.get("priority"), "orchestration.priority"),
        model=string_value(raw_task.get("model") or common_model, f"tasks[{index}].model"),
        endpoint=string_value(raw_task.get("endpoint") or common_endpoint, f"tasks[{index}].endpoint"),
        payload=payload,
        orchestration=orchestration,
        artifacts=artifacts,
        labels=labels,
    )


def validate_common_orchestration(orchestration: dict[str, Any]) -> None:
    schema_version = string_value(
        orchestration.get("schema_version"),
        "orchestration.schema_version",
    )
    if schema_version != SCHEMA_VERSION:
        raise TaskProtocolError(
            f"orchestration.schema_version must be {SCHEMA_VERSION!r}."
        )

    for field_name in ("tenant", "project", "service", "task", "priority"):
        string_value(orchestration.get(field_name), f"orchestration.{field_name}")

    priority = str(orchestration["priority"])
    if priority not in PRIORITIES:
        raise TaskProtocolError(
            "orchestration.priority must be one of: "
            + ", ".join(sorted(PRIORITIES))
            + "."
        )


def require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskProtocolError(f"{field_name} must be a JSON object.")
    return value


def optional_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TaskProtocolError(f"{field_name} must be a JSON object.")
    return value


def required_string(payload: dict[str, Any], field_name: str) -> str:
    return string_value(payload.get(field_name), field_name)


def string_value(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskProtocolError(f"{field_name} must be a non-empty string.")
    return value.strip()
