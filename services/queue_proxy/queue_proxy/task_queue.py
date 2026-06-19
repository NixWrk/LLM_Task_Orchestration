from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = "llmo.task.v1"
ACTIVE_STATES = {"submitted", "queued", "allocating", "starting", "warming", "running"}
PRIORITIES = {"interactive", "foreground", "batch", "maintenance"}
CONTEXT_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
DEFAULT_MAX_OUTPUT_TOKENS = 1024
TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4


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
    estimated_input_tokens: int
    max_output_tokens: int
    required_context_tokens: int
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
    estimated_input_tokens: int
    max_output_tokens: int
    required_context_tokens: int
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
            "estimated_input_tokens": self.estimated_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "required_context_tokens": self.required_context_tokens,
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
                estimated_input_tokens=task.estimated_input_tokens,
                max_output_tokens=task.max_output_tokens,
                required_context_tokens=task.required_context_tokens,
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

    def context_plans_by_model(self) -> dict[str, dict[str, Any]]:
        tasks_by_model: dict[str, list[StoredTask]] = {}
        for task in self._tasks_by_id.values():
            if task.state in ACTIVE_STATES:
                tasks_by_model.setdefault(task.model, []).append(task)

        return {
            model: context_plan_for_model(tasks)
            for model, tasks in sorted(tasks_by_model.items())
        }


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
    tokens = task_tokens(raw_task, orchestration, payload, index)

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
        estimated_input_tokens=tokens["estimated_input_tokens"],
        max_output_tokens=tokens["max_output_tokens"],
        required_context_tokens=tokens["required_context_tokens"],
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

    for field_name in ("lms_parallel", "max_parallel", "lms_context_length"):
        optional_positive_int(
            orchestration.get(field_name),
            f"orchestration.{field_name}",
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


def task_tokens(
    raw_task: dict[str, Any],
    orchestration: dict[str, Any],
    payload: dict[str, Any],
    index: int,
) -> dict[str, int]:
    common_tokens = optional_mapping(
        orchestration.get("tokens"),
        f"tasks[{index}].orchestration.tokens",
    )
    task_tokens_payload = optional_mapping(raw_task.get("tokens"), f"tasks[{index}].tokens")
    tokens = {**common_tokens, **task_tokens_payload}

    estimated_input_tokens = optional_positive_int(
        tokens.get("estimated_input_tokens"),
        f"tasks[{index}].tokens.estimated_input_tokens",
    )
    if estimated_input_tokens is None:
        estimated_input_tokens = estimate_input_tokens(payload)

    max_output_tokens = optional_positive_int(
        tokens.get("max_output_tokens")
        or payload.get("max_tokens")
        or payload.get("max_output_tokens")
        or payload.get("max_completion_tokens")
        or orchestration.get("max_output_tokens"),
        f"tasks[{index}].tokens.max_output_tokens",
    )
    if max_output_tokens is None:
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS

    return {
        "estimated_input_tokens": estimated_input_tokens,
        "max_output_tokens": max_output_tokens,
        "required_context_tokens": max(1, estimated_input_tokens + max_output_tokens),
    }


def optional_positive_int(value: Any, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TaskProtocolError(f"{field_name} must be a positive integer.") from exc
    if parsed < 1:
        raise TaskProtocolError(f"{field_name} must be a positive integer.")
    return parsed


def estimate_input_tokens(payload: dict[str, Any]) -> int:
    text = collect_input_text(payload)
    if not text:
        return 0
    return max(1, round(len(text) / TOKEN_ESTIMATE_CHARS_PER_TOKEN))


def collect_input_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(collect_input_text(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("messages", "input", "prompt", "content", "text"):
            if key in value:
                parts.append(collect_input_text(value[key]))
        return "\n".join(parts)
    return ""


def context_plan_for_model(tasks: list[StoredTask]) -> dict[str, Any]:
    max_required_context = max((task.required_context_tokens for task in tasks), default=0)
    requested_parallel = max(
        (requested_parallel_for_task(task) for task in tasks),
        default=1,
    )
    desired_parallel = max(1, min(len(tasks), requested_parallel))
    context_cap = min(
        (
            cap
            for cap in (context_cap_for_task(task) for task in tasks)
            if cap is not None
        ),
        default=None,
    )
    recommended_context = context_bucket(max_required_context)
    oversized_tasks = [
        task.job_id
        for task in tasks
        if context_cap is not None and task.required_context_tokens > context_cap
    ]
    if context_cap is not None:
        recommended_context = min(recommended_context, context_cap)

    return {
        "queued_tasks": len(tasks),
        "max_required_context_tokens": max_required_context,
        "recommended_lms_context_length": recommended_context,
        "requested_parallel": requested_parallel,
        "recommended_lms_parallel": desired_parallel,
        "total_slot_context_tokens": recommended_context * desired_parallel,
        "context_cap_tokens": context_cap,
        "oversized_tasks": oversized_tasks,
        "reload_required": False,
        "task_contexts": [
            {
                "task_id": task.task_id,
                "job_id": task.job_id,
                "required_context_tokens": task.required_context_tokens,
                "estimated_input_tokens": task.estimated_input_tokens,
                "max_output_tokens": task.max_output_tokens,
            }
            for task in tasks
        ],
    }


def requested_parallel_for_task(task: StoredTask) -> int:
    return (
        optional_positive_int(task.orchestration.get("lms_parallel"), "lms_parallel")
        or optional_positive_int(task.orchestration.get("max_parallel"), "max_parallel")
        or 1
    )


def context_cap_for_task(task: StoredTask) -> int | None:
    return optional_positive_int(task.orchestration.get("lms_context_length"), "lms_context_length")


def context_bucket(required_context_tokens: int) -> int:
    required = max(1, required_context_tokens)
    for bucket in CONTEXT_BUCKETS:
        if required <= bucket:
            return bucket
    return required
