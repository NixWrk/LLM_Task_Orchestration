from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from typing import Protocol
from uuid import uuid4

SCHEMA_VERSION = "llmo.task.v1"
ACTIVE_STATES = {"submitted", "queued", "allocating", "starting", "warming", "running"}
CLAIMABLE_STATES = {"queued"}
PRIORITIES = {"interactive", "foreground", "batch", "maintenance"}
FAIRNESS_GROUP_FIELDS = ("tenant", "project", "service", "task", "priority", "model")
CONTEXT_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
DEFAULT_MAX_OUTPUT_TOKENS = 1024
TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4
TEMPLATE_PATTERN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*}}")


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
    created_at: str = field(default_factory=lambda: now_iso())
    updated_at: str = field(default_factory=lambda: now_iso())
    attempt_count: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    next_attempt_at: str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

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
            "attempt_count": self.attempt_count,
            "next_attempt_at": self.next_attempt_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reused": reused,
        }

    def to_detail(self) -> dict[str, Any]:
        detail = self.to_summary()
        detail.update(
            {
                "payload": self.payload,
                "orchestration": self.orchestration,
                "artifacts": self.artifacts,
                "labels": self.labels,
                "attempt_count": self.attempt_count,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "next_attempt_at": self.next_attempt_at,
                "result": self.result,
                "error": self.error,
            }
        )
        return detail

    def to_record(self) -> dict[str, Any]:
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
            "payload": self.payload,
            "orchestration": self.orchestration,
            "artifacts": self.artifacts,
            "labels": self.labels,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "attempt_count": self.attempt_count,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "next_attempt_at": self.next_attempt_at,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> StoredTask:
        return cls(
            task_id=string_value(record.get("task_id"), "task_id"),
            tenant=string_value(record.get("tenant"), "tenant"),
            project=string_value(record.get("project"), "project"),
            service=string_value(record.get("service"), "service"),
            task=string_value(record.get("task"), "task"),
            job_id=string_value(record.get("job_id"), "job_id"),
            idempotency_key=string_value(
                record.get("idempotency_key"),
                "idempotency_key",
            ),
            priority=string_value(record.get("priority"), "priority"),
            model=string_value(record.get("model"), "model"),
            endpoint=string_value(record.get("endpoint"), "endpoint"),
            estimated_input_tokens=optional_non_negative_int(
                record.get("estimated_input_tokens"),
                "estimated_input_tokens",
            )
            or 0,
            max_output_tokens=optional_positive_int(
                record.get("max_output_tokens"),
                "max_output_tokens",
            )
            or DEFAULT_MAX_OUTPUT_TOKENS,
            required_context_tokens=optional_positive_int(
                record.get("required_context_tokens"),
                "required_context_tokens",
            )
            or 1,
            state=string_value(record.get("state") or "queued", "state"),
            payload=optional_mapping(record.get("payload"), "payload"),
            orchestration=optional_mapping(record.get("orchestration"), "orchestration"),
            artifacts=optional_mapping(record.get("artifacts"), "artifacts"),
            labels=optional_mapping(record.get("labels"), "labels"),
            created_at=optional_string(record.get("created_at")) or now_iso(),
            updated_at=optional_string(record.get("updated_at")) or now_iso(),
            attempt_count=optional_non_negative_int(
                record.get("attempt_count"),
                "attempt_count",
            )
            or 0,
            started_at=optional_string(record.get("started_at")),
            finished_at=optional_string(record.get("finished_at")),
            next_attempt_at=optional_string(record.get("next_attempt_at")),
            result=optional_mapping_or_none(record.get("result"), "result"),
            error=optional_mapping_or_none(record.get("error"), "error"),
        )


class TaskStore(Protocol):
    def submit_many(self, tasks: list[QueueTask]) -> tuple[list[StoredTask], list[StoredTask]]:
        ...

    def queue_lengths_by_model(self) -> dict[str, int]:
        ...

    def context_plans_by_model(self) -> dict[str, dict[str, Any]]:
        ...

    def task_counts_by_state(self) -> dict[tuple[str, str, str, str, str, str], int]:
        ...

    def get_task(self, tenant: str, task_id: str) -> StoredTask | None:
        ...

    def list_tasks(
        self,
        tenant: str,
        *,
        state: str | None = None,
        model: str | None = None,
        limit: int = 100,
    ) -> list[StoredTask]:
        ...

    def claim_next(self, *, model: str | None = None) -> StoredTask | None:
        ...

    def record_result(self, task_id: str, result: dict[str, Any]) -> StoredTask:
        ...

    def record_error(self, task_id: str, error: dict[str, Any]) -> StoredTask:
        ...

    def record_retry(
        self,
        task_id: str,
        error: dict[str, Any],
        next_attempt_at: str,
    ) -> StoredTask:
        ...

    def cancel_task(self, tenant: str, task_id: str) -> StoredTask | None:
        ...


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
        self._after_mutation()
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

    def task_counts_by_state(self) -> dict[tuple[str, str, str, str, str, str], int]:
        counts: dict[tuple[str, str, str, str, str, str], int] = {}
        for task in self._tasks_by_id.values():
            key = task_state_count_key(task)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_task(self, tenant: str, task_id: str) -> StoredTask | None:
        task = self._tasks_by_id.get(task_id)
        if task is None or task.tenant != tenant:
            return None
        return task

    def list_tasks(
        self,
        tenant: str,
        *,
        state: str | None = None,
        model: str | None = None,
        limit: int = 100,
    ) -> list[StoredTask]:
        tasks = [
            task
            for task in self._tasks_by_id.values()
            if task.tenant == tenant
            and (state is None or task.state == state)
            and (model is None or task.model == model)
        ]
        tasks.sort(key=lambda task: (task.created_at, task.task_id))
        return tasks[: max(1, limit)]

    def claim_next(self, *, model: str | None = None) -> StoredTask | None:
        now = utc_now()
        candidates = [
            task
            for task in self._tasks_by_id.values()
            if is_task_claimable(task, now, model)
        ]
        if not candidates:
            return None
        task = choose_fair_task(candidates, list(self._tasks_by_id.values()))
        now_text = now_iso(now)
        task.state = "running"
        task.attempt_count += 1
        task.started_at = task.started_at or now_text
        task.updated_at = now_text
        task.next_attempt_at = None
        self._after_mutation()
        return task

    def record_result(self, task_id: str, result: dict[str, Any]) -> StoredTask:
        task = self._tasks_by_id[task_id]
        task.state = "succeeded"
        task.result = result
        task.error = None
        task.next_attempt_at = None
        task.finished_at = now_iso()
        task.updated_at = task.finished_at
        self._after_mutation()
        return task

    def record_error(self, task_id: str, error: dict[str, Any]) -> StoredTask:
        task = self._tasks_by_id[task_id]
        task.state = "failed"
        task.error = error
        task.next_attempt_at = None
        task.finished_at = now_iso()
        task.updated_at = task.finished_at
        self._after_mutation()
        return task

    def record_retry(
        self,
        task_id: str,
        error: dict[str, Any],
        next_attempt_at: str,
    ) -> StoredTask:
        task = self._tasks_by_id[task_id]
        task.state = "queued"
        task.error = error
        task.next_attempt_at = next_attempt_at
        task.updated_at = now_iso()
        self._after_mutation()
        return task

    def cancel_task(self, tenant: str, task_id: str) -> StoredTask | None:
        task = self.get_task(tenant, task_id)
        if task is None:
            return None
        if task.state in {"succeeded", "failed", "cancelled"}:
            return task
        task.state = "cancelled"
        task.next_attempt_at = None
        task.finished_at = now_iso()
        task.updated_at = task.finished_at
        self._after_mutation()
        return task

    def _after_mutation(self) -> None:
        return None

    def _load_tasks(self, tasks: list[StoredTask]) -> None:
        self._tasks_by_id = {}
        self._idempotency_index = {}
        for task in tasks:
            self._tasks_by_id[task.task_id] = task
            self._idempotency_index[(task.tenant, task.idempotency_key)] = task.task_id


class JsonFileTaskStore(InMemoryTaskStore):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__()
        self._load_from_disk()

    def _after_mutation(self) -> None:
        self._save_to_disk()

    def _load_from_disk(self) -> None:
        if not self.path.exists():
            return

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TaskProtocolError("Task store file must contain a JSON object.")
        raw_tasks = raw.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raise TaskProtocolError("Task store file field 'tasks' must be an array.")
        tasks = []
        for index, raw_task in enumerate(raw_tasks):
            if not isinstance(raw_task, dict):
                raise TaskProtocolError(f"Task store tasks[{index}] must be an object.")
            tasks.append(StoredTask.from_record(raw_task))
        self._load_tasks(tasks)

    def _save_to_disk(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "tasks": [
                task.to_record()
                for task in sorted(self._tasks_by_id.values(), key=lambda item: item.task_id)
            ],
        }
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)


def build_task_store(
    path: str | Path | None = None,
    *,
    backend: str | None = None,
    dsn: str | None = None,
) -> TaskStore:
    selected_backend = (backend or "").strip().lower()
    if selected_backend in {"postgres", "postgresql"} or dsn:
        from queue_proxy.postgres_task_store import PostgresTaskStore

        return PostgresTaskStore(str(dsn or path or ""))
    if path:
        return JsonFileTaskStore(path)
    return InMemoryTaskStore()


def parse_task_queue_payload(payload: Any) -> list[QueueTask]:
    if not isinstance(payload, dict):
        raise TaskProtocolError("Task queue request body must be a JSON object.")

    common_orchestration = require_mapping(payload.get("orchestration"), "orchestration")
    validate_common_orchestration(common_orchestration)

    common_model = required_string(payload, "model")
    common_endpoint = string_value(payload.get("endpoint") or "/v1/chat/completions", "endpoint")
    common_payload_template = optional_mapping_or_none(
        payload.get("payload_template"),
        "payload_template",
    )
    common_template_vars = optional_mapping(payload.get("template_vars"), "template_vars")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise TaskProtocolError("Task queue request requires a non-empty tasks array.")

    tasks: list[QueueTask] = []
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict):
            raise TaskProtocolError(f"tasks[{index}] must be a JSON object.")
        tasks.append(
            parse_queue_task(
                raw_task,
                common_orchestration,
                common_model,
                common_endpoint,
                common_payload_template,
                common_template_vars,
                index,
            )
        )
    return tasks


def parse_queue_task(
    raw_task: dict[str, Any],
    common_orchestration: dict[str, Any],
    common_model: str,
    common_endpoint: str,
    common_payload_template: dict[str, Any] | None,
    common_template_vars: dict[str, Any],
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
    model = string_value(raw_task.get("model") or common_model, f"tasks[{index}].model")
    endpoint = string_value(raw_task.get("endpoint") or common_endpoint, f"tasks[{index}].endpoint")
    payload = task_payload_from_raw(
        raw_task,
        common_payload_template,
        {
            "model": model,
            "endpoint": endpoint,
            "job_id": job_id,
            "idempotency_key": idempotency_key,
            "tenant": common_orchestration.get("tenant"),
            "project": common_orchestration.get("project"),
            "service": common_orchestration.get("service"),
            "task": common_orchestration.get("task"),
            "priority": common_orchestration.get("priority"),
            "artifacts": artifacts,
            "labels": labels,
        },
        common_template_vars,
        index,
    )
    tokens = task_tokens(raw_task, orchestration, payload, index)

    return QueueTask(
        tenant=string_value(common_orchestration.get("tenant"), "orchestration.tenant"),
        project=string_value(common_orchestration.get("project"), "orchestration.project"),
        service=string_value(common_orchestration.get("service"), "orchestration.service"),
        task=string_value(common_orchestration.get("task"), "orchestration.task"),
        job_id=job_id,
        idempotency_key=idempotency_key,
        priority=string_value(common_orchestration.get("priority"), "orchestration.priority"),
        model=model,
        endpoint=endpoint,
        estimated_input_tokens=tokens["estimated_input_tokens"],
        max_output_tokens=tokens["max_output_tokens"],
        required_context_tokens=tokens["required_context_tokens"],
        payload=payload,
        orchestration=orchestration,
        artifacts=artifacts,
        labels=labels,
    )


def task_payload_from_raw(
    raw_task: dict[str, Any],
    common_payload_template: dict[str, Any] | None,
    base_vars: dict[str, Any],
    common_template_vars: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    if "payload" in raw_task:
        return optional_mapping(raw_task.get("payload"), f"tasks[{index}].payload")

    raw_template = raw_task.get("payload_template")
    template = (
        optional_mapping(raw_template, f"tasks[{index}].payload_template")
        if raw_template is not None
        else common_payload_template
    )
    if template is None:
        return {}

    raw_vars = optional_mapping(raw_task.get("template_vars"), f"tasks[{index}].template_vars")
    rendered = render_payload_template(
        template,
        {
            **base_vars,
            **common_template_vars,
            **raw_vars,
        },
        f"tasks[{index}].payload_template",
    )
    if not isinstance(rendered, dict):
        raise TaskProtocolError(f"tasks[{index}].payload_template must render to a JSON object.")
    return rendered


def render_payload_template(value: Any, variables: dict[str, Any], field_name: str) -> Any:
    if isinstance(value, str):
        return render_template_string(value, variables, field_name)
    if isinstance(value, list):
        return [
            render_payload_template(item, variables, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: render_payload_template(item, variables, f"{field_name}.{key}")
            for key, item in value.items()
        }
    return value


def render_template_string(value: str, variables: dict[str, Any], field_name: str) -> Any:
    full_match = TEMPLATE_PATTERN.fullmatch(value)
    if full_match is not None:
        return template_variable(variables, full_match.group(1), field_name)

    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        variable_value = template_variable(variables, variable_name, field_name)
        if isinstance(variable_value, (dict, list)):
            return json.dumps(variable_value, ensure_ascii=False, sort_keys=True)
        return str(variable_value)

    return TEMPLATE_PATTERN.sub(replace, value)


def template_variable(variables: dict[str, Any], name: str, field_name: str) -> Any:
    current: Any = variables
    for part in name.split("."):
        if not isinstance(current, dict) or part not in current:
            raise TaskProtocolError(
                f"{field_name} references unknown template variable {name!r}."
            )
        current = current[part]
    return current


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


def optional_mapping_or_none(value: Any, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TaskProtocolError(f"{field_name} must be a JSON object.")
    return value


def required_string(payload: dict[str, Any], field_name: str) -> str:
    return string_value(payload.get(field_name), field_name)


def string_value(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskProtocolError(f"{field_name} must be a non-empty string.")
    return value.strip()


def optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


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


def optional_non_negative_int(value: Any, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TaskProtocolError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise TaskProtocolError(f"{field_name} must be a non-negative integer.")
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


def is_task_claimable(
    task: StoredTask,
    now: datetime,
    model: str | None = None,
) -> bool:
    if task.state not in CLAIMABLE_STATES:
        return False
    if model is not None and task.model != model:
        return False
    if task.next_attempt_at is None:
        return True
    return parse_iso_datetime(task.next_attempt_at) <= now


def choose_fair_task(candidates: list[StoredTask], all_tasks: list[StoredTask]) -> StoredTask:
    group_candidates: dict[tuple[str, ...], list[StoredTask]] = {}
    for task in candidates:
        group_candidates.setdefault(fairness_group_key(task), []).append(task)

    last_claimed_by_group: dict[tuple[str, ...], str] = {}
    for task in all_tasks:
        if task.attempt_count < 1:
            continue
        group_key = fairness_group_key(task)
        current = last_claimed_by_group.get(group_key)
        if current is None or task.updated_at > current:
            last_claimed_by_group[group_key] = task.updated_at

    def group_sort_key(group_key: tuple[str, ...]) -> tuple[bool, str, str, tuple[str, ...]]:
        group_tasks = group_candidates[group_key]
        oldest_created_at = min(task.created_at for task in group_tasks)
        last_claimed_at = last_claimed_by_group.get(group_key)
        return (
            last_claimed_at is not None,
            last_claimed_at or "",
            oldest_created_at,
            group_key,
        )

    selected_group = min(group_candidates, key=group_sort_key)
    return min(
        group_candidates[selected_group],
        key=lambda task: (task.created_at, task.task_id),
    )


def fairness_group_key(task: StoredTask) -> tuple[str, ...]:
    return tuple(str(getattr(task, field_name)) for field_name in FAIRNESS_GROUP_FIELDS)


def task_state_count_key(task: StoredTask) -> tuple[str, str, str, str, str, str]:
    return (
        task.tenant,
        task.project,
        task.service,
        task.task,
        task.model,
        task.state,
    )


def context_bucket(required_context_tokens: int) -> int:
    required = max(1, required_context_tokens)
    for bucket in CONTEXT_BUCKETS:
        if required <= bucket:
            return bucket
    return required


def utc_now() -> datetime:
    return datetime.now(UTC)


def now_iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


def parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
