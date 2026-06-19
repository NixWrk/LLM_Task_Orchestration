from __future__ import annotations

from typing import Any

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except ModuleNotFoundError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    _FALLBACK_METRICS: list[_FallbackMetric] = []

    class _FallbackMetric:
        metric_type = "gauge"

        def __init__(
            self,
            name: str,
            _description: str,
            label_names: list[str] | tuple[str, ...] | None = None,
        ) -> None:
            self.name = name
            self.label_names = tuple(label_names or ())
            self.values: dict[tuple[str, ...], float] = {}
            self.counts: dict[tuple[str, ...], int] = {}
            _FALLBACK_METRICS.append(self)

        def labels(self, *args: Any, **kwargs: Any) -> _FallbackMetricChild:
            if args:
                key = tuple(str(value) for value in args)
            else:
                key = tuple(str(kwargs.get(name, "")) for name in self.label_names)
            return _FallbackMetricChild(self, key)

        def inc(self, amount: float = 1.0) -> None:
            self.labels().inc(amount)

        def set(self, value: float) -> None:
            self.labels().set(value)

        def observe(self, value: float) -> None:
            self.labels().observe(value)

    class _FallbackMetricChild:
        def __init__(self, metric: _FallbackMetric, key: tuple[str, ...]) -> None:
            self.metric = metric
            self.key = key

        def inc(self, amount: float = 1.0) -> None:
            self.metric.values[self.key] = self.metric.values.get(self.key, 0.0) + amount

        def set(self, value: float) -> None:
            self.metric.values[self.key] = float(value)

        def observe(self, value: float) -> None:
            self.metric.values[self.key] = self.metric.values.get(self.key, 0.0) + float(value)
            self.metric.counts[self.key] = self.metric.counts.get(self.key, 0) + 1

    class Counter(_FallbackMetric):
        metric_type = "counter"

    class Gauge(_FallbackMetric):
        metric_type = "gauge"

    class Histogram(_FallbackMetric):
        metric_type = "histogram"

    def generate_latest() -> bytes:
        lines: list[str] = []
        for metric in _FALLBACK_METRICS:
            for key, value in sorted(metric.values.items()):
                labels = fallback_labels(metric.label_names, key)
                if metric.metric_type == "histogram":
                    count = metric.counts.get(key, 0)
                    lines.append(f"{metric.name}_count{labels} {float(count)}")
                    lines.append(f"{metric.name}_sum{labels} {value}")
                else:
                    lines.append(f"{metric.name}{labels} {value}")
        return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""

    def fallback_labels(label_names: tuple[str, ...], key: tuple[str, ...]) -> str:
        if not label_names:
            return ""
        pairs = [
            f'{name}="{escape_label(value)}"'
            for name, value in zip(label_names, key, strict=False)
        ]
        return "{" + ",".join(pairs) + "}"

    def escape_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


REQUESTS = Counter(
    "llm_requests_total",
    "Requests forwarded by the queue proxy.",
    ["model", "endpoint", "status"],
)

ERRORS = Counter(
    "llm_request_errors_total",
    "Requests rejected or failed in the queue proxy.",
    ["model", "error_type"],
)

LATENCY = Histogram(
    "llm_request_latency_seconds",
    "End-to-end request latency observed by the queue proxy.",
    ["model", "endpoint"],
)

QUEUE_LENGTH = Gauge(
    "llm_queue_length",
    "Queued requests per model.",
    ["model"],
)

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Active requests per model.",
    ["model"],
)

INPUT_TOKENS = Counter(
    "llm_input_tokens_total",
    "Estimated input tokens admitted by the queue proxy.",
    ["model"],
)

OUTPUT_TOKEN_BUDGET = Counter(
    "llm_output_token_budget_total",
    "Output token budget admitted by the queue proxy.",
    ["model"],
)

TASK_EVENTS = Counter(
    "llmo_task_events_total",
    "Durable task lifecycle events.",
    ["tenant", "project", "service", "task", "model", "event"],
)

TASK_ERRORS = Counter(
    "llmo_task_errors_total",
    "Durable task execution errors.",
    ["tenant", "project", "service", "task", "model", "error_type", "retryable"],
)

TASKS_BY_STATE = Gauge(
    "llmo_tasks_by_state",
    "Durable tasks currently stored by state.",
    ["tenant", "project", "service", "task", "model", "state"],
)

TASK_QUEUE_WAIT_SECONDS = Histogram(
    "llmo_task_queue_wait_seconds",
    "Time between durable task creation and first executor claim.",
    ["tenant", "project", "service", "task", "model"],
)

TASK_EXECUTION_SECONDS = Histogram(
    "llmo_task_execution_seconds",
    "Time between durable task first executor claim and terminal completion.",
    ["tenant", "project", "service", "task", "model", "state"],
)

_TASK_COUNT_KEYS: set[tuple[str, str, str, str, str, str]] = set()


def record_snapshot(model: str, active_requests: int, queued_requests: int) -> None:
    ACTIVE_REQUESTS.labels(model=model).set(active_requests)
    QUEUE_LENGTH.labels(model=model).set(queued_requests)


def record_task_event(task: Any, event: str) -> None:
    TASK_EVENTS.labels(
        tenant=task.tenant,
        project=task.project,
        service=task.service,
        task=task.task,
        model=task.model,
        event=event,
    ).inc()


def record_task_error(task: Any, error: dict[str, Any]) -> None:
    TASK_ERRORS.labels(
        tenant=task.tenant,
        project=task.project,
        service=task.service,
        task=task.task,
        model=task.model,
        error_type=str(error.get("type") or "unknown"),
        retryable=str(bool(error.get("retryable"))).lower(),
    ).inc()


def observe_task_queue_wait(task: Any, seconds: float) -> None:
    TASK_QUEUE_WAIT_SECONDS.labels(
        tenant=task.tenant,
        project=task.project,
        service=task.service,
        task=task.task,
        model=task.model,
    ).observe(max(0.0, seconds))


def observe_task_execution(task: Any, state: str, seconds: float) -> None:
    TASK_EXECUTION_SECONDS.labels(
        tenant=task.tenant,
        project=task.project,
        service=task.service,
        task=task.task,
        model=task.model,
        state=state,
    ).observe(max(0.0, seconds))


def record_task_counts(counts: dict[tuple[str, str, str, str, str, str], int]) -> None:
    next_keys = set(counts)
    for stale_key in _TASK_COUNT_KEYS - next_keys:
        TASKS_BY_STATE.labels(
            tenant=stale_key[0],
            project=stale_key[1],
            service=stale_key[2],
            task=stale_key[3],
            model=stale_key[4],
            state=stale_key[5],
        ).set(0)
    for key, count in counts.items():
        TASKS_BY_STATE.labels(
            tenant=key[0],
            project=key[1],
            service=key[2],
            task=key[3],
            model=key[4],
            state=key[5],
        ).set(count)
    _TASK_COUNT_KEYS.clear()
    _TASK_COUNT_KEYS.update(next_keys)
