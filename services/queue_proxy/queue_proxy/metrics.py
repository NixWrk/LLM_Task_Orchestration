from prometheus_client import Counter, Gauge, Histogram

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


def record_snapshot(model: str, active_requests: int, queued_requests: int) -> None:
    ACTIVE_REQUESTS.labels(model=model).set(active_requests)
    QUEUE_LENGTH.labels(model=model).set(queued_requests)
