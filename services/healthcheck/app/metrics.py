from prometheus_client import Counter, Gauge, Histogram

from app.state import state_value

BACKEND_HEALTH = Gauge(
    "llm_backend_health",
    "Readiness state of an LLM backend: healthy=1, degraded=0.5, unhealthy=0, unknown=-1.",
    ["backend"],
)

LOADED_MODELS = Gauge(
    "llm_loaded_models",
    "Number of models visible through the LM Studio OpenAI-compatible /v1/models endpoint.",
)

HEALTHCHECK_LATENCY = Histogram(
    "llm_healthcheck_latency_seconds",
    "Latency of the complete readiness check.",
)

HEALTHCHECK_ERRORS = Counter(
    "llm_healthcheck_errors_total",
    "Number of failed readiness checks.",
    ["check"],
)


def record_backend_health(backend: str, status: str) -> None:
    BACKEND_HEALTH.labels(backend=backend).set(state_value(status))


def record_loaded_models(count: int) -> None:
    LOADED_MODELS.set(count)


def record_readiness_latency(seconds: float) -> None:
    HEALTHCHECK_LATENCY.observe(seconds)


def record_error(check_name: str) -> None:
    HEALTHCHECK_ERRORS.labels(check=check_name).inc()
