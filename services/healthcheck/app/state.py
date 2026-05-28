from __future__ import annotations

from typing import Iterable, Mapping

HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"
UNKNOWN = "unknown"

STATE_VALUES = {
    HEALTHY: 1.0,
    DEGRADED: 0.5,
    UNHEALTHY: 0.0,
    UNKNOWN: -1.0,
}


def overall_status(checks: Iterable[Mapping[str, object]]) -> str:
    statuses = [str(check.get("status", UNKNOWN)) for check in checks]

    if not statuses:
        return UNKNOWN
    if UNHEALTHY in statuses:
        return UNHEALTHY
    if UNKNOWN in statuses:
        return UNKNOWN
    if DEGRADED in statuses:
        return DEGRADED
    return HEALTHY


def state_value(status: str) -> float:
    return STATE_VALUES.get(status, STATE_VALUES[UNKNOWN])
