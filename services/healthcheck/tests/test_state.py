from app.state import DEGRADED, HEALTHY, UNKNOWN, UNHEALTHY, overall_status, state_value


def test_overall_status_healthy_when_all_checks_are_healthy() -> None:
    assert overall_status([{"status": HEALTHY}, {"status": HEALTHY}]) == HEALTHY


def test_overall_status_degraded_when_any_check_is_degraded() -> None:
    assert overall_status([{"status": HEALTHY}, {"status": DEGRADED}]) == DEGRADED


def test_overall_status_unhealthy_takes_precedence() -> None:
    assert overall_status([{"status": DEGRADED}, {"status": UNHEALTHY}]) == UNHEALTHY


def test_overall_status_unknown_for_empty_checks() -> None:
    assert overall_status([]) == UNKNOWN


def test_state_value_unknown_for_unrecognized_status() -> None:
    assert state_value("not-a-real-state") == state_value(UNKNOWN)
