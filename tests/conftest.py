from collections.abc import Callable

import pytest


@pytest.fixture
def unused_tcp_port_factory(free_tcp_port_factory: Callable[[], int]) -> Callable[[], int]:
    return free_tcp_port_factory
