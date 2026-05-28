import asyncio

import pytest

from queue_proxy.limiter import ModelLimiter, QueueFull


def test_limiter_rejects_when_queue_is_full() -> None:
    async def scenario() -> None:
        limiter = ModelLimiter(
            model="local-main",
            max_active_requests=1,
            max_queued_requests=0,
            queue_timeout_seconds=0.01,
        )
        await limiter.acquire()
        with pytest.raises(QueueFull):
            await limiter.acquire()
        await limiter.release()

    asyncio.run(scenario())


def test_limiter_snapshot_tracks_active_requests() -> None:
    async def scenario() -> None:
        limiter = ModelLimiter(
            model="local-main",
            max_active_requests=1,
            max_queued_requests=1,
            queue_timeout_seconds=0.01,
        )
        await limiter.acquire()
        snapshot = limiter.snapshot()
        assert snapshot.active_requests == 1
        assert snapshot.queued_requests == 0
        await limiter.release()

    asyncio.run(scenario())
