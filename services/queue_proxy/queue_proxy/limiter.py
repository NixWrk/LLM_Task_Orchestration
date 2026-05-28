from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator


class QueueFull(Exception):
    pass


class QueueTimeout(Exception):
    pass


@dataclass(frozen=True)
class LimiterSnapshot:
    model: str
    active_requests: int
    queued_requests: int
    max_active_requests: int
    max_queued_requests: int


class ModelLimiter:
    def __init__(
        self,
        model: str,
        max_active_requests: int,
        max_queued_requests: int,
        queue_timeout_seconds: float,
    ) -> None:
        self.model = model
        self.max_active_requests = max_active_requests
        self.max_queued_requests = max_queued_requests
        self.queue_timeout_seconds = queue_timeout_seconds
        self._active_requests = 0
        self._queued_requests = 0
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            await self.release()

    async def acquire(self) -> None:
        async with self._condition:
            if self._active_requests < self.max_active_requests:
                self._active_requests += 1
                return

            if self._queued_requests >= self.max_queued_requests:
                raise QueueFull

            self._queued_requests += 1
            try:
                await asyncio.wait_for(
                    self._wait_for_slot_locked(),
                    timeout=self.queue_timeout_seconds,
                )
            except TimeoutError as exc:
                self._queued_requests -= 1
                self._condition.notify_all()
                raise QueueTimeout from exc

    async def _wait_for_slot_locked(self) -> None:
        while self._active_requests >= self.max_active_requests:
            await self._condition.wait()
        self._queued_requests -= 1
        self._active_requests += 1

    async def release(self) -> None:
        async with self._condition:
            self._active_requests = max(0, self._active_requests - 1)
            self._condition.notify_all()

    def snapshot(self) -> LimiterSnapshot:
        return LimiterSnapshot(
            model=self.model,
            active_requests=self._active_requests,
            queued_requests=self._queued_requests,
            max_active_requests=self.max_active_requests,
            max_queued_requests=self.max_queued_requests,
        )


class LimiterRegistry:
    def __init__(self) -> None:
        self._limiters: dict[str, ModelLimiter] = {}

    def get_or_create(
        self,
        model: str,
        max_active_requests: int,
        max_queued_requests: int,
        queue_timeout_seconds: float,
    ) -> ModelLimiter:
        if model not in self._limiters:
            self._limiters[model] = ModelLimiter(
                model,
                max_active_requests,
                max_queued_requests,
                queue_timeout_seconds,
            )
        return self._limiters[model]

    def snapshots(self) -> list[LimiterSnapshot]:
        return [limiter.snapshot() for limiter in self._limiters.values()]
