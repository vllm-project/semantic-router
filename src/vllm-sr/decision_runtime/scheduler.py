"""Bounded, model-scoped admission control for Decision inference."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

ResultT = TypeVar("ResultT")


class SchedulerOverloadedError(RuntimeError):
    """A model has no running or queued capacity left."""


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    model: str
    running: int
    queued: int
    max_concurrency: int
    max_queue: int


@dataclass(slots=True)
class _ModelState:
    condition: asyncio.Condition
    running: int = 0
    queued: int = 0


class ModelScheduler:
    """Apply independent concurrency and queue bounds to every model."""

    def __init__(
        self,
        model_names: Sequence[str],
        *,
        max_concurrency: int = 1,
        max_queue: int = 8,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if max_queue < 0:
            raise ValueError("max_queue must be non-negative")
        if not model_names or len(set(model_names)) != len(model_names):
            raise ValueError("scheduler model names must be nonempty and unique")
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self._states = {
            model: _ModelState(condition=asyncio.Condition()) for model in model_names
        }

    async def run(
        self,
        model: str,
        operation: Callable[[], Awaitable[ResultT]],
    ) -> ResultT:
        state = self._states.get(model)
        if state is None:
            raise KeyError(model)

        async with state.condition:
            if state.running >= self.max_concurrency:
                if state.queued >= self.max_queue:
                    raise SchedulerOverloadedError(model)
                state.queued += 1
                try:
                    await state.condition.wait_for(
                        lambda: state.running < self.max_concurrency
                    )
                except BaseException:
                    state.queued -= 1
                    state.condition.notify(1)
                    raise
                state.queued -= 1
            state.running += 1

        try:
            return await operation()
        finally:
            await _release_running_slot(state)

    async def snapshots(self) -> tuple[SchedulerSnapshot, ...]:
        snapshots = []
        for model, state in self._states.items():
            async with state.condition:
                snapshots.append(
                    SchedulerSnapshot(
                        model=model,
                        running=state.running,
                        queued=state.queued,
                        max_concurrency=self.max_concurrency,
                        max_queue=self.max_queue,
                    )
                )
        return tuple(snapshots)


async def _release_running_slot(state: _ModelState) -> None:
    """Release a slot even if the caller is cancelled more than once.

    A second cancellation can otherwise interrupt acquisition of the condition
    lock inside ``finally`` and leak ``running`` capacity permanently. Keep the
    tiny state transition in its own task and shield that task until it has
    completed. If cancellation arrives while cleanup is pending, propagate it
    only after the scheduler state is consistent again.
    """

    async def cleanup() -> None:
        async with state.condition:
            state.running -= 1
            state.condition.notify(1)

    cleanup_task = asyncio.create_task(cleanup())
    interrupted = False
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            interrupted = True
    await cleanup_task
    if interrupted:
        raise asyncio.CancelledError
