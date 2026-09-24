"""Bounded, model-scoped admission control for Decision inference."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

ResultT = TypeVar("ResultT")

DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_MAX_QUEUE = 8
# Once the original row-credit deficit has been released, bound additional
# bypasses before reserving capacity for the older request.
MAX_ROW_CREDIT_BYPASSES = 8


class SchedulerOverloadedError(RuntimeError):
    """A model has no running or queued capacity left."""


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    model: str
    running: int
    queued: int
    active_rows: int
    max_concurrency: int
    max_queue: int
    max_active_rows: int | None


@dataclass(slots=True)
class _ModelState:
    condition: asyncio.Condition
    running: int = 0
    active_rows: int = 0
    running_tokens: set[_Waiter] = field(default_factory=set)
    waiters: deque[_Waiter] = field(default_factory=deque)


@dataclass(eq=False, slots=True)
class _Waiter:
    row_cost: int
    bypasses: int = 0
    initial_deficit: int = 0
    released_incumbent_rows: int = 0
    incumbents: set[_Waiter] = field(default_factory=set)


class ModelScheduler:
    """Bound concurrent requests and their total decision-row liability per model.

    Row credits are reserved before preparation and held through response
    assembly. Waiting requests hold neither a running slot nor row credits.
    Fitting younger requests can use idle credits while the requests that
    originally blocked an older waiter are still running. Once those original
    requests have released enough rows for it to fit, any further bypass is
    bounded and the older request gains admission priority.
    """

    def __init__(
        self,
        model_names: Sequence[str],
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_queue: int = DEFAULT_MAX_QUEUE,
        max_active_rows: int | None = None,
    ) -> None:
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if type(max_queue) is not int or max_queue < 0:
            raise ValueError("max_queue must be non-negative")
        if max_active_rows is not None and (
            type(max_active_rows) is not int or max_active_rows < 1
        ):
            raise ValueError("max_active_rows must be positive")
        if not model_names or len(set(model_names)) != len(model_names):
            raise ValueError("scheduler model names must be nonempty and unique")
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.max_active_rows = max_active_rows
        self._states = {
            model: _ModelState(condition=asyncio.Condition()) for model in model_names
        }

    async def run(
        self,
        model: str,
        operation: Callable[[], Awaitable[ResultT]],
        *,
        row_cost: int = 1,
    ) -> ResultT:
        state = self._states.get(model)
        if state is None:
            raise KeyError(model)
        if type(row_cost) is not int or row_cost < 1:
            raise ValueError("row_cost must be positive")
        if self.max_active_rows is not None and row_cost > self.max_active_rows:
            raise ValueError("row_cost exceeds max_active_rows")

        candidate = _Waiter(row_cost=row_cost)
        async with state.condition:
            queued = not self._can_admit(state, candidate)
            if queued:
                if len(state.waiters) >= self.max_queue:
                    raise SchedulerOverloadedError(model)
                if self.max_active_rows is not None:
                    candidate.initial_deficit = max(
                        0, state.active_rows + row_cost - self.max_active_rows
                    )
                candidate.incumbents.update(state.running_tokens)
                state.waiters.append(candidate)
                try:
                    await state.condition.wait_for(
                        lambda: self._can_admit(state, candidate)
                    )
                except BaseException:
                    state.waiters.remove(candidate)
                    state.condition.notify_all()
                    raise
            self._mark_bypassed_waiters(state, candidate)
            if queued:
                state.waiters.remove(candidate)
            state.running += 1
            state.active_rows += row_cost
            state.running_tokens.add(candidate)
            state.condition.notify_all()

        try:
            return await operation()
        finally:
            await _release_running_slot(state, candidate)

    def _can_run(self, state: _ModelState, row_cost: int) -> bool:
        return state.running < self.max_concurrency and (
            self.max_active_rows is None
            or state.active_rows + row_cost <= self.max_active_rows
        )

    def _can_admit(self, state: _ModelState, candidate: _Waiter) -> bool:
        if not self._can_run(state, candidate.row_cost):
            return False
        for older in state.waiters:
            if older is candidate:
                break
            if (
                older.bypasses >= MAX_ROW_CREDIT_BYPASSES
                or self._can_run(state, older.row_cost)
            ):
                return False
        return True

    @staticmethod
    def _mark_bypassed_waiters(state: _ModelState, candidate: _Waiter) -> None:
        for older in state.waiters:
            if older is candidate:
                break
            if older.released_incumbent_rows >= older.initial_deficit:
                older.bypasses += 1

    async def snapshots(self) -> tuple[SchedulerSnapshot, ...]:
        snapshots = []
        for model, state in self._states.items():
            async with state.condition:
                snapshots.append(
                    SchedulerSnapshot(
                        model=model,
                        running=state.running,
                        queued=len(state.waiters),
                        active_rows=state.active_rows,
                        max_concurrency=self.max_concurrency,
                        max_queue=self.max_queue,
                        max_active_rows=self.max_active_rows,
                    )
                )
        return tuple(snapshots)


async def _release_running_slot(state: _ModelState, candidate: _Waiter) -> None:
    """Release a slot and its row credits despite repeated cancellation.

    A second cancellation can otherwise interrupt acquisition of the condition
    lock inside ``finally`` and leak ``running`` capacity permanently. Keep the
    tiny state transition in its own task and shield that task until it has
    completed. If cancellation arrives while cleanup is pending, propagate it
    only after the scheduler state is consistent again.
    """

    async def cleanup() -> None:
        async with state.condition:
            state.running -= 1
            state.active_rows -= candidate.row_cost
            state.running_tokens.remove(candidate)
            for waiter in state.waiters:
                if candidate in waiter.incumbents:
                    waiter.incumbents.remove(candidate)
                    waiter.released_incumbent_rows += candidate.row_cost
            state.condition.notify_all()

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
