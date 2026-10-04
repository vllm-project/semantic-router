"""Admission, deadlines and the model worker thread.

Requests are validated and rendered before they reach the queue, so the
queue only holds runnable work. One worker thread owns the model's device:
it takes queued jobs, drops those whose deadline passed, asks each job's
profile to form batches and runs the batches one at a time. A job is answered
as soon as its own batches have run.

Planned batches run in order of their jobs' expected finish: the time a job
was queued plus its tokens at the measured cost per token. Short work goes
first, yet a long job still runs once the work queued before its expected
finish is done, so nothing starves. Between two forwards the worker takes the
jobs that arrived meanwhile (those of profiles that coalesce wait for the
next batching window), so a short request waits for the forward in flight,
not for the rest of a long request. The order of forwards never changes what
a batch contains, so answers are unchanged.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from ..errors import RuntimeServiceError
from ..plugins.base import DEADLINE, Batch, Job, LoadedModel, Profile

__all__ = ["DEADLINE", "Scheduler", "SchedulerLimits"]

# Seconds per token assumed until the first forward is measured, and the
# weight of each later forward in the running estimate.
INITIAL_TOKEN_COST = 2e-5
COST_SMOOTHING = 0.1


@dataclass
class SchedulerLimits:
    max_queue: int = 256
    max_queued_tokens: int = 1 << 22
    batch_window_ms: float = 2.0


@dataclass
class _Pending:
    job: Job
    future: Future
    tokens: int
    results: list[Any] = field(default_factory=list)
    remaining: int = 0
    finish: float = 0.0
    started: bool = False


@dataclass(order=True)
class _Planned:
    """One batch waiting to run, ordered by its earliest owner's expected finish."""

    finish: float
    sequence: int
    batch: Batch = field(compare=False)
    owners: list[_Pending] = field(compare=False)


class Scheduler:
    def __init__(
        self,
        model: LoadedModel,
        profiles: dict[str, Profile],
        limits: SchedulerLimits | None = None,
        observe: Callable[[str, dict[str, Any]], None] | None = None,
        execute: Callable[[Callable[[], Any]], Any] | None = None,
    ):
        """``execute`` runs each batch where the device wants its work (the CPU's one thread)."""
        self.model = model
        self.profiles = profiles
        self.limits = limits or SchedulerLimits()
        self.observe = observe or (lambda event, values: None)
        self.execute = execute or (lambda work: work())
        self._queue: deque[_Pending] = deque()
        self._queued_tokens = 0
        self._lock = threading.Condition()
        self._stopped = False
        self._groups = 0
        self._failure: BaseException | None = None
        self._ready: list[_Planned] = []
        self._sequence = itertools.count()
        self._token_cost: float | None = None
        self._thread = threading.Thread(
            target=self._loop, name="vllm-sr-runtime-worker", daemon=True
        )

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._lock.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=10)

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def depth(self) -> int:
        """Queued jobs plus planned batches still waiting for the device."""
        with self._lock:
            return len(self._queue) + len(self._ready)

    # -- submission ----------------------------------------------------------

    def submit(
        self, items: list[Any], *, deadline: float | None, profile: str
    ) -> Future:
        return self.submit_group([items], deadline=deadline, profile=profile)[0]

    def submit_group(
        self, item_lists: list[list[Any]], *, deadline: float | None, profile: str
    ) -> list[Future]:
        """Queue one job per item list at once, as one group (a bundle's tasks for this model)."""
        if profile not in self.profiles:
            raise RuntimeServiceError(
                "invalid_request", f"profile {profile!r} is not enabled"
            )
        futures: list[Future] = [Future() for _ in item_lists]
        pending = []
        tokens = 0
        enqueued = time.monotonic()
        with self._lock:
            self._groups += 1
            group = self._groups if len(item_lists) > 1 else None
        for future, items in zip(futures, item_lists, strict=True):
            if not items:
                future.set_result([])
                continue
            job_tokens = sum(len(item.ids) for item in items)
            tokens += job_tokens
            job = Job(
                items=items,
                deadline=deadline,
                enqueued=enqueued,
                profile=profile,
                group=group,
            )
            pending.append(_Pending(job, future, job_tokens))
        if not pending:
            return futures
        with self._lock:
            if self._stopped:
                raise RuntimeServiceError("not_ready", "the runtime is shutting down")
            if len(self._queue) + len(pending) > self.limits.max_queue or (
                self._queue
                and self._queued_tokens + tokens > self.limits.max_queued_tokens
            ):
                raise RuntimeServiceError(
                    "overloaded", "the request queue is full; retry later"
                )
            self._queue.extend(pending)
            self._queued_tokens += tokens
            self._lock.notify()
        return futures

    # -- worker --------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            taken = self._take(wait=not self._ready)
            if taken is None:
                self._abandon()
                return
            self._plan(taken)
            if self._ready:
                self._run(heapq.heappop(self._ready))

    def _take(self, *, wait: bool) -> list[_Pending] | None:
        """Queued jobs to plan, or None once stopped.

        With ``wait`` (nothing planned) block until work arrives and, when a
        queued job's profile coalesces, hold the batching window, then take
        everything. Without it take only the jobs of profiles that don't
        coalesce, at once.
        """
        with self._lock:
            if wait:
                while not self._queue and not self._stopped:
                    self._lock.wait()
            if self._stopped:
                return None
            coalescing = [
                p for p in self._queue if self.profiles[p.job.profile].coalesces
            ]
            if not wait:
                if len(coalescing) == len(self._queue):
                    return []
                taken = [
                    p for p in self._queue if not self.profiles[p.job.profile].coalesces
                ]
                self._queue = deque(coalescing)
                self._queued_tokens = sum(p.tokens for p in coalescing)
                return taken
            window = self.limits.batch_window_ms / 1000.0
            if window > 0 and coalescing:
                deadline = time.monotonic() + window
                while (
                    not self._stopped and (remaining := deadline - time.monotonic()) > 0
                ):
                    self._lock.wait(timeout=remaining)
            taken = list(self._queue)
            self._queue.clear()
            self._queued_tokens = 0
            return taken

    def _plan(self, taken: list[_Pending]) -> None:
        now = time.monotonic()
        by_profile: dict[str, list[_Pending]] = {}
        for pending in taken:
            if pending.job.deadline is not None and now > pending.job.deadline:
                self._expire(pending)
                continue
            pending.results = [None] * len(pending.job.items)
            pending.remaining = len(pending.job.items)
            pending.finish = pending.job.enqueued + pending.tokens * (
                self._token_cost or INITIAL_TOKEN_COST
            )
            by_profile.setdefault(pending.job.profile, []).append(pending)
        for name, group in by_profile.items():
            owners = {id(pending.job): pending for pending in group}
            try:
                batches = self.profiles[name].plan(
                    [pending.job for pending in group],
                    self.model.forward_token_budget(),
                )
            except Exception as exc:
                for pending in group:
                    pending.future.set_exception(exc)
                continue
            planned_rows = dict.fromkeys(owners, 0)
            for batch in batches:
                members = {id(job): owners[id(job)] for job, _ in batch.parts}
                for job, indices in batch.parts:
                    planned_rows[id(job)] += len(indices)
                heapq.heappush(
                    self._ready,
                    _Planned(
                        min(member.finish for member in members.values()),
                        next(self._sequence),
                        batch,
                        list(members.values()),
                    ),
                )
            for key, rows in planned_rows.items():
                if rows != owners[key].remaining:
                    owners[key].future.set_exception(
                        RuntimeError(f"profile {name!r} did not plan every item once")
                    )

    def _run(self, planned: _Planned) -> None:
        now = time.monotonic()
        for pending in planned.owners:
            if (
                not pending.future.done()
                and pending.job.deadline is not None
                and now > pending.job.deadline
            ):
                self._expire(pending)
        if all(pending.future.done() for pending in planned.owners):
            return
        batch = planned.batch
        items = batch.items()
        started = time.monotonic()
        for pending in planned.owners:
            if not pending.started:
                pending.started = True
                self.observe("queue", {"seconds": started - pending.job.enqueued})
        try:
            if batch.shared_prefix:
                work = partial(self.model.run, items, shared_prefix=batch.shared_prefix)
            elif not batch.exact:
                work = partial(self.model.run_approximate, items)
            else:
                work = partial(self.model.run, items)
            values = self.execute(work)
        except Exception as exc:
            self._failure = exc
            for pending in planned.owners:
                if not pending.future.done():
                    pending.future.set_exception(exc)
            return
        seconds = time.monotonic() - started
        tokens = sum(len(item.ids) for item in items)
        self.observe(
            "forward", {"seconds": seconds, "rows": len(items), "tokens": tokens}
        )
        if tokens:
            sample = seconds / tokens
            self._token_cost = (
                sample
                if self._token_cost is None
                else self._token_cost + COST_SMOOTHING * (sample - self._token_cost)
            )
        owners = {id(pending.job): pending for pending in planned.owners}
        cursor = 0
        for job, indices in batch.parts:
            pending = owners[id(job)]
            for index in indices:
                pending.results[index] = values[cursor]
                cursor += 1
            pending.remaining -= len(indices)
            if pending.remaining == 0 and not pending.future.done():
                pending.future.set_result(pending.results)

    def _expire(self, pending: _Pending) -> None:
        pending.future.set_result(DEADLINE)
        self.observe("deadline", {"questions": len(pending.job.items)})

    def _abandon(self) -> None:
        """Answer every queued or planned job once the scheduler stops."""
        with self._lock:
            queued = list(self._queue)
            self._queue.clear()
            self._queued_tokens = 0
        planned = [pending for entry in self._ready for pending in entry.owners]
        self._ready.clear()
        error = RuntimeServiceError("not_ready", "the runtime is shutting down")
        for pending in itertools.chain(queued, planned):
            if not pending.future.done():
                pending.future.set_exception(error)
