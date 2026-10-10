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
jobs that arrived meanwhile (those of profiles that coalesce once they have
waited one batching window), so a short request waits for the forward in flight,
not for the rest of a long request. The order of forwards never changes what
a batch contains, so answers are unchanged.

A model whose batches need no device thread may also run a request on the
thread that planned it (``run_now``), but only when the scheduler is idle:
nothing queued or planned and the worker not running a batch. Whoever runs
batches owns the model until they are done, so the caller and the worker
never run one model at once. Once other jobs are queued the caller stops
after its forward in flight and leaves its remaining batches to the worker,
so arrivals still wait for one forward, not for the rest of the request.
"""

from __future__ import annotations

import contextlib
import heapq
import itertools
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from ..errors import RuntimeServiceError
from ..plugins.base import DEADLINE, Batch, Job, LoadedModel, Profile, Results
from ..timing import RunTiming
from .planner import cost

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
    future: Future[Results[Any]]
    tokens: int
    results: list[Any] = field(default_factory=list)
    remaining: int = 0
    finish: float = 0.0
    started: bool = False
    timing: RunTiming | None = None


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
        model: LoadedModel[Any, Any],
        profiles: dict[str, Profile],
        limits: SchedulerLimits | None = None,
        observe: Callable[[str, dict[str, Any]], None] | None = None,
        execute: Callable[[Callable[[], Any]], Any] | None = None,
        device_fault: Callable[[BaseException], bool] | None = None,
    ):
        """``execute`` runs each batch where the device wants its work (the CPU's one thread).

        ``device_fault`` tells an error that left the device unusable, which is
        recorded as the scheduler's ``failure``, from any other error in a
        forward, which fails only the jobs of that batch.
        """
        self.model = model
        self.profiles = profiles
        self.limits = limits or SchedulerLimits()
        self.observe = observe or (lambda event, values: None)
        self.execute = execute or (lambda work: work())
        self.device_fault = device_fault or (lambda error: False)
        self._queue: deque[_Pending] = deque()
        # Admitted jobs not yet answered, queued or planned, and their tokens.
        self._pending_jobs = 0
        self._pending_tokens = 0
        self._lock = threading.Condition()
        self._stopped = False
        self._groups = 0
        self._failure: BaseException | None = None
        self._ready: list[_Planned] = []
        self._sequence = itertools.count()
        self._token_cost: float | None = None
        self._owned = False
        self._thread = threading.Thread(
            target=self._loop, name="vllm-srun-worker", daemon=True
        )

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> bool:
        """Stop taking work and answer what is left; whether the worker has exited."""
        with self._lock:
            self._stopped = True
            self._lock.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        return not self._thread.is_alive()

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
    ) -> Future[Results[Any]]:
        return self.submit_group([items], deadlines=[deadline], profile=profile)[0]

    def submit_group(
        self,
        item_lists: list[list[Any]],
        *,
        deadlines: list[float | None],
        profile: str,
        timing: RunTiming | None = None,
    ) -> list[Future[Results[Any]]]:
        """Queue one job per item list at once, as one group (a bundle's tasks for this model).

        ``deadlines`` holds each job's own deadline. Admission counts every
        job not yet answered, queued or planned, and refuses the whole group
        or none of it. ``timing`` records the group's forwards.
        """
        futures, pending = self._jobs(item_lists, deadlines, profile, timing)
        if not pending:
            return futures
        tokens = sum(entry.tokens for entry in pending)
        with self._lock:
            if self._stopped:
                raise RuntimeServiceError("not_ready", "the runtime is shutting down")
            if self._pending_jobs + len(pending) > self.limits.max_queue or (
                self._pending_jobs
                and self._pending_tokens + tokens > self.limits.max_queued_tokens
            ):
                raise RuntimeServiceError(
                    "overloaded", "the request queue is full; retry later"
                )
            self._admit(pending)
            self._queue.extend(pending)
            self._lock.notify()
        return futures

    def run_now(
        self,
        item_lists: list[list[Any]],
        *,
        deadlines: list[float | None],
        profile: str,
        timing: RunTiming | None = None,
    ) -> list[Future[Results[Any]]] | None:
        """Run a group on the calling thread if the scheduler is idle, else None.

        Never call it from the event loop: batches run before it returns. When
        other jobs arrive meanwhile, the worker runs the group's remaining
        batches in order with theirs and answers the futures. Models whose
        batches go to a device thread always answer None.
        """
        if self.model.device_thread:
            return None
        with self._lock:
            if self._stopped or self._owned or self._queue or self._ready:
                return None
            self._owned = True
        try:
            futures, pending = self._jobs(item_lists, deadlines, profile, timing)
            with self._lock:
                self._admit(pending)
            self._plan(pending)
            while self._ready:
                self._run(heapq.heappop(self._ready))
                with self._lock:
                    if self._queue:
                        break
        finally:
            self._release()
        return futures

    def _jobs(
        self,
        item_lists: list[list[Any]],
        deadlines: list[float | None],
        profile: str,
        timing: RunTiming | None,
    ) -> tuple[list[Future[Results[Any]]], list[_Pending]]:
        if profile not in self.profiles:
            raise RuntimeServiceError(
                "invalid_request", f"profile {profile!r} is not enabled"
            )
        futures: list[Future[Results[Any]]] = [Future() for _ in item_lists]
        pending = []
        enqueued = time.monotonic()
        with self._lock:
            self._groups += 1
            group = self._groups if len(item_lists) > 1 else None
        for future, items, deadline in zip(futures, item_lists, deadlines, strict=True):
            if not items:
                future.set_result([])
                continue
            job = Job(
                items=items,
                deadline=deadline,
                enqueued=enqueued,
                profile=profile,
                group=group,
            )
            pending.append(
                _Pending(job, future, sum(cost(item) for item in items), timing=timing)
            )
        return futures, pending

    def _admit(self, pending: list[_Pending]) -> None:
        """Count jobs as pending until they are answered (the caller holds the lock)."""
        for entry in pending:
            self._pending_jobs += 1
            self._pending_tokens += entry.tokens
            entry.future.add_done_callback(partial(self._settle, entry))

    def _settle(self, entry: _Pending, future: Future[Results[Any]]) -> None:
        """Release an answered job's admission; a cancelled one also leaves the queue."""
        with self._lock:
            self._pending_jobs -= 1
            self._pending_tokens -= entry.tokens
            if future.cancelled():
                with contextlib.suppress(ValueError):
                    self._queue.remove(entry)

    def _release(self) -> None:
        with self._lock:
            self._owned = False
            self._lock.notify_all()

    # -- worker --------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            taken = self._take(wait=not self._ready)
            if taken is None:
                self._abandon()
                return
            try:
                self._plan(taken)
                if self._ready:
                    self._run(heapq.heappop(self._ready))
            finally:
                self._release()

    def _take(self, *, wait: bool) -> list[_Pending] | None:
        """Own the model and take queued jobs to plan, or None once stopped.

        With ``wait`` (nothing planned) block until work arrives and, when a
        queued job's profile coalesces, hold the batching window, then take
        everything. Without it take at once the jobs of profiles that don't
        coalesce and the coalescing jobs that have already waited a window.
        Either way wait while a ``run_now`` caller owns the model; batches a
        caller left planned count as planned work.
        """
        window = self.limits.batch_window_ms / 1000.0
        with self._lock:
            while (
                self._owned or (wait and not self._queue and not self._ready)
            ) and not self._stopped:
                self._lock.wait()
            if self._stopped:
                return None
            self._owned = True
            if not wait or self._ready:
                now = time.monotonic()
                taken: list[_Pending] = []
                kept: deque[_Pending] = deque()
                for pending in self._queue:
                    ripe = now - pending.job.enqueued >= window
                    if ripe or not self.profiles[pending.job.profile].coalesces:
                        taken.append(pending)
                    else:
                        kept.append(pending)
                self._queue = kept
                return taken
            coalescing = any(
                self.profiles[pending.job.profile].coalesces for pending in self._queue
            )
            if window > 0 and coalescing:
                deadline = time.monotonic() + window
                while (
                    not self._stopped and (remaining := deadline - time.monotonic()) > 0
                ):
                    self._lock.wait(timeout=remaining)
            taken = list(self._queue)
            self._queue.clear()
            return taken

    def _plan(self, taken: list[_Pending]) -> None:
        now = time.monotonic()
        by_profile: dict[str, list[_Pending]] = {}
        for pending in taken:
            if pending.future.done():
                continue
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
                    _fail(pending.future, exc)
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
                    _fail(
                        owners[key].future,
                        RuntimeError(f"profile {name!r} did not plan every item once"),
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
                work = partial(self.model.run_shared, items, batch.shared_prefix)
            elif not batch.exact:
                work = partial(self.model.run_approximate, items)
            else:
                work = partial(self.model.run, items)
            values = self.execute(work)
        except Exception as exc:
            _timed(planned, started, time.monotonic())
            if self.device_fault(exc):
                self._failure = exc
            for pending in planned.owners:
                _fail(pending.future, exc)
            return
        ended = time.monotonic()
        _timed(planned, started, ended)
        seconds = ended - started
        tokens = sum(cost(item) for item in items)
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
            if pending.remaining == 0:
                _resolve(pending.future, pending.results)

    def _expire(self, pending: _Pending) -> None:
        _resolve(pending.future, DEADLINE)
        self.observe("deadline", {"questions": len(pending.job.items)})

    def _abandon(self) -> None:
        """Answer every queued or planned job once the scheduler stops."""
        with self._lock:
            while self._owned:
                self._lock.wait()
            queued = list(self._queue)
            self._queue.clear()
        planned = [pending for entry in self._ready for pending in entry.owners]
        self._ready.clear()
        error = RuntimeServiceError("not_ready", "the runtime is shutting down")
        for pending in itertools.chain(queued, planned):
            _fail(pending.future, error)


def _timed(planned: _Planned, started: float, ended: float) -> None:
    """Record a forward in the timing of every group it ran items for, before any is answered."""
    for pending in planned.owners:
        if pending.timing is not None and not pending.future.done():
            pending.timing.ran(planned.sequence, ended - started, ended)


def _resolve(future: Future[Results[Any]], result: Results[Any]) -> None:
    """Answer ``future`` unless it is answered or its caller cancelled it (the client left)."""
    with contextlib.suppress(InvalidStateError):
        future.set_result(result)


def _fail(future: Future[Results[Any]], error: BaseException) -> None:
    with contextlib.suppress(InvalidStateError):
        future.set_exception(error)
