"""Admission, deadlines and the model worker thread.

Requests are validated and rendered before they reach the queue, so the
queue only holds runnable work. One worker thread owns the model's device:
it takes queued jobs, drops those whose deadline passed, asks the model's
batcher (``batching.Batcher``, from its engine arguments) to form batches
and runs the batches one at a time. A job is answered as soon as its own
batches have run.

Planned batches run in order of their jobs' expected finish: the time a job
was queued plus its tokens at the measured cost per token of their lane
(``LoadedModel.lane``), each lane measured from its own forwards, so slow
image forwards do not make text rows look slow. The batches of different
lanes that one planning round makes share a turn (``Batch.sub``): each runs
as its own forward, the one expected to take least first, and its jobs are
answered as soon as it ran, so text rows never wait for an image forward and
a round never leaves more turns behind than it runs. Short work goes
first, yet a long job still runs once the work queued before its expected
finish is done, so nothing starves. Between two forwards the worker takes the
jobs that arrived meanwhile (on a model whose batcher coalesces, once they
have waited one batching window), so a short request waits for the forward in
flight, not for the rest of a long request, and plans them together with the
jobs of planned batches that have not started, so concurrent requests fill
forwards up to their budget instead of keeping the batches their arrival
formed. The order of forwards never changes what a batch contains, so answers
are unchanged.

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
from ..plugins.base import DEADLINE, Batch, Job, LoadedModel, Results
from ..timing import RunTiming
from .batching import Batcher, reference
from .planner import cost

__all__ = ["DEADLINE", "Scheduler", "SchedulerLimits"]

# Seconds per token assumed until a lane's first forward is measured, and the
# weight of each later forward in the lane's running estimate.
INITIAL_TOKEN_COST = 2e-5
COST_SMOOTHING = 0.1
# A batching window waits for requests still being planned when it opened for
# at most this many windows, checking every BURST_POLL_S once its own has passed.
BURST_WINDOWS = 3
BURST_POLL_S = 0.00025


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
    lanes: dict[str, int] = field(default_factory=dict)
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
        batcher: Batcher,
        limits: SchedulerLimits | None = None,
        observe: Callable[[str, dict[str, Any]], None] | None = None,
        execute: Callable[[Callable[[], Any]], Any] | None = None,
        device_fault: Callable[[BaseException], bool] | None = None,
        planning: Callable[[float], bool] | None = None,
    ):
        """``execute`` runs each batch where the device wants its work (the CPU's one thread).

        ``device_fault`` tells an error that left the device unusable, which is
        recorded as the scheduler's ``failure``, from any other error in a
        forward, which fails only the jobs of that batch. ``planning(opened)``
        says whether a request received before ``opened`` is still being
        planned: a batching window then waits for it (``_take``).
        """
        self.model = model
        self.batcher = batcher
        self.limits = limits or SchedulerLimits()
        self.observe = observe or (lambda event, values: None)
        self.execute = execute or (lambda work: work())
        self.device_fault = device_fault or (lambda error: False)
        self.planning = planning or (lambda opened: False)
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
        self._token_cost: dict[str, float] = {}
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
        self, items: list[Any], *, deadline: float | None, reference: bool = False
    ) -> Future[Results[Any]]:
        return self.submit_group([items], deadlines=[deadline], reference=reference)[0]

    def submit_group(
        self,
        item_lists: list[list[Any]],
        *,
        deadlines: list[float | None],
        timing: RunTiming | None = None,
        reference: bool = False,
        alone: bool = False,
    ) -> list[Future[Results[Any]]]:
        """Queue one job per item list at once, as one group (a bundle's tasks for this model).

        ``deadlines`` holds each job's own deadline. Admission counts every
        job not yet answered, queued or planned, and refuses the whole group
        or none of it. ``timing`` records the group's forwards. ``reference``
        jobs (the golden check) run alone, in their released forwards.
        ``alone`` jobs (``Job.alone``) are planned without the batching window.
        """
        futures, pending = self._jobs(item_lists, deadlines, timing, reference, alone)
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
            futures, pending = self._jobs(item_lists, deadlines, timing)
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
        timing: RunTiming | None,
        reference: bool = False,
        alone: bool = False,
    ) -> tuple[list[Future[Results[Any]]], list[_Pending]]:
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
                group=group,
                reference=reference,
                alone=alone,
            )
            lanes: dict[str, int] = {}
            for item in items:
                lane = self._lane(item)
                lanes[lane] = lanes.get(lane, 0) + cost(item)
            pending.append(
                _Pending(job, future, sum(lanes.values()), lanes, timing=timing)
            )
        return futures, pending

    def _lane(self, item: Any) -> str:
        lane = getattr(self.model, "lane", None)
        return lane(item) if lane is not None else "text"

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
                self._plan(self._unstarted() + taken if taken else taken)
                if self._ready:
                    self._run(heapq.heappop(self._ready))
            finally:
                self._release()

    def _unstarted(self) -> list[_Pending]:
        """Take back the planned batches whose jobs have not started, to plan them with new arrivals.

        A planning round sees only the jobs that arrived during the forward
        before it, so without this, concurrent requests would keep the small
        batches their arrival formed and never fill a forward's budget. Only
        whole jobs of no other batch come back; reference jobs keep their
        released forwards.
        """
        batches: dict[int, int] = {}
        for entry in self._ready:
            for pending in entry.owners:
                batches[id(pending)] = batches.get(id(pending), 0) + 1
        back: list[_Pending] = []
        kept: list[_Planned] = []
        for entry in self._ready:
            if all(
                not pending.started
                and not pending.job.reference
                and batches[id(pending)] == 1
                for pending in entry.owners
            ):
                back.extend(entry.owners)
            else:
                kept.append(entry)
        if back:
            heapq.heapify(kept)
            self._ready = kept
        return sorted(back, key=lambda pending: pending.job.enqueued)

    def _take(self, *, wait: bool) -> list[_Pending] | None:
        """Own the model and take queued jobs to plan, or None once stopped.

        With ``wait`` (nothing planned) block until work arrives and, when the
        batcher coalesces and a queued job may merge, hold the batching window,
        then take everything. The window also waits, up to ``BURST_WINDOWS``
        windows, while requests received before it opened are still being
        planned, so a burst of concurrent requests forms one batch whatever
        their planning takes. Without it take at once the jobs that need not
        wait (all of them when the batcher does not coalesce, and references)
        and the others once they have waited a window. Either way wait while a
        ``run_now`` caller owns the model; batches a caller left planned count
        as planned work.
        """
        with self._lock:
            while (
                self._owned or (wait and not self._queue and not self._ready)
            ) and not self._stopped:
                self._lock.wait()
            if self._stopped:
                return None
            self._owned = True
            window = self.limits.batch_window_ms / 1000.0
            if not wait or self._ready:
                now = time.monotonic()
                taken: list[_Pending] = []
                kept: deque[_Pending] = deque()
                for pending in self._queue:
                    ripe = now - pending.job.enqueued >= window
                    if ripe or not self._waits(pending):
                        taken.append(pending)
                    else:
                        kept.append(pending)
                self._queue = kept
                return taken
            coalescing = any(self._waits(pending) for pending in self._queue)
            if window > 0 and coalescing:
                opened = time.monotonic()
                deadline, limit = opened + window, opened + BURST_WINDOWS * window
                while not self._stopped:
                    now = time.monotonic()
                    if now >= deadline and (now >= limit or not self.planning(opened)):
                        break
                    until = (
                        deadline if now < deadline else min(limit, now + BURST_POLL_S)
                    )
                    self._lock.wait(timeout=until - now)
            taken = list(self._queue)
            self._queue.clear()
            return taken

    def _waits(self, pending: _Pending) -> bool:
        """Whether a queued job waits for the batching window before it is planned."""
        job = pending.job
        return self.batcher.coalesces and not job.reference and not job.alone

    def _plan(self, taken: list[_Pending]) -> None:
        now = time.monotonic()
        by_kind: dict[bool, list[_Pending]] = {}
        for pending in taken:
            if pending.future.done():
                continue
            if pending.job.deadline is not None and now > pending.job.deadline:
                self._expire(pending)
                continue
            pending.results = [None] * len(pending.job.items)
            pending.remaining = len(pending.job.items)
            pending.finish = pending.job.enqueued + sum(
                tokens * self._token_cost.get(lane, INITIAL_TOKEN_COST)
                for lane, tokens in pending.lanes.items()
            )
            by_kind.setdefault(pending.job.reference, []).append(pending)
        for references, group in by_kind.items():
            owners = {id(pending.job): pending for pending in group}
            jobs = [pending.job for pending in group]
            budget = self.model.forward_token_budget()
            try:
                batches = (
                    reference(self.model, jobs, budget)
                    if references
                    else self.batcher.plan(jobs, budget)
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
                        RuntimeError("the batcher did not plan every item once"),
                    )

    def _run(self, planned: _Planned) -> None:
        """Run a planned batch, or each batch of a turn that merges lanes, cheapest first."""
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
        owners = {id(pending.job): pending for pending in planned.owners}
        batches = sorted(planned.batch.sub or (planned.batch,), key=self._seconds)
        for batch in batches:
            members = [owners[id(job)] for job, _ in batch.parts]
            if self._failure is not None:
                for pending in members:
                    _fail(pending.future, self._failure)
            elif not all(pending.future.done() for pending in members):
                self._run_batch(batch, members, next(self._sequence))

    def _seconds(self, batch: Batch) -> float:
        """A batch's expected forward time at its lane's measured cost per token."""
        tokens = sum(cost(item) for item in batch.items())
        return tokens * self._token_cost.get(batch.lane, INITIAL_TOKEN_COST)

    def _run_batch(self, batch: Batch, members: list[_Pending], sequence: int) -> None:
        items = batch.items()
        started = time.monotonic()
        for pending in members:
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
            # Validate the whole forward before publishing any job's results.
            if len(values) != len(items):
                raise RuntimeError(
                    f"model result count mismatch: expected {len(items)}, got {len(values)}"
                )
        except Exception as exc:
            _timed(members, sequence, started, time.monotonic())
            if self.device_fault(exc):
                self._failure = exc
            error = _out_of_memory(exc, len(members)) or exc
            for pending in members:
                _fail(pending.future, error)
            return
        ended = time.monotonic()
        _timed(members, sequence, started, ended)
        seconds = ended - started
        tokens = sum(cost(item) for item in items)
        self.observe(
            "forward", {"seconds": seconds, "rows": len(items), "tokens": tokens}
        )
        lanes = (
            {batch.lane} if not batch.exact else {self._lane(item) for item in items}
        )
        if tokens and len(lanes) == 1:
            lane = lanes.pop()
            sample = seconds / tokens
            known = self._token_cost.get(lane)
            self._token_cost[lane] = (
                sample if known is None else known + COST_SMOOTHING * (sample - known)
            )
        owners = {id(pending.job): pending for pending in members}
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


def _timed(
    members: list[_Pending], sequence: int, started: float, ended: float
) -> None:
    """Record a forward in the timing of every group it ran items for, before any is answered."""
    for pending in members:
        if pending.timing is not None and not pending.future.done():
            pending.timing.ran(sequence, ended - started, ended)


def _resolve(future: Future[Results[Any]], result: Results[Any]) -> None:
    """Answer ``future`` unless it is answered or its caller cancelled it (the client left)."""
    with contextlib.suppress(InvalidStateError):
        future.set_result(result)


def _out_of_memory(error: BaseException, jobs: int) -> RuntimeServiceError | None:
    """A forward out of device memory as the request's answer: too large alone, overloaded beside other requests."""
    if type(error).__name__ != "OutOfMemoryError":  # torch's, on any accelerator
        return None
    if jobs == 1:
        return RuntimeServiceError(
            "request_too_large",
            "the request needs more device memory than the model has free",
        )
    return RuntimeServiceError(
        "overloaded", "the batch holding this request ran out of device memory"
    )


def _fail(future: Future[Results[Any]], error: BaseException) -> None:
    with contextlib.suppress(InvalidStateError):
        future.set_exception(error)
