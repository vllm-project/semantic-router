"""Admission, deadlines and the model worker thread.

Requests are validated and rendered before they reach the queue, so the
queue only holds runnable work. One worker thread owns the model's device:
it drains the queue, drops jobs whose deadline passed, asks the active
profile to form batches, runs them and resolves each job's future.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from ..errors import RuntimeServiceError
from ..plugins.base import DEADLINE, Job, LoadedModel, Profile

__all__ = ["DEADLINE", "Scheduler", "SchedulerLimits"]


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
        with self._lock:
            return len(self._queue)

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

    def _take(self) -> list[_Pending]:
        with self._lock:
            while not self._queue and not self._stopped:
                self._lock.wait()
            if self._stopped:
                return []
            window = self.limits.batch_window_ms / 1000.0
            if window > 0 and any(
                self.profiles[p.job.profile].numerics == "approximate"
                for p in self._queue
            ):
                deadline = time.monotonic() + window
                while (
                    not self._stopped and (remaining := deadline - time.monotonic()) > 0
                ):
                    self._lock.wait(timeout=remaining)
            taken = list(self._queue)
            self._queue.clear()
            self._queued_tokens = 0
            return taken

    def _loop(self) -> None:
        while True:
            taken = self._take()
            if not taken:
                return
            now = time.monotonic()
            live = []
            for pending in taken:
                if pending.job.deadline is not None and now > pending.job.deadline:
                    pending.future.set_result(DEADLINE)
                    self.observe("deadline", {"questions": len(pending.job.items)})
                    continue
                self.observe("queue", {"seconds": now - pending.job.enqueued})
                pending.results = [None] * len(pending.job.items)
                live.append(pending)
            by_profile: dict[str, list[_Pending]] = {}
            for pending in live:
                by_profile.setdefault(pending.job.profile, []).append(pending)
            for name, group in by_profile.items():
                self._run_profile(self.profiles[name], group)

    def _run_profile(self, profile: Profile, group: list[_Pending]) -> None:
        owners = {id(pending.job): pending for pending in group}
        try:
            batches = profile.plan(
                [pending.job for pending in group], self.model.forward_token_budget()
            )
        except Exception as exc:
            for pending in group:
                pending.future.set_exception(exc)
            return
        failed: set[int] = set()
        for batch in batches:
            if all(id(job) in failed for job, _ in batch.parts):
                continue
            items = batch.items()
            started = time.monotonic()
            try:
                if batch.shared_prefix:
                    work = partial(
                        self.model.run, items, shared_prefix=batch.shared_prefix
                    )
                elif not batch.exact:
                    work = partial(self.model.run_approximate, items)
                else:
                    work = partial(self.model.run, items)
                values = self.execute(work)
            except Exception as exc:
                self._failure = exc
                for job, _ in batch.parts:
                    if id(job) not in failed:
                        failed.add(id(job))
                        owners[id(job)].future.set_exception(exc)
                continue
            self.observe(
                "forward",
                {
                    "seconds": time.monotonic() - started,
                    "rows": len(items),
                    "tokens": sum(len(item.ids) for item in items),
                },
            )
            cursor = 0
            for job, indices in batch.parts:
                pending = owners[id(job)]
                for index in indices:
                    pending.results[index] = values[cursor]
                    cursor += 1
        for pending in group:
            if id(pending.job) not in failed and not pending.future.done():
                pending.future.set_result(pending.results)
