"""Fair cross-request physical batching for one resident Decision model.

The public API admits requests. This module independently flattens their
questions into model rows, coalesces rows from concurrent callers, and restores
the original request/state/question identity after inference. A large batch
request therefore cannot silently monopolize every physical batch.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .backend import (
    BackendBatchRequest,
    BackendBatchResult,
    BackendContractError,
    BackendOverloadedError,
    BackendPrediction,
    BackendResult,
    BackendUnavailableError,
    DecisionBackend,
    ModelDescriptor,
    UnknownModelError,
)
from .contracts import JsonContent, Question, SystemOneRequest


class PhysicalBatchOverloadedError(BackendOverloadedError):
    """The bounded row queue cannot atomically admit another request."""


@dataclass(frozen=True, slots=True)
class DecisionRow:
    """One independently scored question row passed to a model-family runtime."""

    model: str
    state: JsonContent
    question_id: str
    question: Question


@dataclass(frozen=True, slots=True)
class DecisionRowResult:
    """One row result in the exact same order as the executor input."""

    question_id: str
    type: Literal["noul", "choice", "score"]
    probabilities: tuple[float, ...]
    input_tokens: int


class DecisionRowExecutor(Protocol):
    """Backend-specific resident model that scores one physical row batch."""

    async def ready(self) -> bool:
        """Return whether the resident model can accept work."""

    async def predict_rows(
        self, rows: tuple[DecisionRow, ...]
    ) -> tuple[DecisionRowResult, ...]:
        """Score rows without changing their order or identity."""


@dataclass(slots=True)
class _RowJob:
    rows: tuple[DecisionRow, ...]
    future: asyncio.Future[tuple[DecisionRowResult, ...]]
    cursor: int = 0
    results: list[DecisionRowResult | None] = field(init=False)

    def __post_init__(self) -> None:
        self.results = [None] * len(self.rows)

    @property
    def remaining(self) -> int:
        return len(self.rows) - self.cursor


@dataclass(frozen=True, slots=True)
class _SelectedRow:
    job: _RowJob
    index: int
    row: DecisionRow


class PhysicalBatchBackend(DecisionBackend):
    """Expose a row executor through the strict single and batch backend seams.

    Each instance owns exactly one immutable model identity. Concurrent callers
    share a bounded, round-robin row queue. The short coalescing window is a
    tunable latency/throughput tradeoff, while ``physical_batch_size`` is the
    maximum number of rows sent to one model forward.
    """

    def __init__(
        self,
        model: ModelDescriptor,
        executor: DecisionRowExecutor,
        *,
        physical_batch_size: int = 8,
        max_pending_rows: int = 4096,
        coalesce_seconds: float = 0.0005,
    ) -> None:
        if physical_batch_size < 1:
            raise ValueError("physical_batch_size must be positive")
        if max_pending_rows < physical_batch_size:
            raise ValueError("max_pending_rows must be at least physical_batch_size")
        if coalesce_seconds < 0:
            raise ValueError("coalesce_seconds must be non-negative")
        self._model = model
        self._executor = executor
        self.physical_batch_size = physical_batch_size
        self.max_pending_rows = max_pending_rows
        self.coalesce_seconds = coalesce_seconds
        self._condition = asyncio.Condition()
        self._pending: deque[_RowJob] = deque()
        self._pending_rows = 0
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    def models(self) -> Sequence[ModelDescriptor]:
        return (self._model,)

    async def ready(self) -> bool:
        if self._closed:
            return False
        executor_ready = await self._executor.ready()
        return executor_ready and not self._closed

    async def infer(self, request: SystemOneRequest) -> BackendResult:
        self._require_model(request.model)
        rows = _request_rows(request)
        results = await self._submit(rows)
        return _backend_result(request.model, rows, results)

    async def infer_batch(
        self, requests: tuple[BackendBatchRequest, ...]
    ) -> tuple[BackendBatchResult, ...]:
        if not requests:
            raise BackendContractError("backend batch must not be empty")
        model = requests[0].request.model
        self._require_model(model)
        if any(item.request.model != model for item in requests):
            raise BackendContractError("backend batch changed model identity")

        # Question-major order lets equal questions across states share physical
        # batches. Coordinates, not row contents, restore the caller's order.
        question_ids = tuple(requests[0].request.questions)
        if any(tuple(item.request.questions) != question_ids for item in requests):
            raise BackendContractError("backend batch questions are not shared")
        rows: list[DecisionRow] = []
        coordinates: list[tuple[int, int]] = []
        for question_index, question_id in enumerate(question_ids):
            for state_index, item in enumerate(requests):
                rows.append(
                    DecisionRow(
                        model=model,
                        state=item.request.state,
                        question_id=question_id,
                        question=item.request.questions[question_id],
                    )
                )
                coordinates.append((state_index, question_index))

        results = await self._submit(tuple(rows))
        restored: list[list[DecisionRowResult | None]] = [
            [None] * len(question_ids) for _ in requests
        ]
        for coordinate, result in zip(coordinates, results, strict=True):
            state_index, question_index = coordinate
            restored[state_index][question_index] = result

        output = []
        for item, state_results in zip(requests, restored, strict=True):
            if any(result is None for result in state_results):
                raise BackendContractError("backend batch restoration is incomplete")
            typed_results = tuple(
                result for result in state_results if result is not None
            )
            state_rows = _request_rows(item.request)
            output.append(
                BackendBatchResult(
                    state_id=item.state_id,
                    result=_backend_result(
                        model,
                        state_rows,
                        typed_results,
                    ),
                )
            )
        return tuple(output)

    async def aclose(self) -> None:
        """Stop admission and fail every caller without exposing partial work.

        Cleanup completes even when the task closing the backend is cancelled.
        The cancellation is re-raised only after queue and worker state are
        consistent, so the lifecycle owner may safely enforce a shutdown
        deadline without stranding inference callers.
        """

        interrupted = await _finish_cleanup(self._close())
        if interrupted:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        """Perform the cancellation-safe state transition used by ``aclose``."""

        async with self._condition:
            if not self._closed:
                self._closed = True
                while self._pending:
                    job = self._pending.popleft()
                    if not job.future.done():
                        job.future.set_exception(
                            BackendUnavailableError("physical batch backend is closed")
                        )
                self._pending_rows = 0
            worker = self._worker
            self._condition.notify_all()
        if worker is not None:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    async def _submit(
        self, rows: tuple[DecisionRow, ...]
    ) -> tuple[DecisionRowResult, ...]:
        if not rows:
            raise BackendContractError("request produced no inference rows")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[DecisionRowResult, ...]] = loop.create_future()
        job = _RowJob(rows=rows, future=future)
        async with self._condition:
            if self._closed:
                raise BackendUnavailableError("physical batch backend is closed")
            if len(rows) > self.max_pending_rows - self._pending_rows:
                raise PhysicalBatchOverloadedError(self._model.name)
            self._pending.append(job)
            self._pending_rows += len(rows)
            if self._worker is None or self._worker.done():
                self._worker = asyncio.create_task(self._run_worker())
            self._condition.notify()
        try:
            return await future
        except asyncio.CancelledError:
            await _finish_cleanup(self._cancel_job(job))
            raise

    async def _cancel_job(self, job: _RowJob) -> None:
        async with self._condition:
            if not job.future.done():
                job.future.cancel()
            removed = 0
            retained: deque[_RowJob] = deque()
            while self._pending:
                candidate = self._pending.popleft()
                if candidate is job:
                    removed += candidate.remaining
                else:
                    retained.append(candidate)
            self._pending = retained
            self._pending_rows -= removed
            self._condition.notify_all()

    async def _run_worker(self) -> None:
        while True:
            selected: list[_SelectedRow] = []
            try:
                async with self._condition:
                    await self._condition.wait_for(
                        lambda: self._closed or self._pending
                    )
                    if self._closed:
                        return
                    selected = self._take_rows(self.physical_batch_size)

                if (
                    len(selected) < self.physical_batch_size
                    and self.coalesce_seconds > 0
                ):
                    await asyncio.sleep(self.coalesce_seconds)
                    async with self._condition:
                        selected.extend(
                            self._take_rows(self.physical_batch_size - len(selected))
                        )

                if not selected:
                    continue
                rows = tuple(item.row for item in selected)
                try:
                    results = await self._executor.predict_rows(rows)
                    _validate_executor_results(rows, results)
                except Exception as error:
                    await self._fail_jobs(selected, error)
                    continue
                await self._complete_rows(selected, results)
            except asyncio.CancelledError as error:
                failure: BaseException = error
                if self._closed:
                    failure = BackendUnavailableError(
                        "physical batch backend is closed"
                    )
                await _finish_cleanup(self._fail_jobs(selected, failure))
                raise

    def _take_rows(self, limit: int) -> list[_SelectedRow]:
        selected: list[_SelectedRow] = []
        while self._pending and len(selected) < limit:
            job = self._pending.popleft()
            if job.future.done():
                self._pending_rows -= job.remaining
                continue
            index = job.cursor
            job.cursor += 1
            self._pending_rows -= 1
            selected.append(_SelectedRow(job=job, index=index, row=job.rows[index]))
            if job.remaining:
                self._pending.append(job)
        return selected

    async def _fail_jobs(
        self, selected: list[_SelectedRow], error: BaseException
    ) -> None:
        failed = {id(item.job): item.job for item in selected}
        async with self._condition:
            retained: deque[_RowJob] = deque()
            while self._pending:
                job = self._pending.popleft()
                if id(job) in failed:
                    self._pending_rows -= job.remaining
                else:
                    retained.append(job)
            self._pending = retained
            for job in failed.values():
                if not job.future.done():
                    job.future.set_exception(error)
            self._condition.notify_all()

    async def _complete_rows(
        self,
        selected: list[_SelectedRow],
        results: tuple[DecisionRowResult, ...],
    ) -> None:
        completed: dict[int, _RowJob] = {}
        for item, result in zip(selected, results, strict=True):
            if item.job.future.done():
                continue
            item.job.results[item.index] = result
            if all(value is not None for value in item.job.results):
                completed[id(item.job)] = item.job
        for job in completed.values():
            if job.future.done():
                continue
            job.future.set_result(
                tuple(value for value in job.results if value is not None)
            )

    def _require_model(self, model: str) -> None:
        if model != self._model.name:
            raise UnknownModelError(model)


async def _finish_cleanup(cleanup: Awaitable[None]) -> bool:
    """Run cleanup to completion while recording repeated cancellation.

    ``asyncio.shield`` prevents cancellation of the cleanup task itself. A loop
    is still required because callers can cancel the waiting task again before
    cleanup acquires the backend condition.
    """

    cleanup_task = asyncio.ensure_future(cleanup)
    interrupted = False
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            interrupted = True
    await cleanup_task
    return interrupted


def _request_rows(request: SystemOneRequest) -> tuple[DecisionRow, ...]:
    return tuple(
        DecisionRow(
            model=request.model,
            state=request.state,
            question_id=question_id,
            question=question,
        )
        for question_id, question in request.questions.items()
    )


def _backend_result(
    model: str,
    rows: tuple[DecisionRow, ...],
    results: tuple[DecisionRowResult, ...],
) -> BackendResult:
    _validate_executor_results(rows, results)
    return BackendResult(
        model=model,
        predictions=tuple(
            BackendPrediction(
                question_id=result.question_id,
                type=result.type,
                probabilities=result.probabilities,
            )
            for result in results
        ),
        input_tokens=sum(result.input_tokens for result in results),
        output_tokens=0,
    )


def _validate_executor_results(
    rows: tuple[DecisionRow, ...],
    results: tuple[DecisionRowResult, ...],
) -> None:
    if not isinstance(results, tuple) or len(results) != len(rows):
        raise BackendContractError("row executor changed the physical batch length")
    for row, result in zip(rows, results, strict=True):
        if not isinstance(result, DecisionRowResult):
            raise BackendContractError("row executor returned an invalid result type")
        if result.question_id != row.question_id or result.type != row.question.type:
            raise BackendContractError("row executor changed question identity")
        if type(result.input_tokens) is not int or result.input_tokens < 0:
            raise BackendContractError("row executor returned invalid token usage")
