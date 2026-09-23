"""Cross-request batching, isolation, and cancellation regression tests."""

import asyncio
import hashlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.backend import (  # noqa: E402
    BackendBatchRequest,
    BackendContractError,
    BackendInputTooLargeError,
    BackendUnavailableError,
    ModelDescriptor,
)
from decision_runtime.contracts import SystemOneRequest  # noqa: E402
from decision_runtime.physical_batching import (  # noqa: E402
    DecisionRow,
    DecisionRowResult,
    PhysicalBatchBackend,
    PhysicalBatchOverloadedError,
    PreparedDecisionRow,
)

MODEL = ModelDescriptor(
    name="llm-semantic-router/Decision-1.0-Kai-0.6B",
    description="Test model",
    release_date="2026-09-23",
)


def request(state: str, question_ids=("move",)) -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {
            "model": MODEL.name,
            "state": state,
            "questions": {
                question_id: {
                    "type": "choice",
                    "instructions": "Choose a move.",
                    "criteria": {"left": None, "right": None},
                }
                for question_id in question_ids
            },
        }
    )


def mixed_request(state: str) -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {
            "model": MODEL.name,
            "state": state,
            "questions": {
                "allowed": {
                    "type": "noul",
                    "instructions": "Is this move allowed?",
                },
                "move": {
                    "type": "choice",
                    "instructions": "Choose a move.",
                    "criteria": {"left": None, "right": None},
                },
                "quality": {
                    "type": "score",
                    "instructions": "Rate the move.",
                    "criteria": ["poor", "good"],
                },
            },
        }
    )


class RecordingExecutor:
    def __init__(
        self,
        *,
        gate: asyncio.Event | None = None,
        batch_key=lambda row: "default",
    ) -> None:
        self.gate = gate
        self.batch_key = batch_key
        self.prepare_calls: list[tuple[DecisionRow, ...]] = []
        self.calls: list[tuple[DecisionRow, ...]] = []
        self.prepared_calls: list[tuple[PreparedDecisionRow, ...]] = []

    async def ready(self) -> bool:
        return True

    async def prepare_rows(
        self, rows: tuple[DecisionRow, ...]
    ) -> tuple[PreparedDecisionRow, ...]:
        self.prepare_calls.append(rows)
        return tuple(
            PreparedDecisionRow(
                row=row,
                batch_key=self.batch_key(row),
                payload={"state": row.state},
            )
            for row in rows
        )

    async def predict_rows(
        self, rows: tuple[PreparedDecisionRow, ...]
    ) -> tuple[DecisionRowResult, ...]:
        self.prepared_calls.append(rows)
        raw_rows = tuple(item.row for item in rows)
        self.calls.append(raw_rows)
        if self.gate is not None:
            await self.gate.wait()
        output = []
        for row in raw_rows:
            digest = hashlib.sha256(f"{row.state}|{row.question_id}".encode()).digest()
            left = 0.2 + (digest[0] / 2550)
            output.append(
                DecisionRowResult(
                    question_id=row.question_id,
                    type=row.question.type,
                    probabilities=(left, 1.0 - left),
                    input_tokens=len(str(row.state)),
                )
            )
        return tuple(output)


async def _wait_until_selected(backend: PhysicalBatchBackend) -> None:
    while backend._worker is None or backend._pending_rows:
        await asyncio.sleep(0)


def test_concurrent_requests_share_physical_batch_without_sharing_results():
    async def scenario():
        executor = RecordingExecutor()
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=8,
            coalesce_seconds=0.01,
        )
        try:
            left, right = await asyncio.gather(
                backend.infer(request("left board")),
                backend.infer(request("right board")),
            )
        finally:
            await backend.aclose()

        assert len(executor.calls) == 1
        assert [row.state for row in executor.calls[0]] == [
            "left board",
            "right board",
        ]
        assert left.predictions != right.predictions

    asyncio.run(scenario())


def test_large_jobs_are_round_robin_and_never_exceed_physical_batch_size():
    async def scenario():
        executor = RecordingExecutor()
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=3,
            coalesce_seconds=0.01,
        )
        try:
            first, second = await asyncio.gather(
                backend.infer(request("first", ("a", "b", "c", "d"))),
                backend.infer(request("second", ("x", "y"))),
            )
        finally:
            await backend.aclose()
        assert len(first.predictions) == 4
        assert len(second.predictions) == 2
        assert all(len(call) <= 3 for call in executor.calls)
        assert [row.state for row in executor.calls[0]] == [
            "first",
            "second",
            "first",
        ]

    asyncio.run(scenario())


def test_incompatible_batch_keys_are_partitioned_without_losing_fairness():
    async def scenario():
        executor = RecordingExecutor(batch_key=lambda row: row.question.type)
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=8,
            coalesce_seconds=0.01,
        )
        try:
            left, right = await asyncio.gather(
                backend.infer(mixed_request("left")),
                backend.infer(mixed_request("right")),
            )
        finally:
            await backend.aclose()

        assert len(left.predictions) == len(right.predictions) == 3
        assert len(executor.prepared_calls) == 3
        assert {
            tuple(item.batch_key for item in call) for call in executor.prepared_calls
        } == {("noul", "noul"), ("choice", "choice"), ("score", "score")}
        assert all(
            [item.row.state for item in call] == ["left", "right"]
            for call in executor.prepared_calls
        )

    asyncio.run(scenario())


def test_preparation_failure_is_atomic_and_does_not_poison_concurrent_request():
    class SizeCheckingExecutor(RecordingExecutor):
        async def prepare_rows(self, rows):
            self.prepare_calls.append(rows)
            if any(row.state == "too-large" for row in rows):
                raise BackendInputTooLargeError("profile token limit exceeded")
            return tuple(
                PreparedDecisionRow(row=row, batch_key="valid", payload=None)
                for row in rows
            )

    async def scenario():
        executor = SizeCheckingExecutor()
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=8,
            coalesce_seconds=0.01,
        )
        try:
            failed, valid = await asyncio.gather(
                backend.infer(request("too-large", ("a", "b"))),
                backend.infer(request("valid", ("x", "y"))),
                return_exceptions=True,
            )
        finally:
            await backend.aclose()

        assert isinstance(failed, BackendInputTooLargeError)
        assert not isinstance(valid, BaseException)
        assert [[row.state for row in call] for call in executor.calls] == [
            ["valid", "valid"]
        ]
        assert backend._pending_rows == 0

    asyncio.run(scenario())


def test_invalid_preparation_result_fails_only_that_logical_request():
    class BrokenPreparationExecutor(RecordingExecutor):
        async def prepare_rows(self, rows):
            if rows[0].state == "broken":
                return ()
            return await super().prepare_rows(rows)

    async def scenario():
        executor = BrokenPreparationExecutor()
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=8,
            coalesce_seconds=0.01,
        )
        try:
            broken, valid = await asyncio.gather(
                backend.infer(request("broken")),
                backend.infer(request("valid")),
                return_exceptions=True,
            )
        finally:
            await backend.aclose()

        assert isinstance(broken, BackendContractError)
        assert "prepared row count" in str(broken)
        assert not isinstance(valid, BaseException)
        assert [[row.state for row in call] for call in executor.calls] == [["valid"]]

    asyncio.run(scenario())


def test_batch_failure_is_isolated_to_jobs_in_that_compatible_forward():
    class SelectiveFailureExecutor(RecordingExecutor):
        def __init__(self):
            super().__init__(batch_key=lambda row: str(row.state).split("-")[0])

        async def predict_rows(self, rows):
            assert len({row.batch_key for row in rows}) == 1
            if rows[0].batch_key == "bad":
                self.prepared_calls.append(rows)
                self.calls.append(tuple(item.row for item in rows))
                raise RuntimeError("bad physical batch")
            return await super().predict_rows(rows)

    async def scenario():
        executor = SelectiveFailureExecutor()
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=8,
            coalesce_seconds=0.01,
        )
        try:
            bad_one, bad_two, good_one, good_two = await asyncio.gather(
                backend.infer(request("bad-one")),
                backend.infer(request("bad-two")),
                backend.infer(request("good-one")),
                backend.infer(request("good-two")),
                return_exceptions=True,
            )
        finally:
            await backend.aclose()

        assert isinstance(bad_one, RuntimeError)
        assert isinstance(bad_two, RuntimeError)
        assert not isinstance(good_one, BaseException)
        assert not isinstance(good_two, BaseException)
        assert [
            {item.batch_key for item in call} for call in executor.prepared_calls
        ] == [{"bad"}, {"good"}]

    asyncio.run(scenario())


def test_batch_restores_state_and_question_order_after_question_major_execution():
    async def scenario():
        executor = RecordingExecutor()
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=8,
            coalesce_seconds=0,
        )
        payloads = (
            BackendBatchRequest("left", request("left", ("a", "b"))),
            BackendBatchRequest("right", request("right", ("a", "b"))),
        )
        try:
            results = await backend.infer_batch(payloads)
        finally:
            await backend.aclose()
        assert [(row.state, row.question_id) for row in executor.calls[0]] == [
            ("left", "a"),
            ("right", "a"),
            ("left", "b"),
            ("right", "b"),
        ]
        assert [item.state_id for item in results] == ["left", "right"]
        assert [
            prediction.question_id for prediction in results[0].result.predictions
        ] == [
            "a",
            "b",
        ]
        assert [
            prediction.question_id for prediction in results[1].result.predictions
        ] == [
            "a",
            "b",
        ]

    asyncio.run(scenario())


def test_admission_is_atomic_when_row_queue_is_full():
    async def scenario():
        gate = asyncio.Event()
        executor = RecordingExecutor(gate=gate)
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=2,
            max_pending_rows=2,
            coalesce_seconds=0,
        )
        first = asyncio.create_task(backend.infer(request("first", ("a", "b"))))
        while not executor.calls:
            await asyncio.sleep(0)
        queued = asyncio.create_task(backend.infer(request("queued", ("c", "d"))))
        await asyncio.sleep(0)
        with pytest.raises(PhysicalBatchOverloadedError):
            await backend.infer(request("rejected"))
        gate.set()
        await first
        await queued
        await backend.aclose()

    asyncio.run(scenario())


def test_cancelled_job_releases_pending_capacity():
    async def scenario():
        gate = asyncio.Event()
        executor = RecordingExecutor(gate=gate)
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=1,
            max_pending_rows=2,
            coalesce_seconds=0,
        )
        active = asyncio.create_task(backend.infer(request("active")))
        while not executor.calls:
            await asyncio.sleep(0)
        cancelled = asyncio.create_task(backend.infer(request("cancelled", ("a", "b"))))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        replacement = asyncio.create_task(
            backend.infer(request("replacement", ("x", "y")))
        )
        gate.set()
        await active
        await replacement
        await backend.aclose()

    asyncio.run(scenario())


def test_cancellation_during_preparation_never_admits_partial_work():
    class BlockingPreparationExecutor(RecordingExecutor):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_rows(self, rows):
            self.entered.set()
            await self.release.wait()
            return await super().prepare_rows(rows)

    async def scenario():
        executor = BlockingPreparationExecutor()
        backend = PhysicalBatchBackend(MODEL, executor)
        caller = asyncio.create_task(backend.infer(request("cancelled", ("a", "b"))))
        await executor.entered.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert backend._pending_rows == 0
        assert backend._worker is None
        assert executor.calls == []
        await backend.aclose()

    asyncio.run(scenario())


def test_shutdown_during_preparation_rejects_post_prepare_admission():
    class BlockingPreparationExecutor(RecordingExecutor):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def prepare_rows(self, rows):
            self.entered.set()
            await self.release.wait()
            return await super().prepare_rows(rows)

    async def scenario():
        executor = BlockingPreparationExecutor()
        backend = PhysicalBatchBackend(MODEL, executor)
        caller = asyncio.create_task(backend.infer(request("closing")))
        await executor.entered.wait()
        await backend.aclose()
        executor.release.set()
        with pytest.raises(BackendUnavailableError, match="backend is closed"):
            await caller
        assert backend._pending_rows == 0
        assert executor.calls == []

    asyncio.run(scenario())


def test_repeated_cancellation_finishes_job_cleanup_and_releases_capacity():
    async def scenario():
        gate = asyncio.Event()
        executor = RecordingExecutor(gate=gate)
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=1,
            max_pending_rows=2,
            coalesce_seconds=0,
        )
        active = asyncio.create_task(backend.infer(request("active")))
        while not executor.calls:
            await asyncio.sleep(0)

        cancelled = asyncio.create_task(backend.infer(request("cancelled", ("a", "b"))))
        while backend._pending_rows != 2:
            await asyncio.sleep(0)

        await backend._condition.acquire()
        try:
            cancelled.cancel()
            await asyncio.sleep(0)
            cancelled.cancel()
            await asyncio.sleep(0)
            assert not cancelled.done()
        finally:
            backend._condition.release()

        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert backend._pending_rows == 0

        replacement = asyncio.create_task(
            backend.infer(request("replacement", ("x", "y")))
        )
        gate.set()
        await active
        await replacement
        await backend.aclose()

    asyncio.run(scenario())


def test_aclose_during_coalescing_fails_every_selected_caller():
    async def scenario():
        executor = RecordingExecutor()
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=3,
            coalesce_seconds=60,
        )
        await backend._condition.acquire()
        callers = [
            asyncio.create_task(backend.infer(request("selected-first"))),
            asyncio.create_task(backend.infer(request("selected-second"))),
        ]
        try:
            await asyncio.sleep(0)
        finally:
            backend._condition.release()
        await asyncio.wait_for(_wait_until_selected(backend), timeout=1)
        assert executor.calls == []

        await backend.aclose()
        for caller in callers:
            with pytest.raises(BackendUnavailableError, match="backend is closed"):
                await asyncio.wait_for(caller, timeout=1)

    asyncio.run(scenario())


def test_worker_repeated_cancellation_finishes_selected_cleanup():
    async def scenario():
        backend = PhysicalBatchBackend(
            MODEL,
            RecordingExecutor(),
            physical_batch_size=2,
            coalesce_seconds=60,
        )
        caller = asyncio.create_task(backend.infer(request("selected")))
        await asyncio.wait_for(_wait_until_selected(backend), timeout=1)
        worker = backend._worker
        assert worker is not None

        await backend._condition.acquire()
        try:
            worker.cancel()
            await asyncio.sleep(0)
            worker.cancel()
            await asyncio.sleep(0)
            assert not worker.done()
        finally:
            backend._condition.release()

        with pytest.raises(asyncio.CancelledError):
            await worker
        with pytest.raises(asyncio.CancelledError):
            await caller
        await backend.aclose()

    asyncio.run(scenario())


def test_aclose_finishes_under_repeated_cancellation():
    async def scenario():
        backend = PhysicalBatchBackend(
            MODEL,
            RecordingExecutor(),
            physical_batch_size=2,
            coalesce_seconds=60,
        )
        caller = asyncio.create_task(backend.infer(request("selected")))
        await asyncio.wait_for(_wait_until_selected(backend), timeout=1)
        await backend._condition.acquire()
        close_task = asyncio.create_task(backend.aclose())
        try:
            await asyncio.sleep(0)
            close_task.cancel()
            await asyncio.sleep(0)
            close_task.cancel()
            await asyncio.sleep(0)
            assert not close_task.done()
        finally:
            backend._condition.release()

        with pytest.raises(asyncio.CancelledError):
            await close_task
        with pytest.raises(BackendUnavailableError, match="backend is closed"):
            await asyncio.wait_for(caller, timeout=1)
        assert await backend.ready() is False
        with pytest.raises(BackendUnavailableError, match="backend is closed"):
            await backend.infer(request("after-close"))

    asyncio.run(scenario())


def test_ready_rechecks_closed_after_executor_readiness_returns():
    class DelayedReadyExecutor(RecordingExecutor):
        def __init__(self):
            super().__init__()
            self.ready_started = asyncio.Event()
            self.ready_release = asyncio.Event()

        async def ready(self):
            self.ready_started.set()
            await self.ready_release.wait()
            return True

    async def scenario():
        executor = DelayedReadyExecutor()
        backend = PhysicalBatchBackend(MODEL, executor)
        ready_task = asyncio.create_task(backend.ready())
        await executor.ready_started.wait()
        await backend.aclose()
        executor.ready_release.set()
        assert await ready_task is False

    asyncio.run(scenario())


def test_executor_identity_error_fails_whole_job_without_partial_response():
    class BrokenExecutor(RecordingExecutor):
        async def predict_rows(self, rows):
            results = list(await super().predict_rows(rows))
            results[-1] = DecisionRowResult(
                question_id="wrong",
                type=results[-1].type,
                probabilities=results[-1].probabilities,
                input_tokens=results[-1].input_tokens,
            )
            return tuple(results)

    async def scenario():
        backend = PhysicalBatchBackend(
            MODEL,
            BrokenExecutor(),
            physical_batch_size=2,
            coalesce_seconds=0,
        )
        try:
            with pytest.raises(Exception, match="changed question identity"):
                await backend.infer(request("state", ("a", "b")))
        finally:
            await backend.aclose()

    asyncio.run(scenario())
