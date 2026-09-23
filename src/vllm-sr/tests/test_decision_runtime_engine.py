"""Backend, engine, and scheduler behavior without an HTTP framework."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.backend import (  # noqa: E402
    BackendBatchRequest,
    BackendBatchResult,
    BackendContractError,
    BackendPrediction,
    BackendResult,
    ModelDescriptor,
    UnknownModelError,
)
from decision_runtime.contracts import (  # noqa: E402
    SystemOneBatchRequest,
    SystemOneRequest,
)
from decision_runtime.engine import DecisionEngine  # noqa: E402
from decision_runtime.fake_backend import FakeDecisionBackend  # noqa: E402
from decision_runtime.scheduler import (  # noqa: E402
    ModelScheduler,
    SchedulerOverloadedError,
)

MODEL = ModelDescriptor(
    name="decision-test",
    description="Test model",
    release_date="2026-09-23",
)


def test_package_root_does_not_import_server_or_inference_frameworks():
    environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    script = """
import sys
import decision_runtime
for forbidden in ("fastapi", "uvicorn", "torch"):
    assert forbidden not in sys.modules, forbidden
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )


def test_console_entrypoint_reports_the_optional_extra_before_server_import():
    environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    script = r"""
import importlib

real_import = importlib.import_module

def blocked(name, package=None):
    if name in {"fastapi", "uvicorn"}:
        raise ImportError(name)
    return real_import(name, package)

importlib.import_module = blocked
from decision_runtime.entrypoint import main

try:
    main()
except SystemExit as error:
    assert "vllm-sr[decision-runtime]" in str(error)
else:
    raise AssertionError("entrypoint unexpectedly started")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )


def request():
    return SystemOneRequest.model_validate(
        {
            "model": MODEL.name,
            "state": "A production system is unavailable.",
            "questions": {
                "outage": {
                    "type": "noul",
                    "instructions": "Is this an outage?",
                },
                "owner": {
                    "type": "choice",
                    "instructions": "Who should own this?",
                    "criteria": {"platform": None, "billing": None},
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate severity.",
                    "criteria": ["low", "medium", "high"],
                },
            },
        }
    )


def batch_request():
    payload = request()
    return SystemOneBatchRequest(
        model=payload.model,
        states=[
            {"id": "first", "state": "A production system is unavailable."},
            {"id": "second", "state": "The customer needs a billing refund."},
        ],
        questions=payload.questions,
    )


def test_fake_backend_is_deterministic_and_engine_emits_strict_response():
    async def scenario():
        engine = DecisionEngine(FakeDecisionBackend([MODEL]))
        first = await engine.evaluate(request())
        second = await engine.evaluate(request())
        assert first == second
        assert set(first.model_dump()) == {"model", "answers", "usage"}
        assert set(first.answers["outage"].model_dump()) == {"type", "noul"}
        assert set(first.answers["owner"].model_dump()) == {
            "type",
            "choice",
            "confidence",
            "probabilities",
        }

    asyncio.run(scenario())


def test_engine_rejects_unknown_model_before_backend_work():
    async def scenario():
        engine = DecisionEngine(FakeDecisionBackend([MODEL]))
        payload = request().model_copy(update={"model": "absent"})
        with pytest.raises(UnknownModelError):
            await engine.evaluate(payload)

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["", "   ", " padded", "padded ", 17])
def test_engine_rejects_unreachable_backend_model_names(name):
    descriptor = ModelDescriptor(
        name=name,
        description="Invalid test model",
        release_date="2026-09-23",
    )

    with pytest.raises(ValueError, match="non-empty, trimmed strings"):
        DecisionEngine(FakeDecisionBackend([descriptor]))


def test_engine_rejects_malformed_backend_distribution():
    class InvalidBackend:
        def models(self):
            return [MODEL]

        async def ready(self):
            return True

        async def infer(self, payload):
            return BackendResult(
                model=MODEL.name,
                predictions=(
                    BackendPrediction(
                        question_id="outage",
                        type="noul",
                        probabilities=(0.8, 0.8),
                    ),
                ),
                input_tokens=1,
                output_tokens=1,
            )

    async def scenario():
        payload = request().model_copy(
            update={"questions": {"outage": request().questions["outage"]}}
        )
        with pytest.raises(BackendContractError):
            await DecisionEngine(InvalidBackend()).evaluate(payload)

    asyncio.run(scenario())


def test_engine_rejects_cross_model_backend_result():
    other_model = ModelDescriptor(
        name="decision-other",
        description="Other test model",
        release_date="2026-09-23",
    )

    class CrossModelBackend(FakeDecisionBackend):
        async def infer(self, payload):
            result = await super().infer(payload)
            return BackendResult(
                model=other_model.name,
                predictions=result.predictions,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

    async def scenario():
        engine = DecisionEngine(CrossModelBackend([MODEL, other_model]))
        with pytest.raises(BackendContractError, match="changed the requested model"):
            await engine.evaluate(request())

    asyncio.run(scenario())


def test_engine_treats_advertised_model_rejection_as_backend_contract_failure():
    class RejectingBackend(FakeDecisionBackend):
        async def infer(self, payload):
            raise UnknownModelError(payload.model)

    async def scenario():
        with pytest.raises(BackendContractError, match="advertised inventory"):
            await DecisionEngine(RejectingBackend([MODEL])).evaluate(request())

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "probabilities",
    [
        (False, True),
        ("0.25", "0.75"),
        [0.25, 0.75],
    ],
)
def test_engine_normalizes_malformed_probability_types_to_contract_error(
    probabilities,
):
    class InvalidProbabilityBackend(FakeDecisionBackend):
        async def infer(self, payload):
            result = await super().infer(payload)
            first = result.predictions[0]
            return BackendResult(
                model=result.model,
                predictions=(
                    BackendPrediction(
                        question_id=first.question_id,
                        type=first.type,
                        probabilities=probabilities,
                    ),
                ),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

    async def scenario():
        payload = request().model_copy(
            update={"questions": {"outage": request().questions["outage"]}}
        )
        with pytest.raises(BackendContractError, match="probabilities"):
            await DecisionEngine(InvalidProbabilityBackend([MODEL])).evaluate(payload)

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_usage", [True, -1, 1.5])
def test_engine_rejects_invalid_backend_token_usage(invalid_usage):
    class InvalidUsageBackend(FakeDecisionBackend):
        async def infer(self, payload):
            result = await super().infer(payload)
            return BackendResult(
                model=result.model,
                predictions=result.predictions,
                input_tokens=invalid_usage,
                output_tokens=result.output_tokens,
            )

    async def scenario():
        with pytest.raises(BackendContractError, match="token usage"):
            await DecisionEngine(InvalidUsageBackend([MODEL])).evaluate(request())

    asyncio.run(scenario())


def test_batch_engine_uses_one_seam_and_keeps_distinct_state_results():
    class ObservedBackend(FakeDecisionBackend):
        def __init__(self):
            super().__init__([MODEL])
            self.single_calls = 0
            self.batch_calls = 0
            self.received = ()

        async def infer(self, payload):
            self.single_calls += 1
            return await super().infer(payload)

        async def infer_batch(self, requests):
            self.batch_calls += 1
            self.received = requests
            return await super().infer_batch(requests)

    async def scenario():
        backend = ObservedBackend()
        response = await DecisionEngine(backend).evaluate_batch(batch_request())

        assert backend.batch_calls == 1
        assert backend.single_calls == 0
        assert all(isinstance(item, BackendBatchRequest) for item in backend.received)
        assert [result.id for result in response.results] == ["first", "second"]
        assert response.results[0].answers != response.results[1].answers
        assert response.usage.input_tokens == sum(
            result.usage.input_tokens for result in response.results
        )
        assert response.usage.output_tokens == sum(
            result.usage.output_tokens for result in response.results
        )

    asyncio.run(scenario())


def test_batch_engine_never_falls_back_to_repeated_single_inference():
    class SingleOnlyBackend:
        def __init__(self):
            self.delegate = FakeDecisionBackend([MODEL])
            self.single_calls = 0

        def models(self):
            return self.delegate.models()

        async def ready(self):
            return await self.delegate.ready()

        async def infer(self, payload):
            self.single_calls += 1
            return await self.delegate.infer(payload)

    async def scenario():
        backend = SingleOnlyBackend()
        with pytest.raises(BackendContractError, match="batch inference"):
            await DecisionEngine(backend).evaluate_batch(batch_request())
        assert backend.single_calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["reordered", "missing"])
def test_batch_engine_rejects_backend_state_reordering_or_omission(mutation):
    class InvalidBatchBackend(FakeDecisionBackend):
        async def infer_batch(self, requests):
            results = await super().infer_batch(requests)
            if mutation == "reordered":
                return tuple(reversed(results))
            return results[:-1]

    async def scenario():
        with pytest.raises(BackendContractError, match="identity or order"):
            await DecisionEngine(InvalidBatchBackend([MODEL])).evaluate_batch(
                batch_request()
            )

    asyncio.run(scenario())


def test_batch_engine_rejects_non_batch_result_wrappers_atomically():
    class InvalidBatchBackend(FakeDecisionBackend):
        async def infer_batch(self, requests):
            valid = await super().infer_batch(requests)
            return (
                valid[0],
                BackendBatchResult(
                    state_id=valid[1].state_id,
                    result="not-a-backend-result",
                ),
            )

    async def scenario():
        with pytest.raises(BackendContractError, match="BackendResult"):
            await DecisionEngine(InvalidBatchBackend([MODEL])).evaluate_batch(
                batch_request()
            )

    asyncio.run(scenario())


def test_scheduler_bounds_running_and_queued_work_per_model():
    async def scenario():
        scheduler = ModelScheduler([MODEL.name], max_concurrency=1, max_queue=1)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocking():
            entered.set()
            await release.wait()
            return "done"

        first = asyncio.create_task(scheduler.run(MODEL.name, blocking))
        await entered.wait()
        second = asyncio.create_task(scheduler.run(MODEL.name, blocking))
        await asyncio.sleep(0)
        snapshot = (await scheduler.snapshots())[0]
        assert (snapshot.running, snapshot.queued) == (1, 1)
        with pytest.raises(SchedulerOverloadedError):
            await scheduler.run(MODEL.name, blocking)
        release.set()
        assert await first == "done"
        assert await second == "done"

    asyncio.run(scenario())


def test_scheduler_releases_running_and_queued_cancellations():
    async def scenario():
        scheduler = ModelScheduler([MODEL.name], max_concurrency=1, max_queue=1)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocking():
            entered.set()
            await release.wait()

        running = asyncio.create_task(scheduler.run(MODEL.name, blocking))
        await entered.wait()
        queued = asyncio.create_task(scheduler.run(MODEL.name, blocking))
        await asyncio.sleep(0)

        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        snapshot = (await scheduler.snapshots())[0]
        assert (snapshot.running, snapshot.queued) == (1, 0)

        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        snapshot = (await scheduler.snapshots())[0]
        assert (snapshot.running, snapshot.queued) == (0, 0)

    asyncio.run(scenario())


def test_scheduler_double_cancellation_cannot_leak_a_running_slot():
    async def scenario():
        scheduler = ModelScheduler([MODEL.name], max_concurrency=1, max_queue=0)
        state = scheduler._states[MODEL.name]
        entered = asyncio.Event()
        operation_release = asyncio.Event()

        async def blocking():
            entered.set()
            await operation_release.wait()

        task = asyncio.create_task(scheduler.run(MODEL.name, blocking))
        await entered.wait()
        await state.condition.acquire()
        try:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            state.condition.release()

        with pytest.raises(asyncio.CancelledError):
            await task
        snapshot = (await scheduler.snapshots())[0]
        assert (snapshot.running, snapshot.queued) == (0, 0)
        assert await scheduler.run(MODEL.name, _immediate_result) == "available"

    async def _immediate_result():
        return "available"

    asyncio.run(scenario())
