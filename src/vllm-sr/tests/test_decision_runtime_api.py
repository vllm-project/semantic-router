"""In-process HTTP conformance tests for the Decision runtime."""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_fake_backend import FakeDecisionBackend  # noqa: E402
from decision_runtime.api import ARTIFACT_RESPONSE_HEADERS, create_app  # noqa: E402
from decision_runtime.backend import (  # noqa: E402
    BackendInputTooLargeError,
    BackendOverloadedError,
    BackendPrediction,
    BackendResult,
    ModelDescriptor,
    UnknownModelError,
)
from decision_runtime.engine import DecisionEngine  # noqa: E402
from decision_runtime.physical_batching import (  # noqa: E402
    DecisionRowResult,
    PhysicalBatchBackend,
    PreparedDecisionRow,
)
from decision_runtime.scheduler import ModelScheduler  # noqa: E402

MODEL = ModelDescriptor(
    name="decision-test",
    description="Test model",
    release_date="2026-09-23",
)


def payload():
    return {
        "model": MODEL.name,
        "state": "Please refund the duplicate charge.",
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Is a refund requested?",
            },
            "category": {
                "type": "choice",
                "instructions": "Classify the request.",
                "criteria": {"billing": None, "technical": None},
            },
            "urgency": {
                "type": "score",
                "instructions": "Rate urgency.",
                "criteria": ["low", "high"],
            },
        },
    }


def batch_payload():
    request = payload()
    return {
        "model": request["model"],
        "states": [
            {"id": "first", "state": request["state"]},
            {"id": "second", "state": "The product crashes on startup."},
        ],
        "questions": request["questions"],
    }


def app_for(backend=None, *, max_concurrency=1, max_queue=8):
    engine = DecisionEngine(backend or FakeDecisionBackend([MODEL]))
    scheduler = ModelScheduler(
        [MODEL.name],
        max_concurrency=max_concurrency,
        max_queue=max_queue,
    )
    return create_app(engine, scheduler=scheduler)


def test_http_e2e_exposes_models_health_readiness_and_strict_response():
    async def scenario():
        async with _client(app_for()) as client:
            assert (await client.get("/health")).json() == {"status": "ok"}
            assert (await client.get("/ready")).json() == {"ready": True}
            assert (await client.get("/v1/models")).json() == {
                "models": [
                    {
                        "name": MODEL.name,
                        "description": MODEL.description,
                        "release_date": MODEL.release_date,
                    }
                ]
            }
            response = await client.post("/v1/systemone", json=payload())
            assert response.status_code == 200
            body = response.json()
            assert set(body) == {"model", "answers", "usage"}
            assert set(body["answers"]["refund"]) == {"type", "noul"}
            assert "confidence" in body["answers"]["category"]
            assert "confidence" in body["answers"]["urgency"]

    asyncio.run(scenario())


def test_create_app_default_admission_allows_http_physical_coalescing():
    class RecordingRowExecutor:
        def __init__(self):
            self.calls = []

        async def ready(self):
            return True

        async def prepare_rows(self, rows):
            return tuple(
                PreparedDecisionRow(row=row, batch_key="test", payload=None)
                for row in rows
            )

        async def predict_rows(self, rows):
            self.calls.append(rows)
            return tuple(
                DecisionRowResult(
                    question_id=item.row.question_id,
                    type=item.row.question.type,
                    probabilities=(0.25, 0.75),
                    input_tokens=len(str(item.row.state)),
                )
                for item in rows
            )

    async def scenario():
        executor = RecordingRowExecutor()
        backend = PhysicalBatchBackend(
            MODEL,
            executor,
            physical_batch_size=8,
            coalesce_seconds=0.05,
        )
        app = create_app(DecisionEngine(backend))
        first_payload = payload()
        first_payload["state"] = "first"
        second_payload = payload()
        second_payload["state"] = "second"

        async def wait_for_selection():
            while backend._worker is None or backend._pending_rows:
                await asyncio.sleep(0)

        try:
            async with _client(app) as client:
                first_response = asyncio.create_task(
                    client.post("/v1/systemone", json=first_payload)
                )
                await asyncio.wait_for(wait_for_selection(), timeout=1)
                responses = await asyncio.gather(
                    first_response,
                    client.post("/v1/systemone", json=second_payload),
                )
        finally:
            await backend.aclose()

        assert [response.status_code for response in responses] == [200, 200]
        assert len(executor.calls) == 1
        assert len(executor.calls[0]) == 6
        assert {item.row.state for item in executor.calls[0]} == {"first", "second"}

    asyncio.run(scenario())


def test_http_validation_uses_documented_422_issue_shape():
    async def scenario():
        async with _client(app_for()) as client:
            missing_model = payload()
            missing_model.pop("model")
            response = await client.post("/v1/systemone", json=missing_model)
            assert response.status_code == 422
            issue = response.json()["detail"][0]
            assert issue["loc"] == ["body", "model"]
            assert set(issue) == {"loc", "msg", "type", "input"}
            assert issue["input"] == missing_model

            legacy = payload()
            legacy["states"] = [{"id": "legacy", "state": "not allowed"}]
            assert (await client.post("/v1/systemone", json=legacy)).status_code == 422

            debug = payload()
            debug["debug"] = True
            assert (await client.post("/v1/systemone", json=debug)).status_code == 422

            missing_instructions = payload()
            missing_instructions["questions"]["refund"].pop("instructions")
            assert (
                await client.post("/v1/systemone", json=missing_instructions)
            ).status_code == 422

            null_instructions = payload()
            null_instructions["questions"]["refund"]["instructions"] = None
            assert (
                await client.post("/v1/systemone", json=null_instructions)
            ).status_code == 422

    asyncio.run(scenario())


def test_http_rejects_missing_instructions_and_single_candidate():
    async def scenario():
        async with _client(app_for()) as client:
            for instructions in (None, ...):
                body = payload()
                body["questions"] = {
                    "category": {
                        "type": "choice",
                        "criteria": {"billing": None, "technical": None},
                    },
                    "urgency": {
                        "type": "score",
                        "criteria": ["Routine", "Urgent"],
                    },
                }
                if instructions is None:
                    for question in body["questions"].values():
                        question["instructions"] = None
                response = await client.post("/v1/systemone", json=body)
                assert response.status_code == 422, response.text

                batch = {
                    "model": body["model"],
                    "states": [{"id": "first", "state": body["state"]}],
                    "questions": body["questions"],
                }
                batch_response = await client.post("/v1/systemone/batches", json=batch)
                assert batch_response.status_code == 422, batch_response.text

            body = payload()
            body["questions"] = {
                "category": {
                    "type": "choice",
                    "instructions": "Choose a category.",
                    "criteria": {"only": None},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate urgency.",
                    "criteria": ["Only level"],
                },
            }
            assert (await client.post("/v1/systemone", json=body)).status_code == 422
            batch = {
                "model": body["model"],
                "states": [{"id": "first", "state": body["state"]}],
                "questions": body["questions"],
            }
            assert (
                await client.post("/v1/systemone/batches", json=batch)
            ).status_code == 422

    asyncio.run(scenario())


def test_batch_http_e2e_is_strict_ordered_and_state_specific():
    async def scenario():
        async with _client(app_for()) as client:
            response = await client.post("/v1/systemone/batches", json=batch_payload())
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"model", "results", "usage"}
        assert [result["id"] for result in body["results"]] == [
            "first",
            "second",
        ]
        assert all(
            set(result) == {"id", "answers", "usage"} for result in body["results"]
        )
        assert body["results"][0]["answers"] != body["results"][1]["answers"]
        assert body["usage"] == {
            "input_tokens": sum(
                result["usage"]["input_tokens"] for result in body["results"]
            ),
            "output_tokens": sum(
                result["usage"]["output_tokens"] for result in body["results"]
            ),
        }

    asyncio.run(scenario())


def test_http_admission_prices_single_and_batch_decision_rows():
    class RecordingScheduler(ModelScheduler):
        def __init__(self):
            super().__init__([MODEL.name], max_active_rows=4096)
            self.costs = []

        async def run(self, model, operation, *, row_cost=1):
            self.costs.append(row_cost)
            return await super().run(model, operation, row_cost=row_cost)

    async def scenario():
        scheduler = RecordingScheduler()
        app = create_app(
            DecisionEngine(FakeDecisionBackend([MODEL])), scheduler=scheduler
        )
        async with _client(app) as client:
            single = await client.post("/v1/systemone", json=payload())
            batch = await client.post("/v1/systemone/batches", json=batch_payload())
            status = (await client.get("/api/status")).json()["scheduler"][0]
        assert (single.status_code, batch.status_code) == (200, 200)
        assert scheduler.costs == [3, 6]
        assert status["max_active_rows"] == 4096
        assert status["active_rows"] == 0

    asyncio.run(scenario())


def test_batch_validation_rejects_duplicate_ids_and_excess_decisions():
    async def scenario():
        async with _client(app_for()) as client:
            duplicate = batch_payload()
            duplicate["states"][1]["id"] = "first"
            response = await client.post("/v1/systemone/batches", json=duplicate)
            assert response.status_code == 422
            assert response.json()["detail"][0]["loc"] == ["body", "states"]

            too_many = batch_payload()
            too_many["states"] = [
                {"id": f"state-{index}", "state": "state"} for index in range(513)
            ]
            response = await client.post("/v1/systemone/batches", json=too_many)
            assert response.status_code == 422
            assert "1024 decisions" in response.json()["detail"][0]["msg"]

    asyncio.run(scenario())


def test_unknown_model_is_a_field_scoped_422():
    async def scenario():
        request = payload()
        request["model"] = "unknown"
        async with _client(app_for()) as client:
            response = await client.post("/v1/systemone", json=request)
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "model"]

    asyncio.run(scenario())


def test_unknown_model_metrics_use_one_bounded_label():
    async def scenario():
        async with _client(app_for()) as client:
            for model in ("arbitrary-one", "arbitrary-two"):
                request = payload()
                request["model"] = model
                assert (
                    await client.post("/v1/systemone", json=request)
                ).status_code == 422

            metrics = (await client.get("/metrics")).text
            assert (
                metrics.count(
                    'decision_runtime_evaluations_total{model="__unknown__",outcome="invalid_model"}'
                )
                == 1
            )
            assert "arbitrary-one" not in metrics
            assert "arbitrary-two" not in metrics

    asyncio.run(scenario())


def test_status_and_metrics_keep_diagnostics_outside_systemone():
    async def scenario():
        async with _client(app_for()) as client:
            await client.post("/v1/systemone", json=payload())
            status = (await client.get("/api/status")).json()
            assert status["contracts"] == [
                "systemone.single.v1",
                "systemone.batches.v1",
            ]
            assert status["confidence"]["typesafe_equivalent"] is False
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert "decision_runtime_evaluations_total" in metrics.text
            assert 'model="decision-test",outcome="success"' in metrics.text

    asyncio.run(scenario())


def test_successful_inference_attests_artifact_without_changing_json_contract():
    provenance = {
        "model": MODEL.name,
        "revision": "a" * 40,
        "manifest_sha256": "b" * 64,
        "content_sha256": "c" * 64,
    }
    app = create_app(
        DecisionEngine(FakeDecisionBackend([MODEL])),
        artifact_provenance=provenance,
    )

    async def scenario():
        async with _client(app) as client:
            assert (await client.get("/api/status")).json()["artifact"] == provenance
            for path, request, expected_fields in (
                ("/v1/systemone", payload(), {"model", "answers", "usage"}),
                (
                    "/v1/systemone/batches",
                    batch_payload(),
                    {"model", "results", "usage"},
                ),
            ):
                response = await client.post(path, json=request)
                assert response.status_code == 200
                assert set(response.json()) == expected_fields
                assert {
                    field: response.headers[header]
                    for field, header in ARTIFACT_RESPONSE_HEADERS.items()
                } == provenance

            rejected = dict(payload(), model="unknown")
            error = await client.post("/v1/systemone", json=rejected)
            assert error.status_code == 422
            assert all(
                header not in error.headers
                for header in ARTIFACT_RESPONSE_HEADERS.values()
            )

    asyncio.run(scenario())


def test_artifact_header_source_requires_one_matching_complete_identity():
    backend = DecisionEngine(FakeDecisionBackend([MODEL]))
    valid = {
        "model": MODEL.name,
        "revision": "a" * 40,
        "manifest_sha256": "b" * 64,
        "content_sha256": "c" * 64,
    }
    for invalid in (
        dict(valid, model="another-model"),
        dict(valid, revision="main"),
        {key: value for key, value in valid.items() if key != "content_sha256"},
    ):
        with pytest.raises(ValueError, match="artifact provenance"):
            create_app(backend, artifact_provenance=invalid)

    for unsafe_name in ("bad\nmodel", "bad,model", "modèle"):
        unsafe_model = ModelDescriptor(unsafe_name, "Test model", "2026-09-23")
        with pytest.raises(ValueError, match="artifact provenance"):
            create_app(
                DecisionEngine(FakeDecisionBackend([unsafe_model])),
                artifact_provenance=dict(valid, model=unsafe_model.name),
            )


def test_openapi_preserves_strict_single_state_schema():
    openapi = app_for().openapi()
    batch_operation = openapi["paths"]["/v1/systemone/batches"]["post"]
    assert "Decision Runtime extension" in batch_operation["description"]
    assert "POST /v1/systemone" in batch_operation["description"]
    schema = openapi["components"]["schemas"]
    batch_request_schema = schema["SystemOneBatchRequest"]
    assert batch_request_schema["required"] == ["model", "states", "questions"]
    assert batch_request_schema["additionalProperties"] is False
    assert set(batch_request_schema["properties"]) == {"model", "states", "questions"}
    batch_response_schema = schema["SystemOneBatchResponse"]
    assert set(batch_response_schema["properties"]) == {"model", "results", "usage"}
    request_schema = schema["SystemOneRequest"]
    assert request_schema["required"] == ["state", "model", "questions"]
    assert request_schema["additionalProperties"] is False
    assert "states" not in request_schema["properties"]
    assert schema["NoulQuestion"]["required"] == ["type", "instructions"]
    assert schema["ChoiceQuestion"]["required"] == [
        "type",
        "instructions",
        "criteria",
    ]
    assert schema["ChoiceQuestion"]["properties"]["criteria"]["minProperties"] == 2
    assert schema["ChoiceQuestion"]["properties"]["criteria"]["maxProperties"] == 255
    assert schema["ScoreQuestion"]["required"] == [
        "type",
        "instructions",
        "criteria",
    ]
    assert schema["ScoreQuestion"]["properties"]["criteria"]["minItems"] == 2
    assert schema["ScoreQuestion"]["properties"]["criteria"]["maxItems"] == 10
    response_schema = schema["SystemOneResponse"]
    assert response_schema["additionalProperties"] is False
    assert set(response_schema["properties"]) == {"model", "answers", "usage"}

    batch_request_schema = schema["SystemOneBatchRequest"]
    assert batch_request_schema["required"] == ["model", "states", "questions"]
    assert batch_request_schema["additionalProperties"] is False
    assert set(batch_request_schema["properties"]) == {
        "model",
        "states",
        "questions",
    }
    assert batch_request_schema["properties"]["states"]["maxItems"] == 1024
    assert batch_request_schema["properties"]["questions"]["maxProperties"] == 1024
    batch_response_schema = schema["SystemOneBatchResponse"]
    assert batch_response_schema["additionalProperties"] is False
    assert set(batch_response_schema["properties"]) == {"model", "results", "usage"}


def test_not_ready_is_503_without_changing_health():
    async def scenario():
        backend = FakeDecisionBackend([MODEL], available=False)
        async with _client(app_for(backend)) as client:
            assert (await client.get("/health")).status_code == 200
            readiness = await client.get("/ready")
            assert readiness.status_code == 503
            assert readiness.json() == {"ready": False}
            assert (
                await client.post("/v1/systemone", json=payload())
            ).status_code == 503

    asyncio.run(scenario())


def test_queue_overload_maps_to_official_529_with_retry_after():
    class BlockingBackend(FakeDecisionBackend):
        def __init__(self):
            super().__init__([MODEL])
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def infer(self, request):
            self.entered.set()
            await self.release.wait()
            return await super().infer(request)

    async def scenario():
        backend = BlockingBackend()
        app = app_for(backend, max_concurrency=1, max_queue=0)
        async with _client(app) as client:
            first = asyncio.create_task(client.post("/v1/systemone", json=payload()))
            await backend.entered.wait()
            second = await client.post("/v1/systemone", json=payload())
            assert second.status_code == 529
            assert second.headers["retry-after"] == "1"
            backend.release.set()
            assert (await first).status_code == 200

    asyncio.run(scenario())


def test_physical_backend_overload_maps_to_official_529_with_retry_after():
    class OverloadedBackend(FakeDecisionBackend):
        async def infer(self, request):
            raise BackendOverloadedError(request.model)

    async def scenario():
        async with _client(app_for(OverloadedBackend([MODEL]))) as client:
            response = await client.post("/v1/systemone", json=payload())
            assert response.status_code == 529
            assert response.headers["retry-after"] == "1"
            metrics = (await client.get("/metrics")).text
            assert 'model="decision-test",outcome="overloaded"' in metrics

    asyncio.run(scenario())


def test_backend_token_limit_maps_to_413_for_single_and_batch_requests():
    class InputTooLargeBackend(FakeDecisionBackend):
        async def infer(self, request):
            raise BackendInputTooLargeError(request.model)

        async def infer_batch(self, requests):
            raise BackendInputTooLargeError(requests[0].request.model)

    async def scenario():
        async with _client(app_for(InputTooLargeBackend([MODEL]))) as client:
            for route, body in (
                ("/v1/systemone", payload()),
                ("/v1/systemone/batches", batch_payload()),
            ):
                response = await client.post(route, json=body)
                assert response.status_code == 413
                assert response.json() == {
                    "detail": "Decision input exceeds the model token limit"
                }
                assert "retry-after" not in response.headers

            metrics = (await client.get("/metrics")).text
            assert 'model="decision-test",outcome="input_too_large"' in metrics

    asyncio.run(scenario())


def test_batch_uses_same_unknown_model_and_overload_mapping():
    class BlockingBatchBackend(FakeDecisionBackend):
        def __init__(self):
            super().__init__([MODEL])
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def infer_batch(self, requests):
            self.entered.set()
            await self.release.wait()
            return await super().infer_batch(requests)

    async def scenario():
        backend = BlockingBatchBackend()
        app = app_for(backend, max_concurrency=1, max_queue=0)
        async with _client(app) as client:
            unknown = batch_payload()
            unknown["model"] = "unknown"
            response = await client.post("/v1/systemone/batches", json=unknown)
            assert response.status_code == 422
            assert response.json()["detail"][0]["loc"] == ["body", "model"]

            first = asyncio.create_task(
                client.post("/v1/systemone/batches", json=batch_payload())
            )
            await backend.entered.wait()
            overloaded = await client.post(
                "/v1/systemone/batches", json=batch_payload()
            )
            assert overloaded.status_code == 529
            assert overloaded.headers["retry-after"] == "1"
            backend.release.set()
            assert (await first).status_code == 200

    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", ["reordered", "missing"])
def test_batch_backend_identity_failures_use_backend_error_boundary(mutation):
    class InvalidBatchBackend(FakeDecisionBackend):
        async def infer_batch(self, requests):
            results = await super().infer_batch(requests)
            if mutation == "reordered":
                return tuple(reversed(results))
            return results[:-1]

    async def scenario():
        async with _client(app_for(InvalidBatchBackend([MODEL]))) as client:
            response = await client.post("/v1/systemone/batches", json=batch_payload())
            assert response.status_code == 500
            assert response.json() == {
                "detail": "Decision backend returned an invalid result"
            }

    asyncio.run(scenario())


def test_batch_backend_model_rejection_is_not_a_client_unknown_model_error():
    class RejectingBatchBackend(FakeDecisionBackend):
        async def infer_batch(self, requests):
            raise UnknownModelError(requests[0].request.model)

    async def scenario():
        async with _client(app_for(RejectingBatchBackend([MODEL]))) as client:
            response = await client.post("/v1/systemone/batches", json=batch_payload())
            assert response.status_code == 500
            assert response.json() == {
                "detail": "Decision backend returned an invalid result"
            }
            metrics = (await client.get("/metrics")).text
            assert 'model="decision-test",outcome="backend_error"' in metrics
            assert 'outcome="invalid_model"' not in metrics

    asyncio.run(scenario())


def test_unexpected_backend_failure_is_not_counted_as_success():
    class CrashingBackend(FakeDecisionBackend):
        async def infer(self, request):
            raise RuntimeError("unexpected adapter failure")

    async def scenario():
        app = app_for(CrashingBackend([MODEL]))
        async with _client(app, raise_app_exceptions=False) as client:
            response = await client.post("/v1/systemone", json=payload())
            assert response.status_code == 500
            metrics = (await client.get("/metrics")).text
            assert 'model="decision-test",outcome="internal_error"' in metrics
            assert 'model="decision-test",outcome="success"' not in metrics

    asyncio.run(scenario())


def test_backend_key_error_is_an_internal_failure_not_an_unknown_model():
    class KeyErrorBackend(FakeDecisionBackend):
        async def infer(self, request):
            raise KeyError("adapter-internal lookup")

    async def scenario():
        app = app_for(KeyErrorBackend([MODEL]))
        async with _client(app, raise_app_exceptions=False) as client:
            response = await client.post("/v1/systemone", json=payload())
            assert response.status_code == 500
            metrics = (await client.get("/metrics")).text
            assert 'model="decision-test",outcome="internal_error"' in metrics
            assert 'outcome="invalid_model"' not in metrics

    asyncio.run(scenario())


def test_backend_model_rejection_is_a_backend_error_not_client_422():
    class RejectingBackend(FakeDecisionBackend):
        async def infer(self, request):
            raise UnknownModelError(request.model)

    async def scenario():
        async with _client(app_for(RejectingBackend([MODEL]))) as client:
            response = await client.post("/v1/systemone", json=payload())
            assert response.status_code == 500
            assert response.json() == {
                "detail": "Decision backend returned an invalid result"
            }
            metrics = (await client.get("/metrics")).text
            assert 'model="decision-test",outcome="backend_error"' in metrics
            assert 'outcome="invalid_model"' not in metrics

    asyncio.run(scenario())


def test_malformed_backend_probabilities_use_the_backend_error_boundary():
    class MalformedBackend(FakeDecisionBackend):
        async def infer(self, request):
            result = await super().infer(request)
            predictions = list(result.predictions)
            first = predictions[0]
            predictions[0] = BackendPrediction(
                question_id=first.question_id,
                type=first.type,
                probabilities=(False, True),
            )
            return BackendResult(
                model=result.model,
                predictions=tuple(predictions),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

    async def scenario():
        async with _client(app_for(MalformedBackend([MODEL]))) as client:
            response = await client.post("/v1/systemone", json=payload())
            assert response.status_code == 500
            assert response.json() == {
                "detail": "Decision backend returned an invalid result"
            }

    asyncio.run(scenario())


def _client(app, *, raise_app_exceptions=True):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=app,
            raise_app_exceptions=raise_app_exceptions,
        ),
        base_url="http://test",
    )
