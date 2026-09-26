"""FastAPI surface for a backend-neutral Decision runtime."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, TypeVar

from fastapi import Body, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

from .backend import (
    BackendContractError,
    BackendInputTooLargeError,
    BackendOverloadedError,
    BackendUnavailableError,
    ModelDescriptor,
    UnknownModelError,
)
from .contracts import (
    SystemOneBatchRequest,
    SystemOneBatchResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from .engine import DecisionEngine
from .metrics import RuntimeMetrics
from .request_limits import (
    BoundedJSONBodyMiddleware,
    exceeds_expanded_input_limit,
)
from .scheduler import ModelScheduler, SchedulerOverloadedError

EvaluationResponseT = TypeVar(
    "EvaluationResponseT", SystemOneResponse, SystemOneBatchResponse
)
ARTIFACT_RESPONSE_HEADERS = {
    "model": "X-Decision-Artifact-Model",
    "revision": "X-Decision-Artifact-Revision",
    "manifest_sha256": "X-Decision-Artifact-Manifest-Sha256",
    "content_sha256": "X-Decision-Artifact-Content-Sha256",
}
MAX_ARTIFACT_MODEL_ID_LENGTH = 128
ARTIFACT_MODEL_ASCII_MIN = 33
ARTIFACT_MODEL_ASCII_MAX = 126


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelMetadata(_ResponseModel):
    name: str
    description: str
    release_date: str


class ModelMetadataList(_ResponseModel):
    models: list[ModelMetadata]


def create_app(
    engine: DecisionEngine,
    *,
    scheduler: ModelScheduler | None = None,
    metrics: RuntimeMetrics | None = None,
    artifact_provenance: Mapping[str, str] | None = None,
) -> FastAPI:
    """Build the HTTP API without selecting or owning a model framework.

    The caller owns backend lifecycle. In particular, a runtime factory that
    supplies a ``PhysicalBatchBackend`` must await its ``aclose`` method during
    application shutdown.
    """

    runtime_metrics = metrics or RuntimeMetrics()
    model_names = tuple(model.name for model in engine.models)
    served_model_names = frozenset(model_names)
    runtime_scheduler = scheduler or ModelScheduler(model_names)
    attested_artifact = (
        dict(artifact_provenance) if artifact_provenance is not None else None
    )
    if attested_artifact is not None and (
        len(model_names) != 1
        or set(attested_artifact) != set(ARTIFACT_RESPONSE_HEADERS)
        or attested_artifact["model"] != model_names[0]
        or not isinstance(attested_artifact["model"], str)
        or not 1 <= len(attested_artifact["model"]) <= MAX_ARTIFACT_MODEL_ID_LENGTH
        or any(
            not ARTIFACT_MODEL_ASCII_MIN <= ord(character) <= ARTIFACT_MODEL_ASCII_MAX
            or character == ","
            for character in attested_artifact["model"]
        )
        or any(
            not isinstance(attested_artifact[key], str)
            or len(attested_artifact[key]) != length
            or any(
                character not in "0123456789abcdef"
                for character in attested_artifact[key]
            )
            for key, length in (
                ("revision", 40),
                ("manifest_sha256", 64),
                ("content_sha256", 64),
            )
        )
    ):
        raise ValueError("Decision artifact provenance is invalid")
    app = FastAPI(
        title="vLLM Semantic Router Decision Runtime",
        version="0.1.0",
    )

    @app.middleware("http")
    async def observe_http(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            route_path = getattr(route, "path", "__unmatched__")
            runtime_metrics.record_http(
                route_path, status, time.perf_counter() - started
            )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        detail = []
        for error in exc.errors():
            issue = {
                "loc": list(error["loc"]),
                "msg": error["msg"],
                "type": error["type"],
            }
            for optional_field in ("input", "ctx"):
                if optional_field in error and _is_json_compatible(
                    error[optional_field]
                ):
                    issue[optional_field] = error[optional_field]
            detail.append(issue)
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        if await engine.ready():
            return {"ready": True}
        return JSONResponse(
            status_code=503,
            content={"ready": False},
        )

    @app.get("/v1/models", response_model=ModelMetadataList)
    async def models():
        return ModelMetadataList(
            models=[_model_metadata(model) for model in engine.models]
        )

    async def run_evaluation(
        model: str,
        operation: Callable[[], Awaitable[EvaluationResponseT]],
        http_response: Response,
        *,
        row_cost: int,
    ) -> EvaluationResponseT | JSONResponse:
        started = time.perf_counter()
        outcome = "internal_error"
        try:
            if model not in served_model_names:
                raise UnknownModelError(model)
            response = await runtime_scheduler.run(
                model,
                operation,
                row_cost=row_cost,
            )
            if attested_artifact is not None:
                # These internal transport headers bind this successful result
                # to the resident artifact without changing the public JSON.
                for field, header in ARTIFACT_RESPONSE_HEADERS.items():
                    http_response.headers[header] = attested_artifact[field]
            outcome = "success"
            return response
        except UnknownModelError:
            outcome = "invalid_model"
            return JSONResponse(
                status_code=422,
                content={
                    "detail": [
                        {
                            "loc": ["body", "model"],
                            "msg": "Model is not served by this runtime",
                            "type": "value_error.model",
                        }
                    ]
                },
            )
        except (SchedulerOverloadedError, BackendOverloadedError):
            outcome = "overloaded"
            return JSONResponse(
                status_code=529,
                content={"detail": "Decision runtime is temporarily overloaded"},
                headers={"Retry-After": "1"},
            )
        except BackendInputTooLargeError:
            outcome = "input_too_large"
            return JSONResponse(
                status_code=413,
                content={"detail": "Decision input exceeds the model token limit"},
            )
        except BackendUnavailableError:
            outcome = "unavailable"
            return JSONResponse(
                status_code=503,
                content={"detail": "Decision backend is unavailable"},
                headers={"Retry-After": "1"},
            )
        except BackendContractError:
            outcome = "backend_error"
            return JSONResponse(
                status_code=500,
                content={"detail": "Decision backend returned an invalid result"},
            )
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            runtime_metrics.record_evaluation(
                (model if model in served_model_names else "__unknown__"),
                outcome,
                time.perf_counter() - started,
            )

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def system_one(
        payload: Annotated[SystemOneRequest, Body(...)],
        response: Response,
    ) -> SystemOneResponse | JSONResponse:
        if exceeds_expanded_input_limit(payload):
            return JSONResponse(
                status_code=413,
                content={"detail": "Expanded Decision input exceeds the runtime limit"},
            )
        return await run_evaluation(
            payload.model,
            lambda: engine.evaluate(payload),
            response,
            row_cost=len(payload.questions),
        )

    @app.post(
        "/v1/systemone/batches",
        response_model=SystemOneBatchResponse,
        summary="Evaluate many states with shared SystemOne questions",
        description=(
            "Decision Runtime extension to POST /v1/systemone. Send the same "
            "SystemOne question map for every identified state; each result "
            "contains the same typed answers as a single-state response. "
            "The states/results envelope is specific to Decision Runtime, "
            "not an official SystemOne SDK operation."
        ),
    )
    async def system_one_batch(
        payload: Annotated[SystemOneBatchRequest, Body(...)],
        response: Response,
    ) -> SystemOneBatchResponse | JSONResponse:
        if exceeds_expanded_input_limit(payload):
            return JSONResponse(
                status_code=413,
                content={"detail": "Expanded Decision input exceeds the runtime limit"},
            )
        return await run_evaluation(
            payload.model,
            lambda: engine.evaluate_batch(payload),
            response,
            row_cost=len(payload.states) * len(payload.questions),
        )

    @app.get("/api/status")
    async def status():
        snapshots = await runtime_scheduler.snapshots()
        is_ready = await engine.ready()
        result = {
            "status": "ready" if is_ready else "not_ready",
            "contracts": ["systemone.single.v1", "systemone.batches.v1"],
            "confidence": {
                "name": "decision_type_aware_v1",
                "typesafe_equivalent": False,
            },
            "models": [model.name for model in engine.models],
            "scheduler": [
                {
                    "model": snapshot.model,
                    "running": snapshot.running,
                    "queued": snapshot.queued,
                    "active_rows": snapshot.active_rows,
                    "max_concurrency": snapshot.max_concurrency,
                    "max_queue": snapshot.max_queue,
                    "max_active_rows": snapshot.max_active_rows,
                }
                for snapshot in snapshots
            ],
        }
        if attested_artifact is not None:
            result["artifact"] = attested_artifact
        return result

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return PlainTextResponse(
            runtime_metrics.render(await runtime_scheduler.snapshots()),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # Keep transport admission outside BaseHTTPMiddleware so a peer disconnect
    # can stop the ASGI exchange without forcing an artificial response.
    app.add_middleware(BoundedJSONBodyMiddleware)
    return app


def _model_metadata(model: ModelDescriptor) -> ModelMetadata:
    return ModelMetadata(
        name=model.name,
        description=model.description,
        release_date=model.release_date,
    )


def _is_json_compatible(value: object) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True
