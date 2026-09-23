"""FastAPI surface for a backend-neutral Decision runtime."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, TypeVar

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

from .backend import (
    BackendContractError,
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
from .scheduler import ModelScheduler, SchedulerOverloadedError

EvaluationResponseT = TypeVar(
    "EvaluationResponseT", SystemOneResponse, SystemOneBatchResponse
)


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
) -> FastAPI:
    """Build the HTTP API without importing or selecting a model framework."""

    runtime_metrics = metrics or RuntimeMetrics()
    model_names = tuple(model.name for model in engine.models)
    served_model_names = frozenset(model_names)
    runtime_scheduler = scheduler or ModelScheduler(model_names)
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
            return {"status": "ready"}
        return JSONResponse(
            status_code=503,
            content={"detail": "Decision backend is not ready"},
        )

    @app.get("/v1/models", response_model=ModelMetadataList)
    async def models():
        return ModelMetadataList(
            models=[_model_metadata(model) for model in engine.models]
        )

    async def run_evaluation(
        model: str,
        operation: Callable[[], Awaitable[EvaluationResponseT]],
    ) -> EvaluationResponseT | JSONResponse:
        started = time.perf_counter()
        outcome = "internal_error"
        try:
            if model not in served_model_names:
                raise UnknownModelError(model)
            response = await runtime_scheduler.run(
                model,
                operation,
            )
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
        except SchedulerOverloadedError:
            outcome = "overloaded"
            return JSONResponse(
                status_code=529,
                content={"detail": "Decision runtime is temporarily overloaded"},
                headers={"Retry-After": "1"},
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
    ) -> SystemOneResponse | JSONResponse:
        return await run_evaluation(
            payload.model,
            lambda: engine.evaluate(payload),
        )

    @app.post("/v1/systemone/batch", response_model=SystemOneBatchResponse)
    async def system_one_batch(
        payload: Annotated[SystemOneBatchRequest, Body(...)],
    ) -> SystemOneBatchResponse | JSONResponse:
        return await run_evaluation(
            payload.model,
            lambda: engine.evaluate_batch(payload),
        )

    @app.get("/api/status")
    async def status():
        snapshots = await runtime_scheduler.snapshots()
        is_ready = await engine.ready()
        return {
            "status": "ready" if is_ready else "not_ready",
            "contract": "systemone.single.v1",
            "confidence": {
                "name": "decision_normalized_top",
                "typesafe_equivalent": False,
            },
            "models": [model.name for model in engine.models],
            "scheduler": [
                {
                    "model": snapshot.model,
                    "running": snapshot.running,
                    "queued": snapshot.queued,
                    "max_concurrency": snapshot.max_concurrency,
                    "max_queue": snapshot.max_queue,
                }
                for snapshot in snapshots
            ],
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return PlainTextResponse(
            runtime_metrics.render(await runtime_scheduler.snapshots()),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

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
