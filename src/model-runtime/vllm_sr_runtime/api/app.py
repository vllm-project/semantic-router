"""HTTP routes of the runtime contract (``openapi.yaml``)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from ..errors import RuntimeServiceError
from ..runtime import Runtime

OPENAPI_PATH = Path(__file__).with_name("openapi.yaml")


class JSON(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")


def create_app(runtime: Runtime) -> Starlette:
    def observe(endpoint: str, status: int, started: float) -> None:
        runtime.metrics.requests.labels(endpoint=endpoint, status=str(status)).inc()
        runtime.metrics.request_seconds.labels(endpoint=endpoint).observe(
            time.perf_counter() - started
        )

    def surface_route(surface: str):
        async def handle(request: Request) -> Response:
            endpoint = request.url.path
            started = time.perf_counter()
            status = 200
            try:
                body, size = await _read_json(request, runtime.config.max_request_bytes)
                status, response = await runtime.call(surface, body, size)
                return JSON(response, status_code=status)
            except RuntimeServiceError as exc:
                status = exc.status
                return JSON(exc.body(), status_code=exc.status)
            finally:
                observe(endpoint, status, started)

        return handle

    async def bundle(request: Request) -> Response:
        started = time.perf_counter()
        status = 200
        try:
            body, size = await _read_json(request, runtime.config.max_request_bytes)
            status, response = await runtime.bundle(body, size)
            return JSON(response, status_code=status)
        except RuntimeServiceError as exc:
            status = exc.status
            return JSON(exc.body(), status_code=exc.status)
        finally:
            observe(request.url.path, status, started)

    async def models(request: Request) -> Response:
        # The first call imports every plugin to describe it; /health must not wait for that.
        return JSON(
            {"object": "list", "data": await run_in_threadpool(runtime.model_cards)}
        )

    async def health(request: Request) -> Response:
        body: dict[str, Any] = {
            "status": runtime.health.state,
            "reason": runtime.health.reason,
            "model": runtime.served_id,
        }
        if len(runtime.served) > 1:
            body["models"] = runtime.health.describe()
        return JSON(body, status_code=200 if runtime.health.ready else 503)

    async def live(request: Request) -> Response:
        return JSON({"status": "alive", "reason": None, "model": runtime.served_id})

    async def metrics(request: Request) -> Response:
        if runtime.scheduler is not None:
            runtime.metrics.queue_depth.set(
                sum(
                    served.scheduler.depth()
                    for served in runtime.served
                    if served.scheduler
                )
            )
        return Response(
            runtime.metrics.render(), media_type="text/plain; version=0.0.4"
        )

    async def openapi(request: Request) -> Response:
        return PlainTextResponse(
            OPENAPI_PATH.read_text(encoding="utf-8"), media_type="application/yaml"
        )

    decisions = surface_route("decisions")
    return Starlette(
        routes=[
            Route("/v1/decisions", decisions, methods=["POST"]),
            Route("/v1/systemone", decisions, methods=["POST"]),
            Route("/v1/classify", surface_route("classify"), methods=["POST"]),
            Route("/v1/embeddings", surface_route("embeddings"), methods=["POST"]),
            Route("/v1/rerank", surface_route("rerank"), methods=["POST"]),
            Route("/v1/bundle", bundle, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
            Route("/health/live", live, methods=["GET"]),
            Route("/metrics", metrics, methods=["GET"]),
            Route("/openapi.yaml", openapi, methods=["GET"]),
        ],
        exception_handlers={404: _not_found, 405: _not_allowed},
    )


async def _read_json(request: Request, limit: int) -> tuple[Any, int]:
    """The parsed body and its size in bytes."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise RuntimeServiceError(
            "request_too_large", f"the request body exceeds {limit} bytes"
        )
    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise RuntimeServiceError(
                "request_too_large", f"the request body exceeds {limit} bytes"
            )
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks)), size
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeServiceError(
            "invalid_request", f"the request body is not JSON: {exc}"
        ) from exc


async def _not_found(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        {
            "error": {
                "code": "invalid_request",
                "message": f"no route {request.url.path}",
            }
        },
        404,
    )


async def _not_allowed(request: Request, exc: Exception) -> Response:
    return JSONResponse(
        {"error": {"code": "invalid_request", "message": "method not allowed"}}, 405
    )
