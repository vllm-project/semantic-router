"""HTTP routes of the runtime contract (``openapi.yaml``)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any, TypeVar

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from ..errors import RuntimeServiceError
from ..runtime import Runtime
from ..timing import ServerTiming

OPENAPI_PATH = Path(__file__).with_name("openapi.yaml")
_SURROGATE_ESCAPE = re.compile(rb"\\u[dD][89a-fA-F]")
# The contract version (``info.version`` in openapi.yaml), reported to clients.
API_VERSION = "2.1.0"
# Recorded for a request whose client disconnected before its answer (nginx's code).
CLIENT_CLOSED = 499

T = TypeVar("T")


class JSON(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")


def non_finite() -> RuntimeServiceError:
    return RuntimeServiceError(
        "internal_error", "the model produced a non-finite number"
    )


def respond(content: Any, status: int = 200) -> Response:
    """A JSON response; one holding NaN or infinity becomes the contract's 500."""
    try:
        return JSON(content, status_code=status)
    except ValueError:
        error = non_finite()
        return JSON(error.body(), status_code=error.status)


def respond_bundle(content: dict[str, Any]) -> Response:
    """A bundle response; a task whose result holds NaN or infinity alone answers 500."""
    try:
        return JSON(content)
    except ValueError:
        results = []
        for result in content["results"]:
            try:
                json.dumps(result, allow_nan=False)
                results.append(result)
            except ValueError:
                error = non_finite()
                results.append(
                    {"id": result["id"], "status": error.status, **error.body()}
                )
        return JSON({**content, "results": results})


def stamped(response: Response, timing: ServerTiming, started: float) -> Response:
    """``response`` with its request's ``Server-Timing`` header; ``started`` is the handler's start."""
    response.headers["server-timing"] = timing.header(time.perf_counter() - started)
    return response


def create_app(runtime: Runtime) -> Starlette:
    def observe(endpoint: str, status: int, started: float) -> None:
        runtime.metrics.requests.labels(endpoint=endpoint, status=str(status)).inc()
        runtime.metrics.request_seconds.labels(endpoint=endpoint).observe(
            time.perf_counter() - started
        )

    def surface_route(surface: str) -> Callable[[Request], Awaitable[Response]]:
        async def handle(request: Request) -> Response:
            endpoint = request.url.path
            started = time.perf_counter()
            timing = ServerTiming()
            status = 200
            try:
                body, size = await _read_json(request, runtime.config.max_request_bytes)
                timing.parse = time.perf_counter() - started
                outcome = await until_disconnect(
                    request, runtime.call(surface, body, size, timing)
                )
                if outcome is None:
                    status = CLIENT_CLOSED
                    return Response(status_code=CLIENT_CLOSED)
                status, response = outcome
                serializing = time.perf_counter()
                answer = respond(response, status)
                timing.serialize = time.perf_counter() - serializing
                status = answer.status_code
                return stamped(answer, timing, started)
            except RuntimeServiceError as exc:
                status = exc.status
                return stamped(
                    JSON(exc.body(), status_code=exc.status), timing, started
                )
            finally:
                observe(endpoint, status, started)

        return handle

    async def bundle(request: Request) -> Response:
        started = time.perf_counter()
        timing = ServerTiming()
        status = 200
        try:
            body, size = await _read_json(request, runtime.config.max_request_bytes)
            timing.parse = time.perf_counter() - started
            outcome = await until_disconnect(
                request, runtime.bundle(body, size, timing)
            )
            if outcome is None:
                status = CLIENT_CLOSED
                return Response(status_code=CLIENT_CLOSED)
            status, response = outcome
            serializing = time.perf_counter()
            answer = (
                respond_bundle(response)
                if status == HTTPStatus.OK
                else respond(response, status)
            )
            timing.serialize = time.perf_counter() - serializing
            return stamped(answer, timing, started)
        except RuntimeServiceError as exc:
            status = exc.status
            return stamped(JSON(exc.body(), status_code=exc.status), timing, started)
        finally:
            observe(request.url.path, status, started)

    async def models(request: Request) -> Response:
        # The first call imports every plugin to describe it; /health must not wait for that.
        return JSON(
            {
                "object": "list",
                "api_version": API_VERSION,
                "data": await run_in_threadpool(runtime.model_cards),
                "limits": {
                    "max_bundle_tasks": runtime.config.max_bundle_tasks,
                    "max_request_bytes": runtime.config.max_request_bytes,
                },
            }
        )

    def only_model() -> str | None:
        """The served model's ID; a process serving several lists them in ``models``."""
        return runtime.served[0].served_id if len(runtime.served) == 1 else None

    async def health(request: Request) -> Response:
        body: dict[str, Any] = {
            "api_version": API_VERSION,
            "status": runtime.health.state,
            "reason": runtime.health.reason,
            "model": only_model(),
        }
        if len(runtime.served) > 1:
            body["models"] = runtime.health.describe()
        return JSON(body, status_code=200 if runtime.health.ready else 503)

    async def live(request: Request) -> Response:
        return JSON({"api_version": API_VERSION, "status": "alive"})

    async def metrics(request: Request) -> Response:
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


async def until_disconnect(request: Request, work: Awaitable[T]) -> T | None:
    """``work``'s result, or None once the client disconnects first.

    Then ``work`` is cancelled, which cancels its jobs' futures, so the
    scheduler skips their batches that have not run yet.
    """
    task = asyncio.ensure_future(work)
    left = asyncio.ensure_future(_disconnected(request))
    try:
        await asyncio.wait((task, left), return_when=asyncio.FIRST_COMPLETED)
    finally:
        left.cancel()
        if not task.done():
            task.cancel()
    return task.result() if task.done() else None


async def _disconnected(request: Request) -> None:
    """Return when the client disconnects; the body has been read already."""
    while (await request.receive())["type"] != "http.disconnect":
        pass


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
    raw = b"".join(chunks)
    try:
        body = json.loads(raw.decode("utf-8"))
        # Strict UTF-8 cannot carry a surrogate, so only a \uD800-\uDFFF escape
        # can leave an unpaired one that no tokenizer accepts. Only a body with
        # such an escape is re-encoded to find out.
        if _SURROGATE_ESCAPE.search(raw):
            json.dumps(body, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RuntimeServiceError(
            "invalid_request", "the request contains an unpaired surrogate"
        ) from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeServiceError(
            "invalid_request", f"the request body is not JSON: {exc}"
        ) from exc
    return body, size


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
