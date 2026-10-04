"""HTTP routes of the runtime contract (``openapi.yaml``)."""

from __future__ import annotations

import asyncio
import json
import re
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
_SURROGATE_ESCAPE = re.compile(rb"\\u[dD][89a-fA-F]")


class JSON(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")


def create_app(runtime: Runtime) -> Starlette:
    async def decisions(request: Request) -> Response:
        endpoint = request.url.path
        started = time.perf_counter()
        status = 200
        try:
            body = await _read_json(request, runtime.config.max_request_bytes)
            if not runtime.health.ready:
                raise RuntimeServiceError(
                    "not_ready", f"the model is {runtime.health.state}"
                )
            parsed = runtime.parse(body)
            plan = await run_in_threadpool(runtime.plan, parsed)
            submitted = time.monotonic()
            future = runtime.submit(plan, parsed)
            results = await asyncio.wrap_future(future)
            finished = time.monotonic()
            queue_ms = (submitted - parsed.received) * 1000.0
            compute_ms = (finished - submitted) * 1000.0
            return JSON(runtime.assemble(parsed, plan, results, queue_ms, compute_ms))
        except RuntimeServiceError as exc:
            status = exc.status
            return JSON(exc.body(), status_code=exc.status)
        except Exception as exc:
            status = 500
            failure = runtime.device_failure()
            if failure is not None:
                runtime.degrade(f"{type(failure).__name__}: {failure}")
            error = RuntimeServiceError(
                "internal_error", f"{type(exc).__name__}: {exc}"
            )
            return JSON(error.body(), status_code=500)
        finally:
            runtime.metrics.requests.labels(endpoint=endpoint, status=str(status)).inc()
            runtime.metrics.request_seconds.labels(endpoint=endpoint).observe(
                time.perf_counter() - started
            )

    async def models(request: Request) -> Response:
        return JSON({"object": "list", "data": [runtime.model_card()]})

    async def health(request: Request) -> Response:
        body = {
            "status": runtime.health.state,
            "reason": runtime.health.reason,
            "model": runtime.served_id,
        }
        return JSON(body, status_code=200 if runtime.health.ready else 503)

    async def live(request: Request) -> Response:
        return JSON({"status": "alive", "reason": None, "model": runtime.served_id})

    async def metrics(request: Request) -> Response:
        if runtime.scheduler is not None:
            runtime.metrics.queue_depth.set(runtime.scheduler.depth())
        return Response(
            runtime.metrics.render(), media_type="text/plain; version=0.0.4"
        )

    async def openapi(request: Request) -> Response:
        return PlainTextResponse(
            OPENAPI_PATH.read_text(encoding="utf-8"), media_type="application/yaml"
        )

    return Starlette(
        routes=[
            Route("/v1/decisions", decisions, methods=["POST"]),
            Route("/v1/systemone", decisions, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
            Route("/health/live", live, methods=["GET"]),
            Route("/metrics", metrics, methods=["GET"]),
            Route("/openapi.yaml", openapi, methods=["GET"]),
        ],
        exception_handlers={404: _not_found, 405: _not_allowed},
    )


async def _read_json(request: Request, limit: int) -> Any:
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
    return body


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
