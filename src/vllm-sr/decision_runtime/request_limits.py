"""Transport and expanded-input admission limits for the public HTTP API."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.responses import JSONResponse

from .contracts import SystemOneBatchRequest, SystemOneRequest

SINGLE_MAX_REQUEST_BYTES = 256 * 1024
BATCH_MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_EXPANDED_INPUT_BYTES = 16 * 1024 * 1024

ASGIApp = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[dict[str, Any]]],
        Callable[..., Awaitable[None]],
    ],
    Awaitable[None],
]


class BoundedJSONBodyMiddleware:
    """Read, bound, and replay API bodies before framework JSON parsing.

    This closes chunked-transfer and misleading Content-Length bypasses. Only
    the two inference endpoints are buffered; health, status, metrics, and
    OpenAPI traffic remain streaming and untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        limit = _request_limit(scope)
        if limit is None:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        content_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip()
        if content_type.lower() != b"application/json":
            await _send_error(send, 415, "Use Content-Type: application/json")
            return
        content_encoding = headers.get(b"content-encoding", b"identity").strip().lower()
        if content_encoding not in {b"", b"identity"}:
            await _send_error(send, 415, "Compressed request bodies are not accepted")
            return
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError:
                await _send_error(send, 400, "Invalid Content-Length")
                return
            if declared_size < 0:
                await _send_error(send, 400, "Invalid Content-Length")
                return
            if declared_size > limit:
                await _send_error(send, 413, "Request body exceeds the endpoint limit")
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            if len(chunk) > limit - len(body):
                await _send_error(send, 413, "Request body exceeds the endpoint limit")
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": bytes(body),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay, send)


def expanded_input_bytes(
    request: SystemOneRequest | SystemOneBatchRequest,
) -> int:
    """Return a conservative UTF-8 size for the logical question rows."""

    questions = tuple(request.questions.values())
    question_bytes = sum(
        _json_size(question.model_dump(mode="json")) for question in questions
    )
    if isinstance(request, SystemOneRequest):
        state_bytes = _json_size(request.state)
        decisions = len(questions)
        return state_bytes * decisions + question_bytes + 256 * decisions

    state_bytes = sum(_json_size(state.state) for state in request.states)
    decisions = len(request.states) * len(questions)
    return (
        state_bytes * len(questions)
        + question_bytes * len(request.states)
        + 256 * decisions
    )


def exceeds_expanded_input_limit(
    request: SystemOneRequest | SystemOneBatchRequest,
) -> bool:
    return expanded_input_bytes(request) > MAX_EXPANDED_INPUT_BYTES


def _json_size(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _request_limit(scope: dict[str, Any]) -> int | None:
    if scope.get("type") != "http" or scope.get("method") != "POST":
        return None
    path = scope.get("path")
    root_path = scope.get("root_path", "").rstrip("/")
    if root_path and isinstance(path, str):
        if path == root_path:
            path = "/"
        elif path.startswith(f"{root_path}/"):
            path = path[len(root_path) :]
    if path == "/v1/systemone":
        return SINGLE_MAX_REQUEST_BYTES
    if path == "/v1/decision/batches":
        return BATCH_MAX_REQUEST_BYTES
    return None


async def _send_error(send, status_code: int, detail: str) -> None:
    response = JSONResponse(status_code=status_code, content={"detail": detail})
    await response(
        {"type": "http"},
        _never_receive,
        send,
    )


async def _never_receive() -> dict[str, Any]:
    return {"type": "http.disconnect"}
