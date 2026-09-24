"""Transport and expanded-input admission limits for the public HTTP API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.responses import JSONResponse

from .contracts import SystemOneBatchRequest, SystemOneRequest
from .model_inputs import (
    QWEN_DEFAULT_NO,
    QWEN_DEFAULT_YES,
    build_model_input,
    content_text,
    qwen_segments,
)

SINGLE_MAX_REQUEST_BYTES = 256 * 1024
BATCH_MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_EXPANDED_INPUT_BYTES = 16 * 1024 * 1024
# Conservatively covers Vela's task/candidate marker framing as well as
# tokenizer-side separators that are absent from the Qwen text rendering.
PER_DECISION_FRAMING_BYTES = 256

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
    """Bound the actual family prompt expansion without allocating every row.

    Qwen's per-option JSON and delimiters can greatly exceed the compact wire
    criteria map, especially at 255 options. Render each shared question once
    with an empty state, then account for each state by UTF-8 length. The
    larger Choice-null policy is used when both model families differ; a fixed
    row margin also covers Vela marker framing without tokenizing model data.
    """

    questions = tuple(request.questions.items())
    states = (
        (request.state,)
        if isinstance(request, SystemOneRequest)
        else tuple(item.state for item in request.states)
    )
    state_bytes = sum(len(content_text(state).encode("utf-8")) for state in states)
    question_bytes = sum(
        _question_rendered_bytes(question_id, question)
        for question_id, question in questions
    )
    decisions = len(states) * len(questions)
    return (
        state_bytes * len(questions)
        + question_bytes * len(states)
        + PER_DECISION_FRAMING_BYTES * decisions
    )


def exceeds_expanded_input_limit(
    request: SystemOneRequest | SystemOneBatchRequest,
) -> bool:
    return expanded_input_bytes(request) > MAX_EXPANDED_INPUT_BYTES


def _question_rendered_bytes(question_id: str, question) -> int:
    has_null_choice = question.type == "choice" and any(
        value is None for value in question.criteria.values()
    )
    policies = (
        ("render_key", "preserve_json_null") if has_null_choice else ("render_key",)
    )
    return max(
        len(
            qwen_segments(
                build_model_input(
                    question_id=question_id,
                    state="",
                    question=question,
                    choice_null_description=policy,
                    noul_default_false=QWEN_DEFAULT_NO,
                    noul_default_true=QWEN_DEFAULT_YES,
                    noul_explicit_null="use_default",
                )
            ).rendered.encode("utf-8")
        )
        for policy in policies
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
    if path == "/v1/systemone/batches":
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
