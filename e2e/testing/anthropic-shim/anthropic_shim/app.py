"""FastAPI proxy that fronts a llama-server with Anthropic-shape fixes.

Forwards every request as-is to the configured upstream after applying
the inbound translations from :mod:`anthropic_shim.translate`, then
post-processes the response to synthesise prompt-cache usage counters.

The proxy is intentionally minimal: it covers ``/v1/messages`` and the
streaming sibling, and passes everything else (``/health``, ``/v1/models``,
arbitrary debug routes) straight through.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .translate import (
    OpenAIStreamToAnthropic,
    anthropic_to_openai,
    apply_cache_usage,
    cache_prefix_hash,
    has_cache_control,
    join_system_array,
    join_tool_result_content,
    openai_to_anthropic,
)

LOGGER = logging.getLogger("anthropic_shim")

# Headers that httpx / Starlette manage themselves; forwarding them
# verbatim either confuses the upstream or breaks chunked transfer.
_HOP_BY_HOP = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

# Maximum number of sessions the request store will track before LRU eviction.
_MAX_REQUEST_STORE_SESSIONS = 32


def _contains_text(value: Any, marker: str) -> bool:
    if isinstance(value, str):
        return marker in value
    if isinstance(value, list):
        return any(_contains_text(item, marker) for item in value)
    if isinstance(value, dict):
        return any(_contains_text(item, marker) for item in value.values())
    return False


def _has_mock_tool_result(body: dict[str, Any]) -> bool:
    calls: set[str] = set()
    for message in body.get("messages", []):
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), list
        ):
            continue
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use":
                call_id = block.get("id")
                if isinstance(call_id, str) and call_id:
                    calls.add(call_id)
            elif block_type == "tool_result" and block.get("tool_use_id") in calls:
                return True
    return False


def _mock_anthropic_tool_response(body: dict[str, Any]) -> dict[str, Any] | None:
    if _has_mock_tool_result(body):
        return {
            "id": "msg_mock_tool_result_123",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", ""),
            "content": [{"type": "text", "text": "tool result accepted"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 6, "output_tokens": 3},
        }
    if _contains_text(body.get("messages"), "__mock_tool_call__") and body.get("tools"):
        return {
            "id": "msg_mock_tool_123",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", ""),
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_mock_lookup",
                    "name": "lookup",
                    "input": {"query": "weather"},
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 3},
        }
    return None


def _mock_anthropic_text_response(body: dict[str, Any]) -> dict[str, Any] | None:
    if not _contains_text(body.get("messages"), "__mock_protocol_matrix__"):
        return None
    return {
        "id": "msg_mock_protocol_matrix_123",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", ""),
        "content": [{"type": "text", "text": "protocol matrix accepted"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 3},
    }


def _mock_anthropic_tool_stream(body: dict[str, Any]) -> Iterator[bytes]:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock_tool_123",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model", ""),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 4, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_mock_lookup",
                    "name": "lookup",
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"weather"}'},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    for event, payload in events:
        yield f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _mock_anthropic_text_stream(body: dict[str, Any], text: str) -> Iterator[bytes]:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock_tool_result_123",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model", ""),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 6, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "", "citations": None},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    for event, payload in events:
        yield f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _mock_incomplete_anthropic_stream(body: dict[str, Any]) -> Iterator[bytes]:
    model = str(body.get("model", ""))
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock_incomplete",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 2, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            },
        ),
    ]
    for event, payload in events:
        yield f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _mock_midstream_anthropic_error(body: dict[str, Any]) -> Iterator[bytes]:
    yield from _mock_incomplete_anthropic_stream(body)
    payload = {
        "type": "error",
        "error": {
            "type": "overloaded_error",
            "message": "mock provider stream failed",
        },
    }
    yield f"event: error\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _filtered_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


class SessionCacheTracker:
    """Tracks per-session request-prefix hashes for cache synthesis.

    A session is identified by the header named in
    ``ANTHROPIC_SHIM_SESSION_HEADER`` (default
    ``x-vsr-test-session-id``). Requests without that header share a
    global session — fine for single-tenant test runs.
    """

    def __init__(self) -> None:
        self._seen: dict[str, set[str]] = {}

    def mark(self, session_id: str, prefix: str) -> bool:
        """Record ``prefix`` for ``session_id`` and return prior-seen status."""
        bucket = self._seen.setdefault(session_id, set())
        already_seen = prefix in bucket
        bucket.add(prefix)
        return already_seen


class RequestStore:
    """Per-session ring buffer (size 1) of the most recent request the shim
    was about to forward.

    Stores the translated request body alongside the headers the shim received,
    so tests can inspect what would have reached upstream without scraping logs.
    Note: ``record()`` is called *before* the upstream POST, so an upstream
    failure leaves a stale entry for that session. This is acceptable for a
    test-only shim — callers should treat the stored body as "last attempted
    forward", not "last successful forward".

    Bounded to ``_MAX_REQUEST_STORE_SESSIONS`` sessions with LRU eviction so
    long-running multi-session test suites do not leak memory.

    Concurrency: safe only under a single-worker async event loop (the default
    uvicorn deployment for this shim). Multi-worker deployment would shard
    sessions across processes non-deterministically. Inbound headers are stored
    verbatim, including credentials (``Authorization`` etc.); acceptable
    because the shim is loopback-only test infrastructure.
    """

    def __init__(self) -> None:
        # OrderedDict gives O(1) LRU move-to-end semantics.
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def record(
        self,
        session_id: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> None:
        """Store the most recent forwarded body + headers for ``session_id``."""
        if session_id in self._store:
            self._store.move_to_end(session_id)
        elif len(self._store) >= _MAX_REQUEST_STORE_SESSIONS:
            self._store.popitem(last=False)  # evict LRU entry
        self._store[session_id] = {"body": body, "headers": headers}

    def get(self, session_id: str) -> dict[str, Any] | None:
        """Return the most recent entry for ``session_id``, or None."""
        entry = self._store.get(session_id)
        if entry is not None:
            self._store.move_to_end(session_id)
        return entry


def create_app(
    upstream_url: str | None = None,
    session_header: str | None = None,
    request_timeout: float | None = None,
    openai_upstream: bool | None = None,
) -> FastAPI:
    upstream = (
        upstream_url
        or os.environ.get("ANTHROPIC_SHIM_UPSTREAM_URL")
        or "http://127.0.0.1:8081"
    ).rstrip("/")
    session_header_name = (
        session_header
        or os.environ.get("ANTHROPIC_SHIM_SESSION_HEADER")
        or "x-vsr-test-session-id"
    ).lower()
    timeout = float(
        request_timeout
        if request_timeout is not None
        else os.environ.get("ANTHROPIC_SHIM_REQUEST_TIMEOUT", "300")
    )
    use_openai_upstream = (
        openai_upstream
        if openai_upstream is not None
        else os.environ.get("ANTHROPIC_SHIM_OPENAI_UPSTREAM", "").lower()
        in {"1", "true", "yes"}
    )

    client = httpx.AsyncClient(base_url=upstream, timeout=timeout)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await _app.state.client.aclose()

    app = FastAPI(title="anthropic-shim", version="0.1.0", lifespan=lifespan)
    app.state.upstream = upstream
    app.state.tracker = SessionCacheTracker()
    app.state.request_store = RequestStore()
    app.state.session_header = session_header_name
    app.state.openai_upstream = use_openai_upstream
    app.state.client = client

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "upstream": upstream}

    @app.get("/debug/last-request")
    async def debug_last_request(request: Request) -> Response:
        """Return the most recent translated request for the given session.

        The session is identified by the configured session header key (default
        ``x-vsr-test-session-id``) supplied as either a request header or a
        query parameter (header takes precedence). Both paths use the same
        configurable key so that reconfiguring ANTHROPIC_SHIM_SESSION_HEADER
        does not leave query-param lookups pointing at the literal default.
        Returns 404 when no request has been seen for that session yet.
        """
        session_id = (
            request.headers.get(app.state.session_header)
            or request.query_params.get(app.state.session_header)
            or "__global__"
        )
        entry = app.state.request_store.get(session_id)
        if entry is None:
            return JSONResponse(
                status_code=404,
                content={"error": "not_found", "session_id": session_id},
            )
        return JSONResponse(content={"session_id": session_id, **entry})

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        return await _handle_messages(request, app)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def passthrough(path: str, request: Request) -> Response:
        return await _proxy_raw(request, path, app)

    return app


def _mock_messages_response(body: dict[str, Any]) -> Response | None:
    if _contains_text(body.get("messages"), "__mock_provider_error__"):
        return JSONResponse(
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
            content={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "mock provider rate limit",
                },
                "request_id": "req_mock_rate_limit",
            },
        )

    if body.get("stream") and _contains_text(
        body.get("messages"), "__mock_midstream_error__"
    ):
        return StreamingResponse(
            _mock_midstream_anthropic_error(body),
            media_type="text/event-stream",
        )

    if body.get("stream") and _contains_text(
        body.get("messages"), "__mock_incomplete_stream__"
    ):
        return StreamingResponse(
            _mock_incomplete_anthropic_stream(body),
            media_type="text/event-stream",
        )

    mock_text_response = _mock_anthropic_text_response(body)
    if mock_text_response is not None:
        if body.get("stream"):
            return StreamingResponse(
                _mock_anthropic_text_stream(body, "protocol matrix accepted"),
                media_type="text/event-stream",
            )
        return JSONResponse(content=mock_text_response)

    mock_tool_response = _mock_anthropic_tool_response(body)
    if mock_tool_response is not None:
        if body.get("stream"):
            if mock_tool_response.get("stop_reason") == "tool_use":
                iterator = _mock_anthropic_tool_stream(body)
            else:
                iterator = _mock_anthropic_text_stream(body, "tool result accepted")
            return StreamingResponse(iterator, media_type="text/event-stream")
        if not body.get("stream"):
            return JSONResponse(content=mock_tool_response)
    return None


async def _handle_messages(request: Request, app: FastAPI) -> Response:
    raw_body = await request.body()
    try:
        body: dict[str, Any] = (
            httpx.Response(HTTPStatus.OK, content=raw_body).json() if raw_body else {}
        )
    except ValueError:
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={"error": "invalid_json", "detail": "body is not JSON"},
        )

    request_had_cache_control = has_cache_control(body)
    prefix_hash = cache_prefix_hash(body) if request_had_cache_control else ""

    body = join_system_array(body)
    body = join_tool_result_content(body)

    mock_response = _mock_messages_response(body)
    if mock_response is not None:
        return mock_response

    session_id = request.headers.get(app.state.session_header) or "__global__"
    headers = _filtered_headers(dict(request.headers))
    headers.pop("content-length", None)

    is_streaming = bool(body.get("stream"))
    if is_streaming:
        app.state.request_store.record(session_id, body, dict(request.headers))
        upstream_path = (
            "/v1/chat/completions" if app.state.openai_upstream else "/v1/messages"
        )
        upstream_body = anthropic_to_openai(body) if app.state.openai_upstream else body
        return await _proxy_stream(
            app,
            upstream_path,
            upstream_body,
            headers,
            request_body=body,
            translate_openai=app.state.openai_upstream,
        )

    # Record what the shim is about to forward so /debug/last-request can
    # return the translated body + inbound headers for test assertions.
    app.state.request_store.record(session_id, body, dict(request.headers))

    upstream_path = (
        "/v1/chat/completions" if app.state.openai_upstream else "/v1/messages"
    )
    upstream_body = anthropic_to_openai(body) if app.state.openai_upstream else body
    upstream_response = await app.state.client.post(
        upstream_path,
        json=upstream_body,
        headers=headers,
    )

    response_headers = _filtered_headers(dict(upstream_response.headers))
    try:
        response_payload = upstream_response.json()
    except ValueError:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=upstream_response.headers.get("content-type"),
        )
    if app.state.openai_upstream and isinstance(response_payload, dict):
        response_payload = openai_to_anthropic(response_payload, body)

    if request_had_cache_control and isinstance(response_payload, dict):
        prefix_seen = app.state.tracker.mark(session_id, prefix_hash)
        apply_cache_usage(
            response_payload,
            request_had_cache_control=True,
            prefix_seen=prefix_seen,
        )

    return JSONResponse(
        content=response_payload,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


async def _proxy_stream(
    app: FastAPI,
    path: str,
    body: dict[str, Any],
    headers: dict[str, str],
    *,
    request_body: dict[str, Any],
    translate_openai: bool,
) -> StreamingResponse:
    """Stream a native Messages response or adapt OpenAI SSE to Messages.

    Cache usage post-processing is not applied to streamed responses
    because the e2e cache-cycle tests use non-streaming requests.
    """

    upstream_request = app.state.client.build_request(
        "POST", path, json=body, headers=headers
    )
    upstream = await app.state.client.send(upstream_request, stream=True)

    async def iterate() -> AsyncIterator[bytes]:
        try:
            if not translate_openai or upstream.status_code >= HTTPStatus.BAD_REQUEST:
                async for chunk in upstream.aiter_raw():
                    yield chunk
                return
            adapter = OpenAIStreamToAnthropic(request_body)
            async for line in upstream.aiter_lines():
                if not line.startswith("data:"):
                    continue
                for event in adapter.feed(line.removeprefix("data:").strip()):
                    yield event
            for event in adapter.finish():
                yield event
        finally:
            await upstream.aclose()

    return StreamingResponse(
        iterate(),
        status_code=upstream.status_code,
        headers=_filtered_headers(dict(upstream.headers)),
        media_type="text/event-stream",
    )


async def _proxy_raw(request: Request, path: str, app: FastAPI) -> Response:
    body = await request.body()
    headers = _filtered_headers(dict(request.headers))
    upstream_response = await app.state.client.request(
        request.method,
        "/" + path,
        params=dict(request.query_params),
        content=body,
        headers=headers,
    )
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_filtered_headers(dict(upstream_response.headers)),
        media_type=upstream_response.headers.get("content-type"),
    )


app = create_app()
