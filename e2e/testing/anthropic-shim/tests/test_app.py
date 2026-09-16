"""Integration tests for the FastAPI proxy using a stub upstream.

These avoid spinning up llama-server by mounting a tiny in-process
ASGI app as the proxy target. They verify that the shim:
- forwards the joined ``system`` string to the upstream
- joins ``tool_result.content`` arrays before forwarding
- post-processes responses to synthesise prompt-cache token counters
- tracks repeat-prefix state per session header
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from anthropic_shim.app import _MAX_REQUEST_STORE_SESSIONS, create_app


class _UpstreamRecorder:
    """In-process stand-in for llama-server.

    Records every request body it receives, and emits a canned response
    so the test can assert on the shim's pre- and post-processing
    without involving a real model.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.input_tokens = 42

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        self.requests.append(body)
        payload = {
            "id": f"msg_{len(self.requests)}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "qwen-test"),
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 1},
        }
        return httpx.Response(200, json=payload)


@pytest.fixture()
def client_with_upstream() -> tuple[httpx.AsyncClient, _UpstreamRecorder]:
    upstream = _UpstreamRecorder()
    app = create_app(upstream_url="http://upstream.invalid")
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream.handler),
        base_url="http://upstream.invalid",
    )
    return (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://shim"
        ),
        upstream,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_structured_output_mock_requires_and_echoes_schema(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
    stream: bool,
) -> None:
    client, upstream = client_with_upstream
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    response = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "messages": [
                {
                    "role": "user",
                    "content": "__mock_structured_output__ Return an answer object.",
                }
            ],
            "max_tokens": 16,
            "stream": stream,
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        },
    )

    assert response.status_code == 200
    assert not upstream.requests
    if stream:
        assert "event: message_stop" in response.text
        assert "structured_output" in response.text
    else:
        text = response.json()["content"][0]["text"]
        assert json.loads(text) == {
            "mock": "mock-vllm",
            "structured_output": {"type": "json_schema", "schema": schema},
        }


@pytest.mark.asyncio
async def test_structured_output_marker_without_schema_reaches_upstream(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, upstream = client_with_upstream
    response = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "__mock_structured_output__"}],
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    assert len(upstream.requests) == 1


@pytest.mark.asyncio
async def test_messages_joins_system_array_before_forwarding(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, upstream = client_with_upstream
    payload = {
        "model": "qwen-test",
        "system": [
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": "Be very concise."},
        ],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    await client.post("/v1/messages", json=payload)
    assert upstream.requests
    forwarded = upstream.requests[-1]
    assert forwarded["system"] == "You are a helpful assistant.\nBe very concise."


@pytest.mark.asyncio
async def test_messages_translates_for_openai_upstream() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "qwen-test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1},
            },
        )

    app = create_app(
        upstream_url="http://upstream.invalid",
        openai_upstream=True,
    )
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://upstream.invalid",
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://shim",
    )

    response = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    await client.aclose()

    assert seen[0].url.path == "/v1/chat/completions"
    assert response.json()["content"] == [{"type": "text", "text": "ok"}]
    assert response.json()["usage"] == {"input_tokens": 8, "output_tokens": 1}


@pytest.mark.asyncio
async def test_messages_stream_translates_openai_sse_to_anthropic_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.content
        assert httpx.Response(200, content=request.content).json()[
            "stream_options"
        ] == {"include_usage": True}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"id":"chatcmpl-stream","model":"qwen-test","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
                b'data: {"id":"chatcmpl-stream","model":"qwen-test","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    app = create_app(
        upstream_url="http://upstream.invalid",
        openai_upstream=True,
    )
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://upstream.invalid",
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://shim",
    )

    response = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": True,
        },
    )
    await client.aclose()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line.removeprefix("event: ")
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert events == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


@pytest.mark.asyncio
async def test_messages_joins_tool_result_array_before_forwarding(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, upstream = client_with_upstream
    payload = {
        "model": "qwen-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc",
                        "content": [
                            {"type": "text", "text": "first"},
                            {"type": "text", "text": "second"},
                        ],
                    }
                ],
            }
        ],
        "max_tokens": 16,
    }
    await client.post("/v1/messages", json=payload)
    forwarded = upstream.requests[-1]
    assert forwarded["messages"][0]["content"][0]["content"] == "first\nsecond"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_protocol_matrix_probe_is_native_anthropic_and_deterministic(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder], stream: bool
) -> None:
    client, upstream = client_with_upstream
    response = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "max_tokens": 16,
            "stream": stream,
            "messages": [{"role": "user", "content": "__mock_protocol_matrix__"}],
        },
    )

    if stream:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert (
            '"type":"text_delta","text":"{\\"protocol\\":\\"anthropic_messages\\"}"'
            in response.text
        )
        assert '"stop_reason":"end_turn"' in response.text
        assert "event: message_stop" in response.text
    else:
        assert response.json()["content"] == [
            {"type": "text", "text": '{"protocol":"anthropic_messages"}'}
        ]
        assert response.json()["stop_reason"] == "end_turn"
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_mock_tool_lifecycle_is_native_anthropic_and_deterministic(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, upstream = client_with_upstream
    first = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "__mock_tool_call__"}],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        },
    )
    tool = first.json()["content"][0]
    assert tool == {
        "type": "tool_use",
        "id": "call_mock_lookup",
        "name": "lookup",
        "input": {"query": "weather"},
    }

    second = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "max_tokens": 16,
            "messages": [
                {"role": "assistant", "content": [tool]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": "sunny",
                        }
                    ],
                },
            ],
        },
    )
    assert second.json()["content"] == [
        {"type": "text", "text": "tool result accepted"}
    ]
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_mock_tool_lifecycle_stream_is_native_anthropic_and_deterministic(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, upstream = client_with_upstream
    response = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "__mock_tool_call__"}],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
        },
    )

    stream = response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_start" in stream
    assert '"type":"tool_use","id":"call_mock_lookup","name":"lookup"' in stream
    assert stream.count('"type":"input_json_delta"') == 2
    assert '"stop_reason":"tool_use"' in stream
    assert "event: message_stop" in stream

    result = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "max_tokens": 16,
            "stream": True,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_mock_lookup",
                            "name": "lookup",
                            "input": {"query": "weather"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_mock_lookup",
                            "content": "sunny",
                        }
                    ],
                },
            ],
        },
    )
    result_stream = result.text
    assert '"type":"text_delta","text":"tool result accepted"' in result_stream
    assert '"stop_reason":"end_turn"' in result_stream
    assert "event: message_stop" in result_stream
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_mock_provider_error_is_native_anthropic_and_deterministic(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, upstream = client_with_upstream
    response = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "__mock_provider_error__"}],
        },
    )
    assert response.status_code == 429
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "mock provider rate limit",
        },
        "request_id": "req_mock_rate_limit",
    }
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_mock_incomplete_stream_has_no_success_terminal(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, upstream = client_with_upstream
    response = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "__mock_incomplete_stream__"}],
        },
    )
    assert response.status_code == 200
    events = [
        line.removeprefix("event: ")
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert events == [
        "message_start",
        "content_block_start",
        "content_block_delta",
    ]
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_mock_midstream_error_follows_partial_content_without_success_terminal(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, upstream = client_with_upstream
    response = await client.post(
        "/v1/messages",
        json={
            "model": "qwen-test",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "__mock_midstream_error__"}],
        },
    )
    assert response.status_code == 200
    assert '"text":"partial"' in response.text
    assert "event: error" in response.text
    assert '"type":"overloaded_error"' in response.text
    assert "event: message_stop" not in response.text
    assert response.text.index('"text":"partial"') < response.text.index("event: error")
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_cache_usage_synthesised_on_first_then_repeat_request(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, _ = client_with_upstream
    payload = {
        "model": "qwen-test",
        "system": [
            {
                "type": "text",
                "text": "long prefix",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    headers = {"x-vsr-test-session-id": "session-a"}

    first = await client.post("/v1/messages", json=payload, headers=headers)
    second = await client.post("/v1/messages", json=payload, headers=headers)

    first_usage = first.json()["usage"]
    second_usage = second.json()["usage"]
    assert first_usage["cache_creation_input_tokens"] == 42
    assert first_usage["cache_read_input_tokens"] == 0
    assert second_usage["cache_creation_input_tokens"] == 0
    assert second_usage["cache_read_input_tokens"] == 42


@pytest.mark.asyncio
async def test_cache_usage_untouched_when_request_has_no_cache_control(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, _ = client_with_upstream
    payload = {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    response = await client.post("/v1/messages", json=payload)
    usage = response.json()["usage"]
    assert "cache_creation_input_tokens" not in usage
    assert "cache_read_input_tokens" not in usage


@pytest.mark.asyncio
async def test_session_isolation_with_distinct_session_headers(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, _ = client_with_upstream
    payload = {
        "model": "qwen-test",
        "system": [
            {
                "type": "text",
                "text": "long prefix",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    first = await client.post(
        "/v1/messages", json=payload, headers={"x-vsr-test-session-id": "alpha"}
    )
    second_other_session = await client.post(
        "/v1/messages", json=payload, headers={"x-vsr-test-session-id": "beta"}
    )
    assert first.json()["usage"]["cache_creation_input_tokens"] == 42
    # different session: counts as first request again
    assert second_other_session.json()["usage"]["cache_creation_input_tokens"] == 42


@pytest.mark.asyncio
async def test_invalid_json_returns_400(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, _ = client_with_upstream
    response = await client.post(
        "/v1/messages",
        content=b"not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


# ── /debug/last-request tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_debug_last_request_returns_404_before_any_request(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, _ = client_with_upstream
    response = await client.get(
        "/debug/last-request",
        headers={"x-vsr-test-session-id": "session-new"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_debug_last_request_returns_native_provider_body_after_messages_post(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, _ = client_with_upstream
    payload = {
        "model": "qwen-test",
        "system": [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ],
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
    }
    session = "debug-session-1"
    await client.post(
        "/v1/messages", json=payload, headers={"x-vsr-test-session-id": session}
    )

    response = await client.get(
        "/debug/last-request",
        headers={"x-vsr-test-session-id": session},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["body"]["system"] == payload["system"]
    assert data["session_id"] == session
    assert "headers" in data


@pytest.mark.asyncio
async def test_debug_last_request_session_via_query_param(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, _ = client_with_upstream
    payload = {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
    }
    session = "debug-qp-session"
    await client.post(
        "/v1/messages", json=payload, headers={"x-vsr-test-session-id": session}
    )

    # Retrieve via query param instead of header.
    response = await client.get(
        f"/debug/last-request?x-vsr-test-session-id={session}",
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == session


@pytest.mark.asyncio
async def test_debug_last_request_reflects_most_recent_request(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, _ = client_with_upstream
    session = "debug-overwrite-session"
    for content in ("first", "second"):
        await client.post(
            "/v1/messages",
            json={
                "model": "qwen-test",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 8,
            },
            headers={"x-vsr-test-session-id": session},
        )

    response = await client.get(
        "/debug/last-request",
        headers={"x-vsr-test-session-id": session},
    )
    assert response.status_code == 200
    # Only the most recent request is retained.
    assert response.json()["body"]["messages"][0]["content"] == "second"


@pytest.mark.asyncio
async def test_debug_last_request_session_isolation(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, _ = client_with_upstream
    payload_a = {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "alpha"}],
        "max_tokens": 8,
    }
    payload_b = {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "beta"}],
        "max_tokens": 8,
    }
    await client.post(
        "/v1/messages", json=payload_a, headers={"x-vsr-test-session-id": "alpha"}
    )
    await client.post(
        "/v1/messages", json=payload_b, headers={"x-vsr-test-session-id": "beta"}
    )

    resp_a = await client.get(
        "/debug/last-request", headers={"x-vsr-test-session-id": "alpha"}
    )
    resp_b = await client.get(
        "/debug/last-request", headers={"x-vsr-test-session-id": "beta"}
    )
    assert resp_a.json()["body"]["messages"][0]["content"] == "alpha"
    assert resp_b.json()["body"]["messages"][0]["content"] == "beta"


@pytest.mark.asyncio
async def test_request_store_lru_evicts_oldest_session(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    """Filling _MAX_REQUEST_STORE_SESSIONS+1 sessions evicts the oldest."""
    client, _ = client_with_upstream
    # Session IDs in insertion order; the first one must be evicted.
    session_ids = [
        f"lru-session-{i:03d}" for i in range(_MAX_REQUEST_STORE_SESSIONS + 1)
    ]

    for sid in session_ids:
        await client.post(
            "/v1/messages",
            json={
                "model": "qwen-test",
                "messages": [{"role": "user", "content": sid}],
                "max_tokens": 8,
            },
            headers={"x-vsr-test-session-id": sid},
        )

    # The oldest session must have been evicted.
    evicted = session_ids[0]
    resp_evicted = await client.get(
        "/debug/last-request",
        headers={"x-vsr-test-session-id": evicted},
    )
    assert (
        resp_evicted.status_code == 404
    ), f"expected evicted session {evicted!r} to return 404, got {resp_evicted.status_code}"

    # The most recent _MAX_REQUEST_STORE_SESSIONS sessions must still be present.
    for sid in session_ids[1:]:
        resp = await client.get(
            "/debug/last-request",
            headers={"x-vsr-test-session-id": sid},
        )
        assert (
            resp.status_code == 200
        ), f"expected session {sid!r} to be present, got {resp.status_code}"
        assert resp.json()["body"]["messages"][0]["content"] == sid


@pytest.mark.asyncio
async def test_simulator_rejects_unknown_provider_field(
    client_with_upstream: tuple[httpx.AsyncClient, _UpstreamRecorder],
) -> None:
    client, upstream = client_with_upstream
    response = await client.post(
        "/v1/messages",
        json={
            "model": "provider-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "silently_swallowed": True,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == {
        "type": "invalid_request_error",
        "message": "unknown request field: silently_swallowed",
    }
    assert upstream.requests == []
