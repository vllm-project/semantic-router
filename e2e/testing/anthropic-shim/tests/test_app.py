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
from anthropic_shim.app import create_app


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
