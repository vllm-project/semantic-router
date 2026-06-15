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
async def test_debug_last_request_returns_translated_body_after_messages_post(
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
    # The shim joins the system array before forwarding.
    assert data["body"]["system"] == "You are helpful.\nBe concise."
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
