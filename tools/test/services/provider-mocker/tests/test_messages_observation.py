"""Native Messages cache, tools, errors, and bounded observation contracts."""

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.provider_boundary import _MAX_REQUEST_STORE_SESSIONS


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://fixture"
    ) as client:
        yield client


async def test_mock_tool_lifecycle_is_native_anthropic_and_deterministic(
    client: httpx.AsyncClient,
) -> None:
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
        "caller": {"type": "direct"},
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


async def test_mock_provider_error_is_native_anthropic_and_deterministic(
    client: httpx.AsyncClient,
) -> None:
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


async def test_mock_incomplete_stream_has_no_success_terminal(
    client: httpx.AsyncClient,
) -> None:
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


async def test_mock_midstream_error_follows_partial_content_without_success_terminal(
    client: httpx.AsyncClient,
) -> None:
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


async def test_cache_usage_synthesised_on_first_then_repeat_request(
    client: httpx.AsyncClient,
) -> None:
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
    assert first_usage["cache_creation_input_tokens"] == 6
    assert first_usage["cache_read_input_tokens"] == 0
    assert second_usage["cache_creation_input_tokens"] == 0
    assert second_usage["cache_read_input_tokens"] == 6


async def test_cache_usage_untouched_when_request_has_no_cache_control(
    client: httpx.AsyncClient,
) -> None:
    payload = {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    response = await client.post("/v1/messages", json=payload)
    usage = response.json()["usage"]
    assert "cache_creation_input_tokens" not in usage
    assert "cache_read_input_tokens" not in usage


async def test_session_isolation_with_distinct_session_headers(
    client: httpx.AsyncClient,
) -> None:
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
    assert first.json()["usage"]["cache_creation_input_tokens"] == 6
    # different session: counts as first request again
    assert second_other_session.json()["usage"]["cache_creation_input_tokens"] == 6


async def test_invalid_json_returns_400(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/v1/messages",
        content=b"not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


async def test_debug_last_request_returns_404_before_any_request(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get(
        "/debug/last-request",
        headers={"x-vsr-test-session-id": "session-new"},
    )
    assert response.status_code == 404


async def test_debug_last_request_returns_native_provider_body_after_messages_post(
    client: httpx.AsyncClient,
) -> None:
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


async def test_debug_last_request_session_via_query_param(
    client: httpx.AsyncClient,
) -> None:
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


async def test_debug_last_request_reflects_most_recent_request(
    client: httpx.AsyncClient,
) -> None:
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


async def test_debug_last_request_session_isolation(
    client: httpx.AsyncClient,
) -> None:
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


async def test_request_store_lru_evicts_oldest_session(
    client: httpx.AsyncClient,
) -> None:
    """Filling _MAX_REQUEST_STORE_SESSIONS+1 sessions evicts the oldest."""
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


async def test_simulator_rejects_unknown_provider_field(
    client: httpx.AsyncClient,
) -> None:
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
