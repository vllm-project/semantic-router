import time
from http import HTTPStatus

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.fault_schedule import (
    FAULT_INJECTED_HEADER,
    FAULT_SCHEDULE_ID_HEADER,
    FAULT_SESSION_ID_HEADER,
    FaultScheduleTracker,
    get_fault_keys,
)
from provider_mocker.settings import Settings


def test_tracker_parsing_and_isolation():
    schedule = {
        "task-alpha": [
            {"call_index": 1, "status": 503},
            {"call_index": 2, "delay": 0.5},
        ],
        "task-beta": [
            {"call_index": 0, "stream_cut_short": True},
        ],
    }
    tracker = FaultScheduleTracker(schedule)

    # Call 0 for task-alpha: no fault
    idx0, f0 = tracker.record_call_and_match("task-alpha")
    assert idx0 == 0
    assert f0 is None

    # Call 1 for task-alpha: status 503
    idx1, f1 = tracker.record_call_and_match("task-alpha")
    assert idx1 == 1
    assert f1 is not None and f1.status == 503

    # Call 2 for task-alpha: delay 0.5
    idx2, f2 = tracker.record_call_and_match("task-alpha")
    assert idx2 == 2
    assert f2 is not None and f2.delay == 0.5

    # Other task (unaffected):
    idx_unaffected, f_unaffected = tracker.record_call_and_match("task-unaffected")
    assert idx_unaffected == 0
    assert f_unaffected is None

    idx_unaffected2, f_unaffected2 = tracker.record_call_and_match("task-unaffected")
    assert idx_unaffected2 == 1
    assert f_unaffected2 is None

    # Separated declarative schedule key and execution counter key:
    # Multiple targets sharing the same schedule "task-beta" both receive call_index=0 fault
    idx_t1, f_t1 = tracker.record_call_and_match("task-beta", "run1:target1:task-beta")
    assert idx_t1 == 0
    assert f_t1 is not None and f_t1.stream_cut_short is True

    idx_t2, f_t2 = tracker.record_call_and_match("task-beta", "run1:target2:task-beta")
    assert idx_t2 == 0
    assert f_t2 is not None and f_t2.stream_cut_short is True


@pytest.mark.asyncio
async def test_chat_completions_status_fault_at_chosen_call_index():
    schedule = {"task-status": [{"call_index": 1, "status": 503}]}
    app = create_app(Settings(fault_schedule=schedule))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers_faulted = {"x-vsr-test-session-id": "task-status"}
        headers_unaffected = {"x-vsr-test-session-id": "task-other"}

        body = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "hello"}],
        }

        # Call 0 on faulted task: OK
        res0 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_faulted
        )
        assert res0.status_code == HTTPStatus.OK
        assert FAULT_INJECTED_HEADER not in res0.headers

        # Call 0 on unaffected task: OK
        res_u0 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_unaffected
        )
        assert res_u0.status_code == HTTPStatus.OK
        assert FAULT_INJECTED_HEADER not in res_u0.headers

        # Call 1 on faulted task: 503 Injected
        res1 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_faulted
        )
        assert res1.status_code == 503
        assert res1.headers.get(FAULT_INJECTED_HEADER) == "true"
        data1 = res1.json()
        assert data1["error"]["type"] == "fault_schedule_injected"

        # Call 1 on unaffected task: STILL OK
        res_u1 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_unaffected
        )
        assert res_u1.status_code == HTTPStatus.OK
        assert FAULT_INJECTED_HEADER not in res_u1.headers

        # Call 2 on faulted task: OK again
        res2 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_faulted
        )
        assert res2.status_code == HTTPStatus.OK
        assert FAULT_INJECTED_HEADER not in res2.headers


@pytest.mark.asyncio
async def test_chat_completions_delay_fault_at_chosen_call_index():
    schedule = {"task-delay": [{"call_index": 1, "delay": 0.2}]}
    app = create_app(Settings(fault_schedule=schedule))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers_faulted = {"x-vsr-fault-schedule-id": "task-delay"}
        headers_unaffected = {"x-vsr-fault-schedule-id": "task-clean"}
        body = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "hello"}],
        }

        # Call 0 on faulted task: fast
        start = time.time()
        res0 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_faulted
        )
        elapsed0 = time.time() - start
        assert res0.status_code == HTTPStatus.OK
        assert elapsed0 < 0.15
        assert FAULT_INJECTED_HEADER not in res0.headers

        # Call 0 on unaffected task: fast
        start = time.time()
        res_u0 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_unaffected
        )
        elapsed_u0 = time.time() - start
        assert res_u0.status_code == HTTPStatus.OK
        assert elapsed_u0 < 0.15
        assert FAULT_INJECTED_HEADER not in res_u0.headers

        # Call 1 on faulted task: delayed
        start = time.time()
        res1 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_faulted
        )
        elapsed1 = time.time() - start
        assert res1.status_code == HTTPStatus.OK
        assert elapsed1 >= 0.18
        assert res1.headers.get(FAULT_INJECTED_HEADER) == "true"

        # Call 1 on unaffected task: fast
        start = time.time()
        res_u1 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_unaffected
        )
        elapsed_u1 = time.time() - start
        assert res_u1.status_code == HTTPStatus.OK
        assert elapsed_u1 < 0.15
        assert FAULT_INJECTED_HEADER not in res_u1.headers

        # Call 2 on faulted task: OK again (fast)
        start = time.time()
        res2 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_faulted
        )
        elapsed2 = time.time() - start
        assert res2.status_code == HTTPStatus.OK
        assert elapsed2 < 0.15
        assert FAULT_INJECTED_HEADER not in res2.headers


@pytest.mark.asyncio
async def test_chat_completions_stream_cut_short_at_chosen_call_index():
    schedule = {"task-stream": [{"call_index": 1, "stream_cut_short": True}]}
    app = create_app(Settings(fault_schedule=schedule))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers_faulted = {"x-vsr-test-session-id": "task-stream"}
        headers_unaffected = {"x-vsr-test-session-id": "task-clean"}
        body = {
            "model": "openai/gpt-oss-20b",
            "stream": True,
            "messages": [{"role": "user", "content": "tell me a long story"}],
        }

        # Call 0 on faulted task: complete stream with [DONE]
        async with client.stream(
            "POST", "/v1/chat/completions", json=body, headers=headers_faulted
        ) as resp:
            assert FAULT_INJECTED_HEADER not in resp.headers
            content0 = "".join([c async for c in resp.aiter_text()])
            assert "[DONE]" in content0

        # Call 0 on unaffected task: complete stream with [DONE]
        async with client.stream(
            "POST", "/v1/chat/completions", json=body, headers=headers_unaffected
        ) as resp:
            assert FAULT_INJECTED_HEADER not in resp.headers
            content_u0 = "".join([c async for c in resp.aiter_text()])
            assert "[DONE]" in content_u0

        # Call 1 on faulted task: stream cut short without [DONE]
        async with client.stream(
            "POST", "/v1/chat/completions", json=body, headers=headers_faulted
        ) as resp:
            assert resp.headers.get(FAULT_INJECTED_HEADER) == "true"
            content1 = "".join([c async for c in resp.aiter_text()])
            assert "[DONE]" not in content1
            assert len(content1) > 0

        # Call 1 on unaffected task: complete stream with [DONE]
        async with client.stream(
            "POST", "/v1/chat/completions", json=body, headers=headers_unaffected
        ) as resp:
            assert FAULT_INJECTED_HEADER not in resp.headers
            content_u1 = "".join([c async for c in resp.aiter_text()])
            assert "[DONE]" in content_u1

        # Call 2 on faulted task: complete stream again with [DONE]
        async with client.stream(
            "POST", "/v1/chat/completions", json=body, headers=headers_faulted
        ) as resp:
            assert FAULT_INJECTED_HEADER not in resp.headers
            content2 = "".join([c async for c in resp.aiter_text()])
            assert "[DONE]" in content2


@pytest.mark.asyncio
async def test_responses_api_and_messages_fault_injection():
    schedule = {
        "session-responses": [
            {"call_index": 0, "status": 502},
            {"call_index": 1, "stream_cut_short": True},
        ]
    }
    app = create_app(Settings(fault_schedule=schedule))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"x-vsr-test-session-id": "session-responses"}

        # Responses API: Call 0 -> 502
        res0 = await client.post(
            "/v1/responses", json={"input": "test"}, headers=headers
        )
        assert res0.status_code == 502
        assert res0.headers.get(FAULT_INJECTED_HEADER) == "true"

        # Responses API: Call 1 -> stream cut short
        async with client.stream(
            "POST",
            "/v1/responses",
            json={"input": "test", "stream": True},
            headers=headers,
        ) as resp:
            assert resp.headers.get(FAULT_INJECTED_HEADER) == "true"
            content = "".join([c async for c in resp.aiter_text()])
            assert "response.completed" not in content

    # Test Messages API
    schedule_msg = {"session-messages": [{"call_index": 0, "status": 500}]}
    app_msg = create_app(Settings(fault_schedule=schedule_msg))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_msg), base_url="http://test"
    ) as client:
        res = await client.post(
            "/v1/messages",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "claude-3",
                "max_tokens": 16,
            },
            headers={"x-vsr-test-session-id": "session-messages"},
        )
        assert res.status_code == 500
        assert res.headers.get(FAULT_INJECTED_HEADER) == "true"
        data = res.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "fault_schedule_injected"


@pytest.mark.asyncio
async def test_dynamic_fault_schedule_endpoint():
    app = create_app(Settings())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Initial: no faults
        res = await client.get("/debug/fault-schedule")
        assert res.status_code == 200
        assert res.json()["stats"]["keys"] == []

        # Set schedule via API
        set_res = await client.post(
            "/debug/fault-schedule",
            json={"dyn-task": [{"call_index": 0, "status": 504}]},
        )
        assert set_res.status_code == 200

        # Post chat completion with dyn-task -> call 0 triggers 504
        chat_res = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-vsr-test-session-id": "dyn-task"},
        )
        assert chat_res.status_code == 504
        assert chat_res.headers.get(FAULT_INJECTED_HEADER) == "true"

        # Verify call_counts recorded
        stats_before = await client.get("/debug/fault-schedule")
        assert stats_before.status_code == 200
        assert stats_before.json()["stats"]["call_counts"].get("dyn-task") == 1

        # Subsequent call before reset: call 1 -> OK 200
        chat_res_ok = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi again"}],
            },
            headers={"x-vsr-test-session-id": "dyn-task"},
        )
        assert chat_res_ok.status_code == HTTPStatus.OK
        assert FAULT_INJECTED_HEADER not in chat_res_ok.headers

        # Reset call counts
        reset_res = await client.post("/debug/fault-schedule/reset")
        assert reset_res.status_code == 200

        # Verify call counts cleared in stats
        stats_after = await client.get("/debug/fault-schedule")
        assert stats_after.status_code == 200
        assert stats_after.json()["stats"]["call_counts"] == {}

        # After reset: call count starts back at 0 -> call 0 triggers 504 again
        chat_res_reset = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi after reset"}],
            },
            headers={"x-vsr-test-session-id": "dyn-task"},
        )
        assert chat_res_reset.status_code == 504
        assert chat_res_reset.headers.get(FAULT_INJECTED_HEADER) == "true"


def test_get_fault_keys_header_order_precedence():
    # Explicit fault-session header must have precedence regardless of dict order
    keys1 = get_fault_keys(
        {"x-session-id": "common", FAULT_SESSION_ID_HEADER: "explicit"}
    )
    assert keys1 == ("explicit", "explicit")

    keys2 = get_fault_keys(
        {FAULT_SESSION_ID_HEADER: "explicit", "x-session-id": "common"}
    )
    assert keys2 == ("explicit", "explicit")

    # With schedule ID present:
    keys3 = get_fault_keys(
        {
            FAULT_SCHEDULE_ID_HEADER: "sched",
            "x-session-id": "common",
            FAULT_SESSION_ID_HEADER: "explicit",
        }
    )
    assert keys3 == ("sched", "explicit")

    keys4 = get_fault_keys(
        {
            FAULT_SCHEDULE_ID_HEADER: "sched",
            FAULT_SESSION_ID_HEADER: "explicit",
            "x-session-id": "common",
        }
    )
    assert keys4 == ("sched", "explicit")

    # Fallback to legacy x-vsr-test-session-id when explicit is absent
    keys5 = get_fault_keys(
        {"x-session-id": "common", "x-vsr-test-session-id": "legacy"}
    )
    assert keys5 == ("legacy", "legacy")

    keys6 = get_fault_keys(
        {"x-vsr-test-session-id": "legacy", "x-session-id": "common"}
    )
    assert keys6 == ("legacy", "legacy")

    # Fallback to ordinary x-session-id when both are absent
    keys7 = get_fault_keys({"x-session-id": "common"})
    assert keys7 == ("common", "common")


@pytest.mark.asyncio
async def test_asgi_header_order_precedence_with_shared_session():
    # Schedule has status 503 at call_index=0 for schedule "case-alpha"
    schedule = {"case-alpha": [{"call_index": 0, "status": 503}]}
    app = create_app(Settings(fault_schedule=schedule))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "hello"}],
        }

        # Target 1: x-session-id comes after x-vsr-fault-session-id
        headers_target1 = [
            ("x-vsr-fault-schedule-id", "case-alpha"),
            ("x-vsr-fault-session-id", "run-1:target-1:case-alpha"),
            ("x-session-id", "shared-session"),
        ]
        res1 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_target1
        )
        assert res1.status_code == 503
        assert res1.headers.get(FAULT_INJECTED_HEADER) == "true"

        # Target 2: x-vsr-fault-session-id comes after x-session-id
        headers_target2 = [
            ("x-vsr-fault-schedule-id", "case-alpha"),
            ("x-session-id", "shared-session"),
            ("x-vsr-fault-session-id", "run-1:target-2:case-alpha"),
        ]
        res2 = await client.post(
            "/v1/chat/completions", json=body, headers=headers_target2
        )
        # Target 2 has its own explicit fault session ID, so its call index is 0 and it receives 503
        assert res2.status_code == 503
        assert res2.headers.get(FAULT_INJECTED_HEADER) == "true"

        # Subsequent call on target 1 (call index 1): 200 OK
        res1_next = await client.post(
            "/v1/chat/completions", json=body, headers=headers_target1
        )
        assert res1_next.status_code == HTTPStatus.OK
        assert FAULT_INJECTED_HEADER not in res1_next.headers

        # Subsequent call on target 2 (call index 1): 200 OK
        res2_next = await client.post(
            "/v1/chat/completions", json=body, headers=headers_target2
        )
        assert res2_next.status_code == HTTPStatus.OK
        assert FAULT_INJECTED_HEADER not in res2_next.headers


@pytest.mark.asyncio
async def test_chat_completions_tool_stream_cut_short_and_receipt():
    schedule = {"task-tool-stream": [{"call_index": 0, "stream_cut_short": True}]}
    app = create_app(Settings(fault_schedule=schedule))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"x-vsr-test-session-id": "task-tool-stream"}
        body = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "__mock_tool_call__"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                }
            ],
            "stream": True,
        }

        # Call 0: tool stream cut short, receipt present, no [DONE]
        async with client.stream(
            "POST", "/v1/chat/completions", json=body, headers=headers
        ) as resp:
            assert resp.status_code == HTTPStatus.OK
            assert resp.headers.get(FAULT_INJECTED_HEADER) == "true"
            chunks = "".join([c async for c in resp.aiter_text()])
            assert "call_mock_lookup" in chunks
            assert "[DONE]" not in chunks

        # Call 1: clean tool stream with [DONE], no receipt
        async with client.stream(
            "POST", "/v1/chat/completions", json=body, headers=headers
        ) as resp:
            assert resp.status_code == HTTPStatus.OK
            assert FAULT_INJECTED_HEADER not in resp.headers
            chunks = "".join([c async for c in resp.aiter_text()])
            assert "call_mock_lookup" in chunks
            assert "[DONE]" in chunks


@pytest.mark.asyncio
async def test_chat_completions_tool_delay_and_receipt():
    schedule = {"task-tool-delay": [{"call_index": 0, "delay": 0.2}]}
    app = create_app(Settings(fault_schedule=schedule))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"x-vsr-test-session-id": "task-tool-delay"}
        body = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "__mock_tool_call__"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                }
            ],
            "stream": False,
        }

        start = time.time()
        resp = await client.post("/v1/chat/completions", json=body, headers=headers)
        elapsed = time.time() - start
        assert resp.status_code == HTTPStatus.OK
        assert elapsed >= 0.18
        assert resp.headers.get(FAULT_INJECTED_HEADER) == "true"
        data = resp.json()
        assert (
            data["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
            == "lookup"
        )

        # Call 1: clean, no receipt, fast
        start = time.time()
        resp1 = await client.post("/v1/chat/completions", json=body, headers=headers)
        elapsed1 = time.time() - start
        assert resp1.status_code == HTTPStatus.OK
        assert elapsed1 < 0.15
        assert FAULT_INJECTED_HEADER not in resp1.headers


@pytest.mark.asyncio
async def test_responses_tool_stream_cut_short_and_receipt():
    schedule = {"resp-tool-stream": [{"call_index": 0, "stream_cut_short": True}]}
    app = create_app(Settings(fault_schedule=schedule))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"x-vsr-test-session-id": "resp-tool-stream"}
        body = {
            "model": "native-test",
            "input": "__mock_tool_call__",
            "tools": [{"type": "function", "name": "lookup"}],
            "stream": True,
        }

        # Call 0: tool stream cut short, receipt present, no response.completed
        async with client.stream(
            "POST", "/v1/responses", json=body, headers=headers
        ) as resp:
            assert resp.status_code == HTTPStatus.OK
            assert resp.headers.get(FAULT_INJECTED_HEADER) == "true"
            chunks = "".join([c async for c in resp.aiter_text()])
            assert "function_call" in chunks
            assert "response.completed" not in chunks

        # Call 1: clean tool stream with response.completed, no receipt
        async with client.stream(
            "POST", "/v1/responses", json=body, headers=headers
        ) as resp:
            assert resp.status_code == HTTPStatus.OK
            assert FAULT_INJECTED_HEADER not in resp.headers
            chunks = "".join([c async for c in resp.aiter_text()])
            assert "function_call" in chunks
            assert "response.completed" in chunks


@pytest.mark.asyncio
async def test_responses_tool_delay_and_receipt():
    schedule = {"resp-tool-delay": [{"call_index": 0, "delay": 0.2}]}
    app = create_app(Settings(fault_schedule=schedule))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"x-vsr-test-session-id": "resp-tool-delay"}
        body = {
            "model": "native-test",
            "input": "__mock_tool_call__",
            "tools": [{"type": "function", "name": "lookup"}],
            "stream": False,
        }

        start = time.time()
        resp = await client.post("/v1/responses", json=body, headers=headers)
        elapsed = time.time() - start
        assert resp.status_code == HTTPStatus.OK
        assert elapsed >= 0.18
        assert resp.headers.get(FAULT_INJECTED_HEADER) == "true"
        data = resp.json()
        assert data["output"][0]["type"] == "function_call"
        assert data["output"][0]["name"] == "lookup"

        # Call 1: clean, fast, no receipt
        start = time.time()
        resp1 = await client.post("/v1/responses", json=body, headers=headers)
        elapsed1 = time.time() - start
        assert resp1.status_code == HTTPStatus.OK
        assert elapsed1 < 0.15
        assert FAULT_INJECTED_HEADER not in resp1.headers
