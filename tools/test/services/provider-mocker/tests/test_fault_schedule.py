import time
from http import HTTPStatus

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.fault_schedule import FAULT_INJECTED_HEADER, FaultScheduleTracker
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
