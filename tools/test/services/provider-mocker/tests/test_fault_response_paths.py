from __future__ import annotations

import json
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from fastapi.responses import StreamingResponse
from provider_mocker import ollama_fixture, openrouter_fixture, vllm_fixture
from provider_mocker.app import create_app
from provider_mocker.fault_schedule import (
    FAULT_INJECTED_HEADER,
    FAULT_SCHEDULE_ID_HEADER,
    FAULT_SESSION_ID_HEADER,
    FaultEntry,
    apply_response_fault,
)
from provider_mocker.settings import Settings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "extra_body"),
    [
        ("default", {}),
        (
            "default",
            {"messages": [{"role": "user", "content": openrouter_fixture.MARKER}]},
        ),
        (
            "default",
            {
                "messages": [{"role": "user", "content": vllm_fixture.MARKER}],
                "stop": "STOP",
            },
        ),
        (
            "default",
            {
                "messages": [
                    {"role": "user", "content": vllm_fixture.REASONING_ORDER_MARKER}
                ]
            },
        ),
        ("default", {"messages": [{"role": "user", "content": ollama_fixture.MARKER}]}),
        ("cli", {}),
        ("memory", {}),
        ("hallucination", {}),
        ("toolcall", {}),
    ],
    ids=[
        "plain",
        "openrouter",
        "vllm-stop",
        "vllm-reasoning",
        "ollama",
        "cli",
        "memory",
        "hallucination",
        "toolcall",
    ],
)
@pytest.mark.parametrize(
    "fault", [{"stream_cut_short": True}, {"delay": 0.01}], ids=["cut", "delay"]
)
async def test_faults_apply_after_every_chat_fixture(
    scenario: str, extra_body: dict[str, Any], fault: dict[str, Any]
) -> None:
    app = create_app(
        Settings(
            scenario=scenario, fault_schedule={"faulted": [{"call_index": 0, **fault}]}
        )
    )
    body = {
        "model": "fixture",
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        **extra_body,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        control = await client.post("/v1/chat/completions", json=body)
        observed = await client.post(
            "/v1/chat/completions",
            json=body,
            headers={
                FAULT_SCHEDULE_ID_HEADER: "faulted",
                FAULT_SESSION_ID_HEADER: "run:target:case",
            },
        )
    assert control.status_code == observed.status_code == HTTPStatus.OK
    assert FAULT_INJECTED_HEADER not in control.headers
    assert observed.headers.get(FAULT_INJECTED_HEADER) == "true"
    if fault.get("stream_cut_short"):
        assert 0 < len(observed.content) < len(control.content)
        assert b"[DONE]" not in observed.content
    else:
        assert observed.headers["content-type"] == control.headers["content-type"]
        assert bool(observed.content) == bool(control.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/chat/completions",
            {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("/v1/responses", {"model": "fixture", "input": "hi"}),
        (
            "/v1/messages",
            {
                "model": "fixture",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ),
    ],
)
async def test_buffered_response_cut_is_not_a_false_receipt(
    path: str, body: dict[str, Any]
) -> None:
    app = create_app(
        Settings(fault_schedule={"cut": [{"call_index": 0, "stream_cut_short": True}]})
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        control = await client.post(path, json=body)
        observed = await client.post(
            path,
            json=body,
            headers={
                FAULT_SCHEDULE_ID_HEADER: "cut",
                FAULT_SESSION_ID_HEADER: "cut-once",
            },
        )
    assert observed.headers.get(FAULT_INJECTED_HEADER) == "true"
    assert 0 < len(observed.content) < len(control.content)
    assert int(observed.headers["content-length"]) == len(observed.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
async def test_cut_preserves_prefix_and_closes_split_or_combined_stream(
    chunk_size: int,
) -> None:
    payload = b'data: {"model":"fixture","choices":[]}\n\ndata: [DONE]\n\n'
    closed = False

    async def chunks() -> AsyncIterator[bytes]:
        nonlocal closed
        try:
            for start in range(0, len(payload), chunk_size):
                yield payload[start : start + chunk_size]
        finally:
            closed = True

    response = apply_response_fault(
        response=StreamingResponse(chunks()),
        fault=FaultEntry(call_index=0, stream_cut_short=True),
    )
    assert isinstance(response, StreamingResponse)
    observed = b"".join([chunk async for chunk in response.body_iterator])
    assert payload.startswith(observed)
    assert b'"model":"fixture"' in observed
    assert b"[DONE]" not in observed
    assert closed


@pytest.mark.asyncio
async def test_noop_schedule_does_not_claim_an_injection() -> None:
    app = create_app(Settings(fault_schedule={"noop": [{"call_index": 0, "delay": 0}]}))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            url="/v1/chat/completions",
            json={"model": "fixture", "messages": [{"role": "user", "content": "hi"}]},
            headers={FAULT_SCHEDULE_ID_HEADER: "noop"},
        )
    assert response.status_code == HTTPStatus.OK
    assert FAULT_INJECTED_HEADER not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [{"status": 503}, {"stream_cut_short": True}, {"delay": 0.001}],
    ids=["status", "cut", "delay"],
)
async def test_retry_preserves_one_logical_request_fault(
    fault: dict[str, Any],
) -> None:
    app = create_app(Settings(fault_schedule={"task": [{"call_index": 0, **fault}]}))
    headers = {
        FAULT_SCHEDULE_ID_HEADER: "task",
        FAULT_SESSION_ID_HEADER: "run:target:task",
        "x-vsr-fault-call-index": "0",
    }
    body = {
        "model": "primary",
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post(
            url="/v1/chat/completions", json=body, headers=headers
        )
        retried = await client.post(
            url="/v1/chat/completions",
            json={**body, "model": "fallback"},
            headers=headers,
        )
        next_call = await client.post(
            url="/v1/chat/completions",
            json=body,
            headers={**headers, "x-vsr-fault-call-index": "1"},
        )
    assert first.headers.get(FAULT_INJECTED_HEADER) == "true"
    assert retried.status_code == HTTPStatus.OK
    assert b"[DONE]" in retried.content
    assert retried.headers.get(FAULT_INJECTED_HEADER) == "true"
    receipt = json.loads(retried.headers["x-vsr-fault-receipt"])
    assert receipt == {"call_index": 0, "provider_model": "primary", **fault}
    assert FAULT_INJECTED_HEADER not in next_call.headers


@pytest.mark.asyncio
async def test_retry_does_not_advance_the_next_logical_fault() -> None:
    app = create_app(
        Settings(
            fault_schedule={
                "task": [
                    {"call_index": 0, "status": 503},
                    {"call_index": 2, "status": 502},
                ]
            }
        )
    )
    body = {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}
    headers = {FAULT_SCHEDULE_ID_HEADER: "task", FAULT_SESSION_ID_HEADER: "same-task"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        statuses = []
        for index in (0, 0, 1, 2):
            response = await client.post(
                url="/v1/chat/completions",
                json=body,
                headers={**headers, "x-vsr-fault-call-index": str(index)},
            )
            statuses.append(response.status_code)
    assert statuses == [503, 200, 200, 502]
