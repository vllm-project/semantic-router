import time
from http import HTTPStatus

import httpx
import pytest
from app import (
    app,
    extract_mock_frame_stall,
    extract_mock_header_delay,
    response_extract_mock_frame_stall,
    response_extract_mock_header_delay,
)
from chat_request import ChatRequest


def test_extract_mock_header_delay():
    # ChatRequest parsing
    req1 = ChatRequest(
        model="test",
        messages=[
            {"role": "user", "content": "hello __mock_header_delay_2.5s__ world"}
        ],
    )
    assert extract_mock_header_delay(req1) == 2.5

    req2 = ChatRequest(
        model="test", messages=[{"role": "user", "content": "regular prompt"}]
    )
    assert extract_mock_header_delay(req2) == 0.0

    # Response API dict parsing
    assert (
        response_extract_mock_header_delay({"input": "__mock_header_delay_1s__ test"})
        == 1.0
    )
    assert response_extract_mock_header_delay({"input": "regular input"}) == 0.0


def test_extract_mock_frame_stall():
    # Parameterized stall duration
    req1 = ChatRequest(
        model="test",
        messages=[{"role": "user", "content": "hello __mock_frame_stall_5s__ world"}],
    )
    assert extract_mock_frame_stall(req1) == 5.0

    # Incomplete stream does NOT trigger frame stall (decoupled)
    req2 = ChatRequest(
        model="test",
        messages=[
            {"role": "user", "content": "hello __mock_incomplete_stream__ world"}
        ],
    )
    assert extract_mock_frame_stall(req2) == 0.0

    req3 = ChatRequest(
        model="test", messages=[{"role": "user", "content": "normal stream"}]
    )
    assert extract_mock_frame_stall(req3) == 0.0

    # Response API dict parsing
    assert (
        response_extract_mock_frame_stall({"input": "__mock_frame_stall_3s__ test"})
        == 3.0
    )
    assert (
        response_extract_mock_frame_stall({"input": "__mock_incomplete_stream__ test"})
        == 0.0
    )
    assert response_extract_mock_frame_stall({"input": "regular input"}) == 0.0


@pytest.mark.asyncio
async def test_chat_completions_controllable_header_delay():
    body = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "__mock_header_delay_0.2s__ test prompt"}
        ],
    }
    start = time.time()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://simulator"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=body)
    elapsed = time.time() - start

    assert resp.status_code == HTTPStatus.OK
    assert elapsed >= 0.18


@pytest.mark.asyncio
async def test_chat_completions_streaming_initial_chunk():
    body = {
        "model": "test-model",
        "stream": True,
        "messages": [{"role": "user", "content": "normal stream"}],
    }
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://simulator"
        ) as client,
        client.stream("POST", "/v1/chat/completions", json=body) as resp,
    ):
        assert resp.status_code == HTTPStatus.OK
        content = ""
        async for chunk in resp.aiter_text():
            content += chunk
        assert "[DONE]" in content


@pytest.mark.asyncio
async def test_chat_completions_streaming_frame_stall():
    body = {
        "model": "test-model",
        "stream": True,
        "messages": [
            {"role": "user", "content": "__mock_frame_stall_0.1s__ test prompt"}
        ],
    }
    start = time.time()
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://simulator"
        ) as client,
        client.stream("POST", "/v1/chat/completions", json=body) as resp,
    ):
        assert resp.status_code == HTTPStatus.OK
        chunks = []
        async for chunk in resp.aiter_text():
            chunks.append(chunk)
    elapsed = time.time() - start

    assert len(chunks) >= 1
    assert any("[DONE]" in c for c in chunks)
    assert elapsed >= 0.08


@pytest.mark.asyncio
async def test_chat_completions_streaming_frame_stall_incomplete():
    body = {
        "model": "test-model",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": "__mock_incomplete_stream__ __mock_frame_stall_0.1s__ test prompt",
            }
        ],
    }
    start = time.time()
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://simulator"
        ) as client,
        client.stream("POST", "/v1/chat/completions", json=body) as resp,
    ):
        assert resp.status_code == HTTPStatus.OK
        chunks = []
        async for chunk in resp.aiter_text():
            chunks.append(chunk)
    elapsed = time.time() - start

    assert len(chunks) >= 1
    assert not any("[DONE]" in c for c in chunks)
    assert elapsed >= 0.08
