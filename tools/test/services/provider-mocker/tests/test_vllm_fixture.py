"""Keep the vLLM fixtures opt-in and faithful to captured wire."""

import json
from pathlib import Path

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.chat_request import ChatRequest
from provider_mocker.vllm_fixture import (
    GENERATION_ERROR,
    MARKER,
    MIDSTREAM_ERROR_MARKER,
    REASONING_ORDER_MARKER,
    STREAM_ERROR_MARKER,
    error_stream,
)

CAPTURE = (
    Path(__file__).resolve().parents[5]
    / "src/semantic-router/pkg/protocolcodec/testdata/providers"
    / "vllm-chat-reasoning-and-answer-one-delta-stream.sse"
)


def choice_frames(sse: str) -> list[tuple[str, str | None]]:
    """Each choice frame's delta, as JSON so key order counts, and finish_reason."""
    chunks = [
        json.loads(line.removeprefix("data: "))
        for line in sse.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    return [
        (json.dumps(chunk["choices"][0]["delta"]), chunk["choices"][0]["finish_reason"])
        for chunk in chunks
        if chunk["choices"]
    ]


@pytest.mark.parametrize("stream", [False, True])
async def test_vllm_stop_sequence_fixture(stream):
    request = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": MARKER}],
        "stop": ["CHARLIE"],
        "stream": stream,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://fixture"
    ) as client:
        response = await client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200
        if stream:
            chunks = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            final = chunks[-1]["choices"][0]
            assert final["finish_reason"] == "stop"
            assert final["stop_reason"] == "CHARLIE"
            assert all(
                "stop_reason" not in chunk["choices"][0] for chunk in chunks[:-1]
            )
        else:
            choice = response.json()["choices"][0]
            assert choice["finish_reason"] == "stop"
            assert choice["stop_reason"] == "CHARLIE"
            assert choice["message"]["content"] == "ALPHA BRAVO "

        # Without a stop list, or without the marker, Chat replies stay ordinary.
        for ordinary in (
            {**request, "stop": None},
            {**request, "messages": [{"role": "user", "content": "ordinary request"}]},
        ):
            reply = await client.post("/v1/chat/completions", json=ordinary)
            assert reply.status_code == 200
            assert "stop_reason" not in reply.text


async def test_vllm_reasoning_order_fixture():
    request = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": REASONING_ORDER_MARKER}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://fixture"
    ) as client:
        streamed = await client.post("/v1/chat/completions", json=request)
        assert streamed.status_code == 200
        assert choice_frames(streamed.text) == choice_frames(CAPTURE.read_text())
        assert '"choices":[],"usage"' in streamed.text
        assert streamed.text.endswith("data: [DONE]\n\n")

        no_usage = await client.post(
            "/v1/chat/completions", json={**request, "stream_options": None}
        )
        assert '"usage"' not in no_usage.text

        buffered = await client.post(
            "/v1/chat/completions", json={**request, "stream": False}
        )
        choice = buffered.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["reasoning"] == ' how are you" five words. final.\n'
        assert choice["message"]["content"] == "\n\nHello there, how are you"

        # Without the marker, Chat replies stay ordinary.
        ordinary = {
            **request,
            "messages": [{"role": "user", "content": "ordinary request"}],
        }
        reply = await client.post("/v1/chat/completions", json=ordinary)
        assert reply.status_code == 200
        assert "Hello there, how are you" not in reply.text


# Bytes captured from vLLM 0.30.0 (bad json_schema regex), streamed and buffered.
CAPTURED_STREAM = CAPTURE.with_name("vllm-chat-stream-error-stream.sse").read_text()
CAPTURED_BUFFERED = (
    '{"error":{"message":"Grammar error: regex parse error:\\n    [\\n    ^\\n'
    'error: unclosed character class","type":"BadRequestError","param":null,"code":400}}'
)


async def test_vllm_stream_error_fixture():
    request = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": STREAM_ERROR_MARKER}],
        "stream": True,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://fixture"
    ) as client:
        streamed = await client.post("/v1/chat/completions", json=request)
        assert streamed.status_code == 200
        assert streamed.text == CAPTURED_STREAM

        buffered = await client.post(
            "/v1/chat/completions", json={**request, "stream": False}
        )
        assert buffered.status_code == 400
        assert buffered.text == CAPTURED_BUFFERED

        midstream = await client.post(
            "/v1/chat/completions",
            json={
                **request,
                "messages": [{"role": "user", "content": MIDSTREAM_ERROR_MARKER}],
            },
        )
        events = [
            line.removeprefix("data: ")
            for line in midstream.text.splitlines()
            if line.startswith("data: ")
        ]
        assert midstream.status_code == 200
        assert json.loads(events[1])["choices"][0]["delta"]["content"] == "partial"
        assert json.loads(events[2]) == {"error": GENERATION_ERROR}
        assert events[3:] == ["[DONE]"]


def test_vllm_error_and_done_share_one_write():
    """Each yield is its own write; the error and [DONE] share one so the E2E's
    Chat result does not depend on how Envoy splits the reads."""
    req = ChatRequest(model="m", messages=[{"role": "user", "content": "x"}])
    *_, last = error_stream(req, 0, GENERATION_ERROR, content="partial")
    assert last.startswith('data: {"error": ')
    assert last.endswith("\n\ndata: [DONE]\n\n")
