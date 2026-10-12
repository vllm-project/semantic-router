"""Keep the vLLM fixtures opt-in and faithful to captured wire."""

import json
from pathlib import Path

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.vllm_fixture import MARKER, REASONING_ORDER_MARKER

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
