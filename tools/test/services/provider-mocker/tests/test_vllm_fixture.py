"""Keep the vLLM stop-sequence fixture opt-in and faithful to captured wire."""

import json

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.vllm_fixture import MARKER


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
