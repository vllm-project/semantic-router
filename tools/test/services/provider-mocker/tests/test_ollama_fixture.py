"""Keep the Ollama empty-content fixture opt-in and faithful to captured wire."""

import json

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.ollama_fixture import MARKER


@pytest.mark.parametrize("stream", [False, True])
async def test_ollama_empty_content_tool_fixture(stream):
    request = {
        "model": "mock/ollama-chat",
        "messages": [{"role": "user", "content": MARKER}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
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
            assert len(chunks) == 3
            assert chunks[0]["choices"][0]["delta"]["content"] == ""
            assert chunks[1]["choices"][0]["delta"]["content"] == ""
            assert (
                chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
                == "call_mock_lookup"
            )
            assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        else:
            message = response.json()["choices"][0]["message"]
            assert message["content"] == ""
            assert message["reasoning"]
            assert message["tool_calls"][0]["function"] == {
                "name": "lookup",
                "arguments": '{"query":"weather"}',
            }

        # Ordinary Chat responses never gain Ollama-specific empty content.
        request["messages"][0]["content"] = "ordinary request"
        ordinary = await client.post("/v1/chat/completions", json=request)
        assert ordinary.status_code == 200
        if stream:
            assert "mock/ollama-chat" in ordinary.text
        else:
            assert ordinary.json()["choices"][0]["message"]["content"]
