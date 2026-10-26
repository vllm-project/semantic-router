"""The OpenRouter fixture enforces continuity across a tool turn."""

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.openrouter_fixture import (
    ANSWER,
    REASONING_DETAILS,
    REASONING_MARKER,
    TOOL_CALL,
)
from provider_mocker.settings import Settings


@pytest.mark.parametrize("streamed", [False, True])
async def test_openrouter_tool_turn_requires_original_reasoning_blocks(streamed):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(Settings())),
        base_url="http://fixture",
    ) as client:
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": REASONING_MARKER}],
            "stream": streamed,
        }
        first = await client.post("/v1/chat/completions", json=request)
        assert first.status_code == 200
        if streamed:
            assert '"reasoning_details"' in first.text
            assert '"finish_reason": "tool_calls"' in first.text
        else:
            assert (
                first.json()["choices"][0]["message"]["reasoning_details"]
                == REASONING_DETAILS
            )
        assistant = {
            "role": "assistant",
            "content": None,
            "reasoning_details": REASONING_DETAILS,
            "tool_calls": [TOOL_CALL],
        }
        request["messages"] += [
            assistant,
            {"role": "tool", "tool_call_id": TOOL_CALL["id"], "content": "result"},
        ]
        second = await client.post("/v1/chat/completions", json=request)
        assert second.status_code == 200
        assert ANSWER in second.text
        assistant["reasoning_details"] = list(reversed(REASONING_DETAILS))
        rejected = await client.post("/v1/chat/completions", json=request)
        assert rejected.status_code == 400
