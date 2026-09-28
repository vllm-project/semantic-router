"""Memory scenarios consume the text view of native Chat content parts."""

import json

import httpx
import pytest
from provider_mocker.app import create_app
from provider_mocker.settings import Settings


def text_parts(*values):
    return [{"type": "text", "text": value} for value in values]


@pytest.mark.parametrize("stream", [False, True])
async def test_memory_echo_accepts_codec_text_parts_and_null_tool_call_content(stream):
    request = {
        "model": "qwen3",
        "stream": stream,
        "messages": [
            {
                "role": "system",
                "content": text_parts("You are MoM, a helpful assistant with memory."),
            },
            {
                "role": "developer",
                "content": text_parts("Retrieved memory:", "Favorite color: purple"),
            },
            {
                "role": "user",
                "content": text_parts(
                    "Please remember this: My favorite color is purple"
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "lookup",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "lookup", "content": "stored"},
        ],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(Settings(scenario="memory"))),
        base_url="http://fixture",
    ) as client:
        response = await client.post("/v1/chat/completions", json=request)
        assert response.status_code == 200
        if stream:
            events = [
                line.removeprefix("data: ")
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            assert events[-1] == "[DONE]"
            content = "".join(
                json.loads(event)["choices"][0]["delta"].get("content", "")
                for event in events[:-1]
                if json.loads(event)["choices"]
            )
        else:
            content = response.json()["choices"][0]["message"]["content"]
        assert (
            content
            == "[system]: You are MoM, a helpful assistant with memory.\n[developer]: Retrieved memory:\nFavorite color: purple\n[user]: Please remember this: My favorite color is purple\n[assistant]: \n[tool]: stored"
        )
        observed = (await client.get("/debug/last-request")).json()
        assert observed["body"] == request


@pytest.mark.parametrize(
    "messages,expected",
    [
        (
            [
                {
                    "role": "system",
                    "content": text_parts(
                        "You are a memory extraction system. Example: budget is $5000 at MIT"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        *text_parts("My favorite color is purple"),
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/tesla.png"},
                        },
                    ],
                },
            ],
            '[{"type": "semantic", "content": "User\'s favorite color is purple"}]',
        ),
        (
            [
                {"role": "system", "content": text_parts("You are a query rewriter")},
                {
                    "role": "user",
                    "content": text_parts(
                        "History:\n[user]: trip to Japan",
                        "Query: Which hotel?\nRewritten query:",
                    ),
                },
            ],
            "Which hotel? (context: trip to Japan)",
        ),
        (
            [
                {"role": "system", "content": text_parts("You are a query rewriter")},
                {"role": "user", "content": text_parts("What is my favorite color?")},
            ],
            "What is my favorite color?",
        ),
    ],
)
async def test_memory_extraction_and_rewriting_accept_text_parts(messages, expected):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(Settings(scenario="memory"))),
        base_url="http://fixture",
    ) as client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "qwen3", "messages": messages}
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == expected
