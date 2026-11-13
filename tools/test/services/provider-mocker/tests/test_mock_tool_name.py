"""A tool-call marker can name the tool the fixture calls, on every native wire."""

import httpx
import pytest
from provider_mocker.app import create_app

PATHS = ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
NAMED_TEXT = "__mock_tool_name:mcp__echo__echo:__\n__mock_tool_call__"


def tool_request(path: str, text: str) -> dict:
    if path == "/v1/chat/completions":
        return {
            "model": "m",
            "messages": [{"role": "user", "content": text}],
            "tools": [{"type": "function", "function": {"name": "mcp__echo__echo"}}],
        }
    if path == "/v1/responses":
        return {
            "model": "m",
            "input": text,
            "tools": [{"type": "function", "name": "mcp__echo__echo"}],
        }
    return {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": text}],
        "tools": [{"name": "mcp__echo__echo", "input_schema": {"type": "object"}}],
    }


def buffered_tool_name(path: str, body: dict) -> str:
    if path == "/v1/chat/completions":
        return body["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    if path == "/v1/responses":
        return body["output"][0]["name"]
    return body["content"][0]["name"]


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://fixture"
    ) as client:
        yield client


@pytest.mark.parametrize("path", PATHS)
async def test_named_tool_call_buffered(client: httpx.AsyncClient, path: str) -> None:
    response = await client.post(path, json=tool_request(path, NAMED_TEXT))
    assert buffered_tool_name(path, response.json()) == "mcp__echo__echo"


@pytest.mark.parametrize("path", PATHS)
async def test_named_tool_call_streamed(client: httpx.AsyncClient, path: str) -> None:
    body = {**tool_request(path, NAMED_TEXT), "stream": True}
    response = await client.post(path, json=body)
    text = response.text.replace('"name":"', '"name": "')
    assert '"name": "mcp__echo__echo"' in text
    assert '"lookup"' not in text


@pytest.mark.parametrize("path", PATHS)
async def test_tool_call_without_a_name_stays_lookup(
    client: httpx.AsyncClient, path: str
) -> None:
    response = await client.post(path, json=tool_request(path, "__mock_tool_call__"))
    assert buffered_tool_name(path, response.json()) == "lookup"
