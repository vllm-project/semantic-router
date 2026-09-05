from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from app import app

IMAGE_GENERATION_BODY: dict[str, Any] = {
    "model": "omni-model",
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "draw a red cat"}],
        }
    ],
    "tool_choice": {"type": "image_generation"},
    "tools": [
        {
            "type": "image_generation",
            "output_format": "png",
            "quality": "high",
            "size": "1024x1024",
        }
    ],
}


@pytest.fixture
def client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://simulator"
    )


def split_sse_events(body: bytes) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.decode().split("\n\n"):
        if not block.strip():
            continue
        event = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if event and data:
            events.append((event, json.loads(data)))
    return events


@pytest.mark.asyncio
async def test_responses_image_generation_non_stream(client):
    response = await client.post("/v1/responses", json=IMAGE_GENERATION_BODY)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    output = payload["output"]
    assert len(output) == 1
    item = output[0]
    assert item["type"] == "image_generation_call"
    assert item["status"] == "completed"
    result = item.get("result", "")
    assert isinstance(result, str) and result
    base64.b64decode(result)  # must be valid base64


@pytest.mark.asyncio
async def test_responses_image_generation_stream(client):
    body = {**IMAGE_GENERATION_BODY, "stream": True}
    response = await client.post("/v1/responses", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = split_sse_events(response.content)
    names = [name for name, _ in events]
    assert names[0] == "response.created"
    assert "response.output_item.added" in names
    assert "response.image_generation_call.in_progress" in names
    assert "response.image_generation_call.generating" in names
    assert "response.image_generation_call.completed" in names
    assert "response.output_item.done" in names
    assert names[-1] == "response.completed"

    partials = [
        payload
        for name, payload in events
        if name == "response.image_generation_call.partial_image"
    ]
    assert len(partials) >= 1
    assert partials[0]["partial_image_index"] == 0
    assert partials[0]["size"] == "1024x1024"
    base64.b64decode(partials[0]["partial_image_b64"])

    added = [
        payload for name, payload in events if name == "response.output_item.added"
    ]
    assert added[0]["item"]["type"] == "image_generation_call"
    assert added[0]["item"]["result"] is None

    done = [payload for name, payload in events if name == "response.output_item.done"]
    assert done[0]["item"]["status"] == "completed"
    base64.b64decode(done[0]["item"]["result"])
