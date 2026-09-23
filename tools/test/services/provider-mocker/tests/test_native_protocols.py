"""Native endpoint contracts which survive fixture consolidation."""

import base64
import json
import struct
import zlib

import httpx
import pytest
from provider_mocker.app import create_app


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app()), base_url="http://fixture"
    ) as client:
        yield client


def events(text):
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def body(text="hello", **kwargs):
    return {
        "model": "native-test",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": text}],
        **kwargs,
    }


def streamed_text(payloads):
    return "".join(
        p.get("delta", {}).get("text", "")
        for p in payloads
        if p["type"] == "content_block_delta"
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_messages_probe_and_structured_schema(client, stream):
    response = await client.post(
        "/v1/messages", json=body("__mock_protocol_matrix__", stream=stream)
    )
    assert response.status_code == 200
    text = (
        streamed_text(events(response.text))
        if stream
        else response.json()["content"][0]["text"]
    )
    assert json.loads(text) == {"protocol": "anthropic_messages"}
    output_format = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
    }
    response = await client.post(
        "/v1/messages",
        json=body(
            "__mock_structured_output__",
            stream=stream,
            output_config={"format": output_format},
        ),
    )
    assert response.status_code == 200
    text = (
        streamed_text(events(response.text))
        if stream
        else response.json()["content"][0]["text"]
    )
    assert json.loads(text) == {
        "mock": "provider-mocker",
        "structured_output": output_format,
    }


async def test_messages_tools_stream_preserves_arguments_and_terminal_order(client):
    response = await client.post(
        "/v1/messages",
        json=body(
            "__mock_tool_call__",
            stream=True,
            tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
        ),
    )
    payloads = events(response.text)
    assert payloads[0]["type"] == "message_start"
    tool = next(
        p["content_block"] for p in payloads if p["type"] == "content_block_start"
    )
    assert tool["id"] == "call_mock_lookup" and tool["name"] == "lookup"
    assert tool["caller"] == {"type": "direct"}
    fragments = [
        p["delta"]["partial_json"]
        for p in payloads
        if p["type"] == "content_block_delta"
    ]
    assert len(fragments) > 1
    assert json.loads("".join(fragments)) == {"query": "weather"}
    assert [p["type"] for p in payloads[-3:]] == [
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert payloads[-2]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "stops,expected,reason",
    [
        ([" "], "Hello", "stop_sequence"),
        (["never-occurs"], "Hello from provider-mocker.", "end_turn"),
    ],
)
async def test_stop_sequence_matches_generated_text(
    client, stream, stops, expected, reason
):
    response = await client.post(
        "/v1/messages", json=body(stream=stream, stop_sequences=stops)
    )
    assert response.status_code == 200
    if stream:
        payloads = events(response.text)
        assert streamed_text(payloads) == expected
        stop = payloads[-2]["delta"]
    else:
        stop = response.json()
        assert stop["content"][0]["text"] == expected
    assert stop["stop_reason"] == reason
    assert stop["stop_sequence"] == (stops[0] if reason == "stop_sequence" else None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", True),
        ("max_tokens", -1),
        ("messages", []),
        ("model", ""),
        ("stream", "true"),
    ],
)
async def test_invalid_messages_envelope_is_anthropic_error(client, field, value):
    response = await client.post("/v1/messages", json=body(**{field: value}))
    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_images_are_valid_deterministic_png_and_request_is_observable(client):
    request = {
        "model": "diffusion",
        "prompt": "red pixel",
        "n": 2,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    headers = {"x-vsr-test-session-id": "image-session"}
    first = await client.post("/v1/images/generations", json=request, headers=headers)
    second = await client.post("/v1/images/generations", json=request, headers=headers)
    assert first.status_code == 200 and first.json() == second.json()
    assert len(first.json()["data"]) == 2
    png = base64.b64decode(first.json()["data"][0]["b64_json"], validate=True)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    offset, kinds = 8, []
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind, payload = (
            png[offset + 4 : offset + 8],
            png[offset + 8 : offset + 8 + length],
        )
        crc = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])[0]
        assert crc == zlib.crc32(kind + payload) & 0xFFFFFFFF
        kinds.append(kind)
        if kind == b"IHDR":
            assert struct.unpack(">II", payload[:8]) == (1, 1)
        offset += length + 12
    assert kinds == [b"IHDR", b"IDAT", b"IEND"]
    observed = await client.get("/debug/last-request", headers=headers)
    assert observed.json()["body"] == request


@pytest.mark.parametrize(
    "patch",
    [
        {"stream": True},
        {"response_format": "url"},
        {"n": 0},
        {"n": True},
        {"n": 5},
        {"prompt": ""},
        {"input_image_mask": "unsupported"},
    ],
)
async def test_images_reject_unsupported_contracts(client, patch):
    response = await client.post(
        "/v1/images/generations", json={"prompt": "draw", **patch}
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert (
        await client.post("/v1/images/edits", json={"prompt": "draw"})
    ).status_code == 404


async def test_observation_excludes_credentials(client):
    headers = {
        "Authorization": "Bearer fixture-secret",
        "x-api-key": "fixture-secret",
        "x-vsr-e2e-trace": "kept",
        "x-vsr-test-session-id": "safe-observation",
    }
    await client.post("/v1/messages", json=body(), headers=headers)
    observed = (await client.get("/debug/last-request", headers=headers)).json()
    assert observed["headers"] == {
        "x-vsr-e2e-trace": "kept",
        "x-vsr-test-session-id": "safe-observation",
    }
    assert "fixture-secret" not in json.dumps(observed)


@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
@pytest.mark.parametrize(
    "marker",
    [
        "__mock_provider_error__",
        "__mock_incomplete_stream__",
        "__mock_midstream_error__",
    ],
)
async def test_native_failure_scenarios_keep_error_and_completion_boundaries(
    client, protocol, marker
):
    request = {"model": "native-test", "stream": True}
    if protocol == "responses":
        request["input"] = marker
        path, success = "/v1/responses", "response.completed"
    else:
        request["messages"] = [{"role": "user", "content": marker}]
        path, success = "/v1/chat/completions", "data: [DONE]"
        if protocol == "messages":
            request["max_tokens"] = 32
            path, success = "/v1/messages", "event: message_stop"
    response = await client.post(path, json=request)
    if marker == "__mock_provider_error__":
        assert response.status_code == 429
        assert response.json()["error"]["type"] == "rate_limit_error"
        return
    assert response.status_code == 200
    assert success not in response.text
    if marker == "__mock_midstream_error__":
        assert "partial" in response.text
        assert "mock provider stream failed" in response.text
    else:
        assert "error" not in response.text
