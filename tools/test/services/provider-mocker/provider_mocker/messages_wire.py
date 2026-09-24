"""Native Anthropic Messages fixtures; no production codec or upstream adapter."""

import json
from collections.abc import Iterator
from typing import Any


def contains_text(value: Any, marker: str) -> bool:
    if isinstance(value, str):
        return marker in value
    if isinstance(value, list):
        return any(contains_text(item, marker) for item in value)
    if isinstance(value, dict):
        return any(contains_text(item, marker) for item in value.values())
    return False


def has_tool_result(body: dict) -> bool:
    calls = set()
    for message in body["messages"]:
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                calls.add(block.get("id"))
            elif (
                block.get("type") == "tool_result" and block.get("tool_use_id") in calls
            ):
                return True
    return False


def build_message(body: dict) -> dict:
    content = [{"type": "text", "text": "Hello from provider-mocker."}]
    reason, matched = "end_turn", None
    if has_tool_result(body):
        content[0]["text"] = "tool result accepted"
    elif contains_text(body["messages"], "__mock_tool_call__") and body.get("tools"):
        content = [
            {
                "type": "tool_use",
                "id": "call_mock_lookup",
                "name": "lookup",
                "input": {"query": "weather"},
                "caller": {"type": "direct"},
            }
        ]
        reason = "tool_use"
    elif contains_text(body["messages"], "__mock_protocol_matrix__"):
        content[0]["text"] = '{"protocol":"anthropic_messages"}'
    elif contains_text(body["messages"], "__mock_structured_output__"):
        config = body.get("output_config")
        output = config.get("format") if isinstance(config, dict) else None
        if isinstance(output, dict) and output.get("type") == "json_schema":
            content[0]["text"] = json.dumps(
                {"mock": "provider-mocker", "structured_output": output},
                separators=(",", ":"),
            )
    if reason == "end_turn":
        text = content[0]["text"]
        stops = [
            (text.find(stop), stop)
            for stop in (body.get("stop_sequences") or [])
            if stop and stop in text
        ]
        if stops:
            position, matched = min(stops)
            content[0]["text"] = text[:position]
            reason = "stop_sequence"
    return {
        "id": "msg_provider_fixture",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": content,
        "stop_reason": reason,
        "stop_sequence": matched,
        "usage": {"input_tokens": 6, "output_tokens": 3},
    }


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def stream_message(response: dict, failure: str = "") -> Iterator[str]:
    start = {
        **response,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {**response["usage"], "output_tokens": 0},
    }
    yield sse("message_start", {"type": "message_start", "message": start})
    for index, block in enumerate(response["content"]):
        tool = block["type"] == "tool_use"
        initial = {**block, "input": {}} if tool else {"type": "text", "text": ""}
        yield sse(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": initial},
        )
        text = (
            json.dumps(block["input"], separators=(",", ":")) if tool else block["text"]
        )
        if failure:
            text = "partial"
        for offset in range(0, len(text), 9):
            delta = (
                {"type": "input_json_delta", "partial_json": text[offset : offset + 9]}
                if tool
                else {"type": "text_delta", "text": text[offset : offset + 9]}
            )
            yield sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
            )
        if failure:
            if failure == "error":
                yield sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "overloaded_error",
                            "message": "mock provider stream failed",
                        },
                    },
                )
            return
        yield sse("content_block_stop", {"type": "content_block_stop", "index": index})
    yield sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": response["stop_reason"],
                "stop_sequence": response["stop_sequence"],
            },
            "usage": response["usage"],
        },
    )
    yield sse("message_stop", {"type": "message_stop"})
