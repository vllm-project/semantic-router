"""Responses-wire builders and SSE generators for the mock provider."""

import base64
import json
import math
import time
import uuid
from collections.abc import Iterator
from typing import Any

from chat_request import ChatRequest


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def build_chat_usage(req: ChatRequest, content: str) -> dict:
    prompt_text = "\n".join(
        m.content for m in req.messages if isinstance(m.content, str)
    )
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(content)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def build_chat_stream_chunk(
    req: ChatRequest,
    response_id: str,
    created_ts: int,
    delta: dict,
    finish_reason: str | None,
    usage: dict | None = None,
) -> str:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": req.model,
        "system_fingerprint": "mock-vllm",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"


def response_input_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    value = body.get("input", [])
    if isinstance(value, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": value}],
            }
        ]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def response_texts(item: dict[str, Any]) -> list[str]:
    content = item.get("content", [])
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for part in content:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return texts


def response_requests_image_generation(body: dict[str, Any]) -> bool:
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "image_generation":
        return True
    if tool_choice == "image_generation":
        return True
    tools = body.get("tools")
    if isinstance(tools, list):
        return any(
            isinstance(tool, dict) and tool.get("type") == "image_generation"
            for tool in tools
        )
    return False


def build_responses_image_generation_response(body: dict[str, Any]) -> dict[str, Any]:
    item = {
        "type": "image_generation_call",
        "id": "ig_mock_" + uuid.uuid4().hex,
        "status": "completed",
        "result": base64.urlsafe_b64encode(b"mock-generated-image").decode(),
    }
    return {
        "id": "resp_image_mock_" + uuid.uuid4().hex,
        "object": "response",
        "created_at": int(time.time()),
        "model": body.get("model", ""),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [item],
        "parallel_tool_calls": False,
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools", []),
        "usage": build_responses_usage(body, "generated image"),
    }


def generate_responses_image_generation_stream(body: dict[str, Any]) -> Iterator[str]:
    response_id = "resp_image_mock_" + uuid.uuid4().hex
    item_id = "ig_mock_" + uuid.uuid4().hex
    result = base64.urlsafe_b64encode(b"mock-generated-image").decode()
    created_at = int(time.time())

    def response_resource(status: str, item: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": body.get("model", ""),
            "status": status,
            "output": [] if item is None else [item],
        }

    in_progress_item = {
        "type": "image_generation_call",
        "id": item_id,
        "status": "in_progress",
        "result": None,
    }
    completed_item = {
        "type": "image_generation_call",
        "id": item_id,
        "status": "completed",
        "result": result,
    }

    def partial(index: int) -> dict[str, Any]:
        return {
            "partial_image_index": index,
            "partial_image_b64": base64.b64encode(f"partial-{index}".encode()).decode(),
            "size": "1024x1024",
            "quality": "high",
            "background": "transparent",
            "output_format": "png",
        }

    events = [
        ("response.created", {"response": response_resource("in_progress", None)}),
        (
            "response.output_item.added",
            {"output_index": 0, "item": in_progress_item},
        ),
        (
            "response.image_generation_call.in_progress",
            {"output_index": 0, "item_id": item_id},
        ),
        (
            "response.image_generation_call.generating",
            {"output_index": 0, "item_id": item_id},
        ),
        (
            "response.image_generation_call.partial_image",
            {"output_index": 0, "item_id": item_id, **partial(0)},
        ),
        (
            "response.image_generation_call.partial_image",
            {"output_index": 0, "item_id": item_id, **partial(1)},
        ),
        (
            "response.image_generation_call.completed",
            {"output_index": 0, "item_id": item_id},
        ),
        ("response.output_item.done", {"output_index": 0, "item": completed_item}),
        (
            "response.completed",
            {
                "response": {
                    **response_resource("completed", completed_item),
                    "usage": build_responses_usage(body, "generated image"),
                }
            },
        ),
    ]
    for sequence, (event, payload) in enumerate(events):
        yield responses_sse(event, {"sequence_number": sequence, **payload})


def build_responses_echo(body: dict[str, Any]) -> str:
    messages = response_input_messages(body)
    roles = [
        str(item.get("role", "")) for item in messages if item.get("type") == "message"
    ]
    text_config = body.get("text")
    structured_output = (
        text_config.get("format") if isinstance(text_config, dict) else None
    )
    echo = {
        "mock": "mock-vllm",
        "protocol": "responses",
        "model": body.get("model", ""),
        "roles": roles,
        "developer": [
            text
            for item in messages
            if item.get("type") == "message" and item.get("role") == "developer"
            for text in response_texts(item)
        ],
        "system": [
            text
            for item in messages
            if item.get("type") == "message" and item.get("role") == "system"
            for text in response_texts(item)
        ],
        "user": [
            text
            for item in messages
            if item.get("type") == "message" and item.get("role") == "user"
            for text in response_texts(item)
        ],
        "total_messages": len(messages),
        "request_fields": sorted(body),
    }
    if structured_output is not None:
        echo["structured_output"] = structured_output
    return json.dumps(echo, separators=(",", ":"), sort_keys=True)


def build_responses_usage(body: dict[str, Any], output: str) -> dict[str, Any]:
    input_text = "\n".join(
        text for item in response_input_messages(body) for text in response_texts(item)
    )
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def build_responses_response(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
    response_id = "resp_mock_" + uuid.uuid4().hex
    item_id = "msg_mock_" + uuid.uuid4().hex
    output = build_responses_echo(body)
    response = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": body.get("model", ""),
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": item_id,
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": output,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": output,
        "usage": build_responses_usage(body, output),
    }
    return response, item_id


def responses_sse(event: str, payload: dict[str, Any]) -> str:
    payload = {"type": event, **payload}
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def responses_in_progress_resource(response: dict[str, Any]) -> dict[str, Any]:
    resource = {
        **response,
        "status": "in_progress",
        "output": [],
        "output_text": "",
    }
    resource.pop("usage", None)
    return resource


def responses_in_progress_item(item: dict[str, Any]) -> dict[str, Any]:
    pending = {**item, "status": "in_progress"}
    if pending.get("type") == "message":
        pending["content"] = []
    if pending.get("type") == "function_call":
        pending["arguments"] = ""
    return pending


def generate_responses_stream(
    response: dict[str, Any], item_id: str, complete: bool = True
) -> Iterator[str]:
    output = response["output_text"]
    item = response["output"][0]
    content = item["content"][0]
    in_progress_response = responses_in_progress_resource(response)
    in_progress_item = responses_in_progress_item(item)
    events = [
        ("response.created", {"response": in_progress_response}),
        ("response.in_progress", {"response": in_progress_response}),
        (
            "response.output_item.added",
            {"output_index": 0, "item": in_progress_item},
        ),
        (
            "response.content_part.added",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {**content, "text": ""},
            },
        ),
        (
            "response.output_text.delta",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": output,
            },
        ),
        (
            "response.output_text.done",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": output,
            },
        ),
        (
            "response.content_part.done",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": content,
            },
        ),
        ("response.output_item.done", {"output_index": 0, "item": item}),
        ("response.completed", {"response": response}),
    ]
    for sequence, (event, payload) in enumerate(events):
        if not complete and event == "response.output_text.done":
            return
        yield responses_sse(event, {"sequence_number": sequence, **payload})


def generate_responses_midstream_error(
    response: dict[str, Any], item_id: str
) -> Iterator[str]:
    item = response["output"][0]
    content = item["content"][0]
    in_progress_response = responses_in_progress_resource(response)
    in_progress_item = responses_in_progress_item(item)
    events = [
        ("response.created", {"response": in_progress_response}),
        (
            "response.output_item.added",
            {"output_index": 0, "item": in_progress_item},
        ),
        (
            "response.content_part.added",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {**content, "text": ""},
            },
        ),
        (
            "response.output_text.delta",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": "partial",
            },
        ),
        (
            "response.failed",
            {
                "response": {
                    **response,
                    "status": "failed",
                    "output": [],
                    "output_text": "",
                    "error": {
                        "code": "provider_overloaded",
                        "message": "mock provider stream failed",
                    },
                }
            },
        ),
    ]
    for sequence, (event, payload) in enumerate(events):
        yield responses_sse(event, {"sequence_number": sequence, **payload})


def response_input_contains(body: dict[str, Any], marker: str) -> bool:
    return any(
        marker in text
        for item in response_input_messages(body)
        for text in response_texts(item)
    )


def response_has_tool_result(body: dict[str, Any]) -> bool:
    items = body.get("input")
    if not isinstance(items, list):
        return False
    calls: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        call_id = item.get("call_id")
        if item_type == "function_call" and isinstance(call_id, str) and call_id:
            calls.add(call_id)
        elif item_type == "function_call_output" and call_id in calls:
            return True
    return False


def chat_requests_mock_tool(req: ChatRequest) -> bool:
    return bool(req.tools) and chat_contains(req, "__mock_tool_call__")


def chat_contains(req: ChatRequest, marker: str) -> bool:
    return any(
        isinstance(message.content, str) and marker in message.content
        for message in req.messages
    )


def chat_has_tool_result(req: ChatRequest) -> bool:
    calls: set[str] = set()
    for message in req.messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                call_id = tool_call.get("id")
                if isinstance(call_id, str) and call_id:
                    calls.add(call_id)
        elif message.role == "tool" and message.tool_call_id in calls:
            return True
    return False


def mock_chat_tool_response(req: ChatRequest, created_ts: int) -> dict[str, Any]:
    return {
        "id": "cmpl-mock-tool-123",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_mock_lookup",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"query":"weather"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "logprobs": None,
            }
        ],
        "usage": build_chat_usage(req, "lookup weather"),
    }


def generate_chat_tool_stream(req: ChatRequest, created_ts: int) -> Iterator[str]:
    response_id = "cmpl-mock-tool-123"
    yield build_chat_stream_chunk(
        req,
        response_id,
        created_ts,
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_mock_lookup",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":'},
                }
            ],
        },
        None,
    )
    yield build_chat_stream_chunk(
        req,
        response_id,
        created_ts,
        {
            "tool_calls": [
                {
                    "index": 0,
                    "function": {"arguments": '"weather"}'},
                }
            ]
        },
        None,
    )
    yield build_chat_stream_chunk(
        req,
        response_id,
        created_ts,
        {},
        "tool_calls",
        build_chat_usage(req, "lookup weather"),
    )
    yield "data: [DONE]\n\n"


def build_responses_tool_response(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "resp_mock_tool_123",
        "object": "response",
        "created_at": int(time.time()),
        "model": body.get("model", ""),
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "item_mock_lookup",
                "call_id": "call_mock_lookup",
                "name": "lookup",
                "arguments": '{"query":"weather"}',
                "status": "completed",
            }
        ],
        "usage": build_responses_usage(body, "lookup weather"),
    }


def generate_responses_tool_stream(body: dict[str, Any]) -> Iterator[str]:
    response = build_responses_tool_response(body)
    item = response["output"][0]
    in_progress_response = responses_in_progress_resource(response)
    in_progress_item = responses_in_progress_item(item)
    events = [
        ("response.created", {"response": in_progress_response}),
        ("response.in_progress", {"response": in_progress_response}),
        (
            "response.output_item.added",
            {"output_index": 0, "item": in_progress_item},
        ),
        (
            "response.function_call_arguments.delta",
            {
                "item_id": item["id"],
                "output_index": 0,
                "delta": '{"query":',
            },
        ),
        (
            "response.function_call_arguments.delta",
            {
                "item_id": item["id"],
                "output_index": 0,
                "delta": '"weather"}',
            },
        ),
        (
            "response.function_call_arguments.done",
            {
                "item_id": item["id"],
                "output_index": 0,
                "name": item["name"],
                "arguments": item["arguments"],
            },
        ),
        ("response.output_item.done", {"output_index": 0, "item": item}),
        ("response.completed", {"response": response}),
    ]
    for sequence, (event, payload) in enumerate(events):
        yield responses_sse(event, {"sequence_number": sequence, **payload})
