"""Native Chat Completions response and stream fixtures."""

import json
from collections.abc import Iterator
from typing import Any

from .chat_request import ChatRequest
from .tokens import estimate_tokens


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
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
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
        "system_fingerprint": "provider-mocker",
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


def chat_requests_mock_tool(req: ChatRequest) -> bool:
    return bool(req.tools) and chat_contains(req, "__mock_tool_call__")


def chat_contains(req: ChatRequest, marker: str) -> bool:
    for message in req.messages:
        content = message.content
        if isinstance(content, str) and marker in content:
            return True
        if isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                    and marker in part["text"]
                ):
                    return True
    return False


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


def generate_chat_custom_tool_kind_stream(
    req: ChatRequest, created_ts: int, variant: str
) -> Iterator[str]:
    """Emit a complete native stream with one valid or switched tool kind."""
    response_id = "cmpl-mock-custom-kind-123"
    yield build_chat_stream_chunk(
        req, response_id, created_ts, {"role": "assistant"}, None
    )
    if variant == "function_to_custom":
        first = {
            "index": 0,
            "id": "call_mock_custom_kind",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
        remaining = [{"index": 0, "custom": {"input": "def"}}]
    else:
        first = {
            "index": 0,
            "id": "call_mock_custom_kind",
            "type": "custom",
            "custom": {"name": "apply_patch", "input": "abc"},
        }
        if variant == "custom_to_function":
            remaining = [
                {
                    "index": 0,
                    "type": "function",
                    "function": {"arguments": '{"x":1}'},
                }
            ]
        elif variant == "custom_to_untyped_function":
            remaining = [{"index": 0, "function": {"arguments": '{"x":1}'}}]
        elif variant == "valid_custom":
            remaining = [
                {"index": 0, "custom": {"input": "def"}},
                {"index": 0},
                {"index": 0, "custom": {"input": "ghi"}},
            ]
        else:
            raise ValueError(f"unknown custom tool kind fixture: {variant}")

    for call in [first, *remaining]:
        yield build_chat_stream_chunk(
            req, response_id, created_ts, {"tool_calls": [call]}, None
        )
    yield build_chat_stream_chunk(
        req,
        response_id,
        created_ts,
        {},
        "tool_calls",
        build_chat_usage(req, "apply_patch"),
    )
    yield "data: [DONE]\n\n"


CUSTOM_TOOL_FIXTURE_INPUT = "*** Begin Patch\n+provider\n*** End Patch"


def build_chat_custom_tool_response(
    req: ChatRequest, created_ts: int
) -> dict[str, Any]:
    return {
        "id": "cmpl-mock-custom-tool-123",
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
                            "id": "call_mock_patch",
                            "type": "custom",
                            "custom": {
                                "name": "apply_patch",
                                "input": CUSTOM_TOOL_FIXTURE_INPUT,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "logprobs": None,
            }
        ],
        "usage": build_chat_usage(req, CUSTOM_TOOL_FIXTURE_INPUT),
    }


def generate_chat_custom_tool_stream(
    req: ChatRequest, created_ts: int
) -> Iterator[str]:
    response_id = "cmpl-mock-custom-tool-123"
    # Some Chat providers announce an id before the custom kind/name.
    yield build_chat_stream_chunk(
        req,
        response_id,
        created_ts,
        {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_mock_patch"}]},
        None,
    )
    first, second = CUSTOM_TOOL_FIXTURE_INPUT[:15], CUSTOM_TOOL_FIXTURE_INPUT[15:]
    yield build_chat_stream_chunk(
        req,
        response_id,
        created_ts,
        {
            "tool_calls": [
                {
                    "index": 0,
                    "type": "custom",
                    "custom": {"name": "apply_patch", "input": first},
                }
            ]
        },
        None,
    )
    yield build_chat_stream_chunk(
        req,
        response_id,
        created_ts,
        {"tool_calls": [{"index": 0, "custom": {"input": second}}]},
        None,
    )
    yield build_chat_stream_chunk(
        req,
        response_id,
        created_ts,
        {},
        "tool_calls",
        build_chat_usage(req, CUSTOM_TOOL_FIXTURE_INPUT),
    )
    yield "data: [DONE]\n\n"


def build_chat_response(
    req: ChatRequest, content: str, usage: dict, created_ts: int
) -> dict:
    return {
        "id": "cmpl-mock-123",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "system_fingerprint": "provider-mocker",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": build_chat_logprobs(req, content),
            }
        ],
        "usage": usage,
    }


def build_chat_logprobs(req: ChatRequest, content: str) -> dict[str, Any] | None:
    if not req.logprobs:
        return None
    token = content[:8] or "mock"
    requested = max(1, min(req.top_logprobs or 1, 5))
    alternatives = [
        {"token": token, "logprob": -1.5, "bytes": list(token.encode())},
        {"token": "other", "logprob": -1.6, "bytes": list(b"other")},
    ]
    while len(alternatives) < requested:
        index = len(alternatives)
        alternative = f"alt-{index}"
        alternatives.append(
            {
                "token": alternative,
                "logprob": -1.6 - index,
                "bytes": list(alternative.encode()),
            }
        )
    return {
        "content": [
            {
                "token": token,
                "logprob": -1.5,
                "bytes": list(token.encode()),
                "top_logprobs": alternatives[:requested],
            }
        ],
        "refusal": [],
    }


def generate_chat_stream(
    req: ChatRequest,
    response: dict,
    content: str,
    usage: dict,
    created_ts: int,
    complete: bool = True,
):
    chunk_size = 24
    response_id = response["id"]
    for i in range(0, len(content), chunk_size):
        yield build_chat_stream_chunk(
            req,
            response_id,
            created_ts,
            {"content": content[i : i + chunk_size]},
            None,
        )
        if not complete:
            return
    yield build_chat_stream_chunk(req, response_id, created_ts, {}, "stop", usage)
    yield "data: [DONE]\n\n"


def generate_chat_midstream_error(
    req: ChatRequest,
    response: dict[str, Any],
    created_ts: int,
) -> Iterator[str]:
    yield build_chat_stream_chunk(
        req,
        response["id"],
        created_ts,
        {"content": "partial"},
        None,
    )
    yield (
        "data: "
        + json.dumps(
            {
                "error": {
                    "message": "mock provider stream failed",
                    "type": "server_error",
                    "param": None,
                    "code": "provider_overloaded",
                }
            },
            separators=(",", ":"),
        )
        + "\n\n"
    )
