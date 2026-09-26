"""Captured Ollama empty-content behavior, opt-in for codec regression E2E."""

from collections.abc import Iterator
from typing import Any

from .chat_request import ChatRequest
from .chat_wire import build_chat_stream_chunk, build_chat_usage

MARKER = "__mock_ollama_empty_content__"
TOOL_ARGUMENTS = '{"query":"weather"}'


def buffered_tool_call(req: ChatRequest, created_ts: int) -> dict[str, Any]:
    """Ollama 0.34.x sends content="" beside reasoning and tool_calls."""
    return {
        "id": "cmpl-mock-ollama-empty",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "I should use lookup for the weather request.",
                    "tool_calls": [
                        {
                            "id": "call_mock_lookup",
                            "index": 0,
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": TOOL_ARGUMENTS,
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": build_chat_usage(req, "lookup weather"),
    }


def streamed_tool_call(req: ChatRequest, created_ts: int) -> Iterator[str]:
    """Keep empty content on tool deltas, as in Ollama's captured SSE."""
    response_id = "cmpl-mock-ollama-empty"
    yield build_chat_stream_chunk(
        req,
        response_id,
        created_ts,
        {
            "role": "assistant",
            "content": "",
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
            "content": "",
            "tool_calls": [{"index": 0, "function": {"arguments": '"weather"}'}}],
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
