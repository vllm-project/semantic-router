"""OpenRouter-style Chat replies for the protocol translation E2E case."""

import json
from collections.abc import Iterator

from .chat_request import ChatRequest
from .chat_wire import build_chat_usage

MARKER = "__mock_openrouter_reply__"
ANSWER = "OpenRouter fixture answer"
REASONING_MARKER = "__mock_openrouter_reply_reasoning__"
REASONING_DETAILS = [
    {
        "type": "reasoning.summary",
        "summary": "Look up the result",
        "id": None,
        "format": "openai-responses-v1",
        "index": 0,
    },
    {
        "type": "reasoning.text",
        "text": "Use the lookup tool",
        "signature": "fixture-signature",
        "id": "reasoning-1",
        "format": "anthropic-claude-v1",
        "index": 1,
    },
    {
        "type": "reasoning.encrypted",
        "data": "opaque-continuity-fixture",
        "id": "reasoning-2",
        "format": "openai-responses-v1",
        "index": 2,
    },
]
TOOL_CALL = {
    "id": "openrouter-call-1",
    "type": "function",
    "function": {"name": "lookup", "arguments": "{}"},
}


def reasoning_turn(req: ChatRequest) -> bool:
    return any(
        isinstance(message.content, str) and REASONING_MARKER in message.content
        for message in req.messages
    )


def reasoning_follow_up_valid(req: ChatRequest) -> bool:
    return any(
        message.role == "assistant"
        and getattr(message, "reasoning_details", None) == REASONING_DETAILS
        and message.tool_calls == [TOOL_CALL]
        for message in req.messages
    )


def _usage(req: ChatRequest) -> dict:
    usage = build_chat_usage(req, ANSWER)
    usage["prompt_tokens_details"]["video_tokens"] = 0
    usage["completion_tokens_details"]["image_tokens"] = 0
    usage.update(
        cost=0.00014,
        is_byok=False,
        cost_details={
            "upstream_inference_cost": 0.0001,
            "upstream_inference_prompt_cost": 0.00006,
            "upstream_inference_completions_cost": 0.00004,
            "server_tool_cost": None,
        },
        server_tool_use={"web_search_requests": 0},
    )
    return usage


def buffered_reply(req: ChatRequest, created_ts: int) -> dict:
    reply = {
        "id": "gen-mock-openrouter-1",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "provider": "OpenRouter fixture upstream",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ANSWER},
                "finish_reason": "stop",
                "native_finish_reason": "stop",
            }
        ],
        "usage": _usage(req),
    }
    if reasoning_turn(req) and not any(
        message.role == "tool" for message in req.messages
    ):
        reply["choices"][0].update(
            message={
                "role": "assistant",
                "content": None,
                "reasoning_details": REASONING_DETAILS,
                "tool_calls": [TOOL_CALL],
            },
            finish_reason="tool_calls",
            native_finish_reason="tool_calls",
        )
    return reply


def streamed_reply(req: ChatRequest, created_ts: int) -> Iterator[str]:
    base = {
        "id": "gen-mock-openrouter-1",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": req.model,
        "provider": "OpenRouter fixture upstream",
    }
    if reasoning_turn(req) and not any(
        message.role == "tool" for message in req.messages
    ):
        for detail in REASONING_DETAILS:
            yield "data: " + json.dumps(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_details": [detail]},
                            "finish_reason": None,
                        }
                    ],
                }
            ) + "\n\n"
        yield "data: " + json.dumps(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, **TOOL_CALL}]},
                        "finish_reason": None,
                    }
                ],
            }
        ) + "\n\n"
        yield "data: " + json.dumps(
            {
                **base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": _usage(req),
            }
        ) + "\n\n"
        yield "data: [DONE]\n\n"
        return
    for delta, finish, usage in (
        ({"content": ANSWER}, None, None),
        ({}, "stop", None),
        ({"content": ""}, "stop", _usage(req)),
    ):
        chunk = dict(base)
        chunk["choices"] = [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
                "native_finish_reason": finish,
            }
        ]
        if usage is not None:
            chunk["usage"] = usage
        yield "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"
    yield "data: [DONE]\n\n"
