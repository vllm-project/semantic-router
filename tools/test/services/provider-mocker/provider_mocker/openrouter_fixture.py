"""OpenRouter-style Chat replies for the protocol translation E2E case."""

import json
from collections.abc import Iterator

from .chat_request import ChatRequest
from .chat_wire import build_chat_usage

MARKER = "__mock_openrouter_reply__"
ANSWER = "OpenRouter fixture answer"


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
    return {
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


def streamed_reply(req: ChatRequest, created_ts: int) -> Iterator[str]:
    base = {
        "id": "gen-mock-openrouter-1",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": req.model,
        "provider": "OpenRouter fixture upstream",
    }
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
