"""Captured vLLM stop-sequence behavior, opt-in for codec regression E2E."""

import json
from collections.abc import Iterator
from typing import Any

from .chat_request import ChatRequest
from .chat_wire import build_chat_usage

MARKER = "__mock_vllm_stop_sequence__"
ANSWER = "ALPHA BRAVO "


def matched_stop(req: ChatRequest) -> str | None:
    """The first requested stop string; vLLM names the one that matched."""
    stop = getattr(req, "stop", None)
    if isinstance(stop, str) and stop:
        return stop
    if isinstance(stop, list) and stop and isinstance(stop[0], str):
        return stop[0]
    return None


def buffered_reply(req: ChatRequest, created_ts: int, stop: str) -> dict[str, Any]:
    """vLLM 0.26.0 reports the matched stop string in choices[].stop_reason."""
    return {
        "id": "chatcmpl-mock-vllm-stop",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ANSWER},
                "logprobs": None,
                "finish_reason": "stop",
                "stop_reason": stop,
            }
        ],
        "usage": build_chat_usage(req, ANSWER),
    }


def streamed_reply(req: ChatRequest, created_ts: int, stop: str) -> Iterator[str]:
    """The terminal chunk carries both finish_reason and stop_reason, as captured."""
    base = {
        "id": "chatcmpl-mock-vllm-stop",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": req.model,
    }
    for delta, finish, stop_reason in (
        ({"role": "assistant", "content": ""}, None, None),
        ({"content": ANSWER.strip()}, None, None),
        ({"content": " "}, "stop", stop),
    ):
        choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
        if stop_reason is not None:
            choice["stop_reason"] = stop_reason
        yield "data: " + json.dumps(
            dict(base, choices=[choice]), separators=(",", ":")
        ) + "\n\n"
    options = (req.model_extra or {}).get("stream_options") or {}
    if options.get("include_usage"):
        usage_chunk = dict(base, choices=[], usage=build_chat_usage(req, ANSWER))
        yield "data: " + json.dumps(usage_chunk, separators=(",", ":")) + "\n\n"
    yield "data: [DONE]\n\n"
