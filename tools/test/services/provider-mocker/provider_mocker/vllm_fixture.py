"""Captured vLLM stop and reasoning behavior, opt-in for codec regression E2E."""

import json
from collections.abc import Iterator
from typing import Any

from fastapi.responses import StreamingResponse

from .chat_request import ChatRequest
from .chat_wire import build_chat_usage, chat_contains

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


REASONING_ORDER_MARKER = "__mock_vllm_reasoning_order__"

# Mirrors pkg/protocolcodec/testdata/providers/vllm-chat-reasoning-and-answer-one-delta-stream.sse
# (vLLM 0.30.0, Qwen3.8-27B-FP8, qwen3 reasoning parser, speculative decoding):
# one delta ends the reasoning and starts the answer. tests/test_vllm_fixture.py
# keeps the two equal.
REASONING_ANSWER_DELTAS = (
    {"role": "assistant", "content": ""},
    {"reasoning": ' how are you" five'},
    {"reasoning": " words. final"},
    {"content": "\n\nHello", "reasoning": ".\n"},
    {"content": " there, how are you"},
)
REASONING_TEXT = "".join(
    delta.get("reasoning", "") for delta in REASONING_ANSWER_DELTAS
)
ANSWER_TEXT = "".join(delta.get("content", "") for delta in REASONING_ANSWER_DELTAS)


def reasoning_answer_stream(req: ChatRequest, created_ts: int) -> Iterator[str]:
    """The captured deltas, the finish frame, any usage chunk, then [DONE]."""
    base = {
        "id": "chatcmpl-mock-vllm-reasoning",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": req.model,
    }
    for delta, finish in [(d, None) for d in REASONING_ANSWER_DELTAS] + [({}, "stop")]:
        choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
        yield "data: " + json.dumps(
            dict(base, choices=[choice]), separators=(",", ":")
        ) + "\n\n"
    options = (req.model_extra or {}).get("stream_options") or {}
    if options.get("include_usage"):
        usage = build_chat_usage(req, REASONING_TEXT + ANSWER_TEXT)
        usage_chunk = dict(base, choices=[], usage=usage)
        yield "data: " + json.dumps(usage_chunk, separators=(",", ":")) + "\n\n"
    yield "data: [DONE]\n\n"


def reasoning_answer_reply(
    req: ChatRequest, created_ts: int
) -> dict[str, Any] | StreamingResponse | None:
    """vLLM's reply for the reasoning marker, or None for any other request."""
    if not chat_contains(req, REASONING_ORDER_MARKER):
        return None
    if req.stream:
        return StreamingResponse(
            reasoning_answer_stream(req, created_ts),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    message = {"role": "assistant", "content": ANSWER_TEXT, "reasoning": REASONING_TEXT}
    return {
        "id": "chatcmpl-mock-vllm-reasoning",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {"index": 0, "message": message, "logprobs": None, "finish_reason": "stop"}
        ],
        "usage": build_chat_usage(req, REASONING_TEXT + ANSWER_TEXT),
    }
