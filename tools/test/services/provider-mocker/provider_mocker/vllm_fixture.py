"""Captured vLLM stop, reasoning and in-stream error behavior, opt-in for codec regression E2E."""

import json
from asyncio import sleep
from collections.abc import AsyncIterator, Iterator
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

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


STREAM_ERROR_MARKER = "__mock_vllm_stream_error__"
MIDSTREAM_ERROR_MARKER = "__mock_vllm_midstream_error__"
# The same streams, with [DONE] in a later write than the error event.
STREAM_ERROR_SPLIT_MARKER = "__mock_vllm_stream_error_split__"
MIDSTREAM_ERROR_SPLIT_MARKER = "__mock_vllm_midstream_error_split__"
DONE_EVENT = "data: [DONE]\n\n"
# A delay, not a guarantee: Envoy normally hands the Router [DONE] in a later
# body chunk. The ExtProc tests pin each read layout exactly.
SPLIT_PAUSE_SECONDS = 0.3

# Captured from vLLM 0.30.0 for a json_schema whose regex does not compile:
# streamed, after HTTP 200, as an SSE event; buffered, as the HTTP 400 body.
GRAMMAR_ERROR = {
    "message": "Grammar error: regex parse error:\n    [\n    ^\nerror: unclosed character class",
    "type": "BadRequestError",
    "param": None,
    "code": 400,
}

# Synthetic: vLLM 0.30.0's shape for an error raised during generation
# (GenerationError), not captured.
GENERATION_ERROR = {
    "message": "Internal server error",
    "type": "InternalServerError",
    "param": None,
    "code": 500,
}


def error_stream(
    req: ChatRequest, created_ts: int, error: dict[str, Any], content: str = ""
) -> Iterator[str]:
    """Any content first, then the error event with its integer HTTP status
    (json.dumps spacing, as captured) and [DONE] in one write, so the Router
    normally reads them together."""
    if content:
        base = {
            "id": "chatcmpl-mock-vllm-error",
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": req.model,
        }
        for delta in ({"role": "assistant", "content": ""}, {"content": content}):
            choice = {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": None,
            }
            yield "data: " + json.dumps(
                dict(base, choices=[choice]), separators=(",", ":")
            ) + "\n\n"
    yield "data: " + json.dumps({"error": error}) + "\n\n" + DONE_EVENT


async def split_before_done(events: Iterator[str]) -> AsyncIterator[str]:
    """The same bytes, but [DONE] goes in its own write after a pause, so the
    Router normally reads it later than the error event."""
    for event in events:
        if event.endswith(DONE_EVENT):
            yield event.removesuffix(DONE_EVENT)
            await sleep(SPLIT_PAUSE_SECONDS)
            yield DONE_EVENT
        else:
            yield event


def error_reply(
    req: ChatRequest, created_ts: int
) -> JSONResponse | StreamingResponse | None:
    """vLLM's replies for the error markers, or None for any other request."""
    stream_split = chat_contains(req, STREAM_ERROR_SPLIT_MARKER)
    midstream_split = chat_contains(req, MIDSTREAM_ERROR_SPLIT_MARKER)
    if stream_split or chat_contains(req, STREAM_ERROR_MARKER):
        if not req.stream:
            return JSONResponse(status_code=400, content={"error": GRAMMAR_ERROR})
        events = error_stream(req, created_ts, GRAMMAR_ERROR)
    elif req.stream and (midstream_split or chat_contains(req, MIDSTREAM_ERROR_MARKER)):
        events = error_stream(req, created_ts, GENERATION_ERROR, content="partial")
    else:
        return None
    if stream_split or midstream_split:
        events = split_before_done(events)
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
