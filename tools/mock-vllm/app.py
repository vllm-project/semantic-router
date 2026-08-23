import json
import math
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: Any | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.2
    stream: bool | None = False
    response_format: dict | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


class ResponsesRequest(BaseModel):
    model: str
    input: Any
    instructions: Any | None = None
    stream: bool | None = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


class AnthropicRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int
    system: Any | None = None
    stream: bool | None = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def build_protocol_content(protocol: str, model: str) -> str:
    return json.dumps(
        {"mock": "mock-vllm", "model": model, "protocol": protocol},
        separators=(",", ":"),
        sort_keys=True,
    )


def requested_tool_name(tools: list[dict[str, Any]] | None) -> str | None:
    if not tools:
        return None
    tool = tools[0]
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = tool.get("name")
    return name if isinstance(name, str) and name else None


def build_protocol_tool_call(protocol: str, tools: list[dict[str, Any]] | None):
    name = requested_tool_name(tools)
    if name != "protocol_marker":
        return None
    return {
        "id": "call_protocol_123",
        "name": name,
        "arguments": json.dumps({"protocol": protocol}, separators=(",", ":")),
    }


def build_chat_content(req: ChatRequest) -> str:
    roles = [m.role for m in req.messages]
    system_messages = [m.content for m in req.messages if m.role == "system"]
    user_messages = [m.content for m in req.messages if m.role == "user"]

    return json.dumps(
        {
            "mock": "mock-vllm",
            "model": req.model,
            "protocol": "openai.chat.v1",
            "roles": roles,
            "system": system_messages,
            "user": user_messages,
            "total_messages": len(req.messages),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def is_hallucination_detection_request(req: ChatRequest) -> bool:
    if not req.response_format or req.response_format.get("type") != "json_schema":
        return False
    schema = req.response_format.get("json_schema", {})
    return schema.get("name") == "hallucination_detection"


def extract_answer_under_review(req: ChatRequest) -> str:
    for m in req.messages:
        if (
            m.role == "user"
            and isinstance(m.content, str)
            and "Answer to verify:\n" in m.content
        ):
            return m.content.split("Answer to verify:\n")[-1]
    return ""


def build_hallucination_detection_content(req: ChatRequest) -> str:
    answer = extract_answer_under_review(req)
    flagged_text = answer[:80] if answer else "mocked hallucination"
    return json.dumps(
        {
            "hallucinated_spans": [
                {
                    "text": flagged_text,
                    "category": "unsupported_addition",
                    "subcategory": "claim",
                }
            ]
        }
    )


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


def build_chat_response(
    req: ChatRequest,
    content: str,
    usage: dict,
    created_ts: int,
    tool_call: dict | None = None,
) -> dict:
    message = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_call is not None:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    },
                }
            ],
        }
        finish_reason = "tool_calls"
    return {
        "id": "cmpl-mock-123",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "system_fingerprint": "mock-vllm",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": usage,
        "token_usage": usage,
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


def generate_chat_stream(
    req: ChatRequest,
    response: dict,
    content: str,
    usage: dict,
    created_ts: int,
    tool_call: dict | None = None,
):
    chunk_size = 24
    response_id = response["id"]
    if tool_call is not None:
        midpoint = len(tool_call["arguments"]) // 2
        for index, arguments in enumerate(
            (tool_call["arguments"][:midpoint], tool_call["arguments"][midpoint:])
        ):
            function = {"arguments": arguments}
            call = {"index": 0, "function": function}
            if index == 0:
                call.update({"id": tool_call["id"], "type": "function"})
                function["name"] = tool_call["name"]
            yield build_chat_stream_chunk(
                req,
                response_id,
                created_ts,
                {"role": "assistant" if index == 0 else None, "tool_calls": [call]},
                None,
            )
        yield build_chat_stream_chunk(
            req, response_id, created_ts, {}, "tool_calls", usage
        )
        yield "data: [DONE]\n\n"
        return
    for i in range(0, len(content), chunk_size):
        yield build_chat_stream_chunk(
            req,
            response_id,
            created_ts,
            {"content": content[i : i + chunk_size]},
            None,
        )
    yield build_chat_stream_chunk(req, response_id, created_ts, {}, "stop", usage)
    yield "data: [DONE]\n\n"


def build_responses_usage(req: ResponsesRequest, content: str) -> dict:
    input_text = json.dumps(req.input, separators=(",", ":"), sort_keys=True)
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(content)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def build_responses_response(
    req: ResponsesRequest,
    content: str,
    usage: dict,
    created_ts: int,
    tool_call: dict | None = None,
) -> dict:
    output = [
        {
            "id": "msg_mock_123",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        }
    ]
    if tool_call is not None:
        output = [
            {
                "id": "fc_mock_123",
                "type": "function_call",
                "status": "completed",
                "call_id": tool_call["id"],
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            }
        ]
    return {
        "id": "resp_mock_123",
        "object": "response",
        "created_at": created_ts,
        "model": req.model,
        "status": "completed",
        "output": output,
        "usage": usage,
    }


def responses_sse(event_type: str, payload: dict) -> str:
    body = {"type": event_type, **payload}
    return (
        "event: "
        + event_type
        + "\ndata: "
        + json.dumps(body, separators=(",", ":"))
        + "\n\n"
    )


def generate_responses_stream(
    req: ResponsesRequest,
    response: dict,
    content: str,
    usage: dict,
    tool_call: dict | None = None,
):
    in_progress = {
        "id": response["id"],
        "object": "response",
        "model": req.model,
        "status": "in_progress",
    }
    item = {
        "id": "msg_mock_123",
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    content_part = {"type": "output_text", "text": "", "annotations": []}
    yield responses_sse("response.created", {"response": in_progress})
    yield responses_sse("response.in_progress", {"response": in_progress})
    if tool_call is not None:
        item = {
            "id": "fc_mock_123",
            "type": "function_call",
            "status": "in_progress",
            "call_id": tool_call["id"],
            "name": tool_call["name"],
            "arguments": "",
        }
        yield responses_sse(
            "response.output_item.added", {"output_index": 0, "item": item}
        )
        midpoint = len(tool_call["arguments"]) // 2
        for arguments in (
            tool_call["arguments"][:midpoint],
            tool_call["arguments"][midpoint:],
        ):
            yield responses_sse(
                "response.function_call_arguments.delta",
                {
                    "item_id": item["id"],
                    "output_index": 0,
                    "delta": arguments,
                },
            )
        yield responses_sse(
            "response.function_call_arguments.done",
            {
                "item_id": item["id"],
                "output_index": 0,
                "arguments": tool_call["arguments"],
            },
        )
        yield responses_sse(
            "response.output_item.done",
            {
                "output_index": 0,
                "item": {
                    **item,
                    "status": "completed",
                    "arguments": tool_call["arguments"],
                },
            },
        )
        yield responses_sse(
            "response.completed", {"response": {**response, "usage": usage}}
        )
        return
    yield responses_sse("response.output_item.added", {"output_index": 0, "item": item})
    yield responses_sse(
        "response.content_part.added",
        {
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "part": content_part,
        },
    )
    yield responses_sse(
        "response.output_text.delta",
        {
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "delta": content,
        },
    )
    yield responses_sse(
        "response.output_text.done",
        {
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "text": content,
        },
    )
    completed_part = {**content_part, "text": content}
    yield responses_sse(
        "response.content_part.done",
        {
            "item_id": item["id"],
            "output_index": 0,
            "content_index": 0,
            "part": completed_part,
        },
    )
    completed_item = {
        **item,
        "status": "completed",
        "content": [completed_part],
    }
    yield responses_sse(
        "response.output_item.done",
        {"output_index": 0, "item": completed_item},
    )
    yield responses_sse(
        "response.completed",
        {
            "response": {
                **response,
                "usage": usage,
            }
        },
    )


def build_anthropic_usage(req: AnthropicRequest, content: str) -> dict:
    input_text = json.dumps(
        {"system": req.system, "messages": req.messages},
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "input_tokens": estimate_tokens(input_text),
        "output_tokens": estimate_tokens(content),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def build_anthropic_response(
    req: AnthropicRequest,
    content: str,
    usage: dict,
    tool_call: dict | None = None,
) -> dict:
    blocks = [{"type": "text", "text": content}]
    stop_reason = "end_turn"
    if tool_call is not None:
        blocks = [
            {
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["name"],
                "input": json.loads(tool_call["arguments"]),
            }
        ]
        stop_reason = "tool_use"
    return {
        "id": "msg_mock_123",
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def anthropic_sse(event_type: str, payload: dict) -> str:
    body = {"type": event_type, **payload}
    return (
        "event: "
        + event_type
        + "\ndata: "
        + json.dumps(body, separators=(",", ":"))
        + "\n\n"
    )


def generate_anthropic_stream(
    req: AnthropicRequest,
    content: str,
    usage: dict,
    tool_call: dict | None = None,
):
    yield anthropic_sse(
        "message_start",
        {
            "message": {
                "id": "msg_mock_123",
                "type": "message",
                "role": "assistant",
                "model": req.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {**usage, "output_tokens": 0},
            }
        },
    )
    if tool_call is not None:
        yield anthropic_sse(
            "content_block_start",
            {
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": {},
                },
            },
        )
        midpoint = len(tool_call["arguments"]) // 2
        for arguments in (
            tool_call["arguments"][:midpoint],
            tool_call["arguments"][midpoint:],
        ):
            yield anthropic_sse(
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": arguments,
                    },
                },
            )
        yield anthropic_sse("content_block_stop", {"index": 0})
        yield anthropic_sse(
            "message_delta",
            {
                "delta": {"type": "message_delta", "stop_reason": "tool_use"},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
        yield anthropic_sse("message_stop", {})
        return
    yield anthropic_sse(
        "content_block_start",
        {"index": 0, "content_block": {"type": "text", "text": ""}},
    )
    yield anthropic_sse(
        "content_block_delta",
        {"index": 0, "delta": {"type": "text_delta", "text": content}},
    )
    yield anthropic_sse("content_block_stop", {"index": 0})
    yield anthropic_sse(
        "message_delta",
        {
            "delta": {"type": "message_delta", "stop_reason": "end_turn"},
            "usage": {"output_tokens": usage["output_tokens"]},
        },
    )
    yield anthropic_sse("message_stop", {})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "openai/gpt-oss-20b", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    created_ts = int(time.time())
    if is_hallucination_detection_request(req):
        content = build_hallucination_detection_content(req)
    else:
        content = build_chat_content(req)
    tool_call = build_protocol_tool_call("openai.chat.v1", req.tools)
    completion = tool_call["arguments"] if tool_call is not None else content
    usage = build_chat_usage(req, completion)
    response = build_chat_response(req, content, usage, created_ts, tool_call)
    if not req.stream:
        return response

    return StreamingResponse(
        generate_chat_stream(req, response, content, usage, created_ts, tool_call),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/v1/responses")
async def responses(req: ResponsesRequest):
    created_ts = int(time.time())
    content = build_protocol_content("openai.responses.v1", req.model)
    tool_call = build_protocol_tool_call("openai.responses.v1", req.tools)
    completion = tool_call["arguments"] if tool_call is not None else content
    usage = build_responses_usage(req, completion)
    response = build_responses_response(req, content, usage, created_ts, tool_call)
    if not req.stream:
        return response
    return StreamingResponse(
        generate_responses_stream(req, response, content, usage, tool_call),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicRequest):
    content = build_protocol_content("anthropic.messages.v1", req.model)
    tool_call = build_protocol_tool_call("anthropic.messages.v1", req.tools)
    completion = tool_call["arguments"] if tool_call is not None else content
    usage = build_anthropic_usage(req, completion)
    response = build_anthropic_response(req, content, usage, tool_call)
    if not req.stream:
        return response
    return StreamingResponse(
        generate_anthropic_stream(req, content, usage, tool_call),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
