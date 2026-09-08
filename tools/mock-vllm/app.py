import json
import math
import time
import uuid
from collections.abc import Iterator
from typing import Any

import uvicorn
from chat_request import ChatRequest, build_chat_content
from classify import router as classify_router
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from provider_boundary import (
    SESSION_HEADER,
    RequestStore,
    invalid_request_response,
    parse_provider_request,
    router,
)
from pydantic import ValidationError

app = FastAPI()
app.state.request_store = RequestStore()
app.include_router(router)
app.include_router(classify_router)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


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
    req: ChatRequest, content: str, usage: dict, created_ts: int
) -> dict:
    return {
        "id": "cmpl-mock-123",
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "system_fingerprint": "mock-vllm",
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body, error_response = await parse_provider_request(
        request, "openai_chat_completions"
    )
    if error_response is not None:
        return error_response
    assert body is not None
    session_id = request.headers.get(SESSION_HEADER) or "__global__"
    app.state.request_store.record(session_id, body)
    try:
        req = ChatRequest.model_validate(body)
    except ValidationError as error:
        detail = error.errors(include_url=False)[0]
        field = ".".join(str(part) for part in detail.get("loc", ())) or None
        return invalid_request_response(detail["msg"], field)

    created_ts = int(time.time())
    control_response = mock_chat_control_response(req, created_ts)
    if control_response is not None:
        return control_response

    if is_hallucination_detection_request(req):
        content = build_hallucination_detection_content(req)
    else:
        content = build_chat_content(req)
    usage = build_chat_usage(req, content)
    response = build_chat_response(req, content, usage, created_ts)
    if not req.stream:
        return response

    if chat_contains(req, "__mock_midstream_error__"):
        return StreamingResponse(
            generate_chat_midstream_error(req, response, created_ts),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return StreamingResponse(
        generate_chat_stream(
            req,
            response,
            content,
            usage,
            created_ts,
            complete=not chat_contains(req, "__mock_incomplete_stream__"),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def mock_chat_control_response(req: ChatRequest, created_ts: int) -> Any | None:
    if chat_contains(req, "__mock_provider_error__"):
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "mock provider rate limit",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
        )
    if chat_has_tool_result(req):
        content = "tool result accepted"
        usage = build_chat_usage(req, content)
        response = build_chat_response(req, content, usage, created_ts)
        if req.stream:
            return StreamingResponse(
                generate_chat_stream(req, response, content, usage, created_ts),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return response
    if chat_requests_mock_tool(req):
        if req.stream:
            return StreamingResponse(
                generate_chat_tool_stream(req, created_ts),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return mock_chat_tool_response(req, created_ts)
    return None


@app.post("/v1/responses")
async def responses(request: Request):
    body, error_response = await parse_provider_request(request, "openai_responses")
    if error_response is not None:
        return error_response
    assert body is not None
    session_id = request.headers.get(SESSION_HEADER) or "__global__"
    app.state.request_store.record(session_id, body)
    if response_input_contains(body, "__mock_provider_error__"):
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "mock provider rate limit",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
        )
    if response_has_tool_result(body):
        body = {**body, "input": "tool result accepted"}
    elif response_input_contains(body, "__mock_tool_call__") and body.get("tools"):
        if body.get("stream"):
            return StreamingResponse(
                generate_responses_tool_stream(body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return build_responses_tool_response(body)
    response, item_id = build_responses_response(body)
    if not body.get("stream"):
        return response
    if response_input_contains(body, "__mock_midstream_error__"):
        return StreamingResponse(
            generate_responses_midstream_error(response, item_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    return StreamingResponse(
        generate_responses_stream(
            response,
            item_id,
            complete=not response_input_contains(body, "__mock_incomplete_stream__"),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
