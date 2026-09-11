import json
import time
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
from responses_wire import (
    build_chat_stream_chunk,
    build_chat_usage,
    build_responses_image_generation_response,
    build_responses_response,
    build_responses_tool_response,
    chat_contains,
    chat_has_tool_result,
    chat_requests_mock_tool,
    generate_chat_tool_stream,
    generate_responses_image_generation_stream,
    generate_responses_midstream_error,
    generate_responses_stream,
    generate_responses_tool_stream,
    mock_chat_tool_response,
    response_has_tool_result,
    response_input_contains,
    response_requests_image_generation,
)

app = FastAPI()
app.state.request_store = RequestStore()
app.include_router(router)
app.include_router(classify_router)


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
    if response_requests_image_generation(body):
        if body.get("stream"):
            return StreamingResponse(
                generate_responses_image_generation_stream(body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return build_responses_image_generation_response(body)
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
