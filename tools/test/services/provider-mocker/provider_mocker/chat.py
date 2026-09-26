"""Native Chat Completions HTTP boundary and fixture selection."""

import json
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from . import ollama_fixture, openrouter_fixture, workflow_chat
from .chat_request import ChatRequest, build_chat_content
from .chat_wire import (
    build_chat_response,
    build_chat_usage,
    chat_contains,
    chat_has_tool_result,
    chat_requests_mock_tool,
    generate_chat_custom_tool_kind_stream,
    generate_chat_midstream_error,
    generate_chat_stream,
    generate_chat_tool_stream,
    mock_chat_tool_response,
)
from .provider_boundary import (
    SESSION_HEADER,
    invalid_request_response,
    parse_provider_request,
)
from .scenarios import respond_to_scenario
from .settings import apply_fixture_delay
from .shadow_control import ShadowControl

router = APIRouter()

CUSTOM_TOOL_KIND_STREAM_MARKERS = {
    "__mock_custom_kind_custom_to_function__": "custom_to_function",
    "__mock_custom_kind_custom_to_untyped_function__": "custom_to_untyped_function",
    "__mock_custom_kind_function_to_custom__": "function_to_custom",
    "__mock_custom_kind_valid_custom__": "valid_custom",
}


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


_chat_control = workflow_chat.ChatControlHelpers(
    chat_contains=chat_contains,
    chat_has_tool_result=chat_has_tool_result,
    chat_requests_mock_tool=chat_requests_mock_tool,
    build_chat_usage=build_chat_usage,
    build_chat_response=build_chat_response,
    mock_chat_tool_response=mock_chat_tool_response,
    generate_chat_stream=generate_chat_stream,
    generate_chat_tool_stream=generate_chat_tool_stream,
)


def mock_chat_control_response(req: ChatRequest, created_ts: int) -> Any | None:
    return workflow_chat.mock_chat_control_response(req, created_ts, _chat_control)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    raw_body = await request.body()
    body, error_response = await parse_provider_request(
        request, "openai_chat_completions"
    )
    if error_response is not None:
        return error_response
    assert body is not None
    session_id = request.headers.get(SESSION_HEADER) or "__global__"
    request.app.state.request_store.record(session_id, body, request.headers, raw_body)
    try:
        req = ChatRequest.model_validate(body)
    except ValidationError as error:
        detail = error.errors(include_url=False)[0]
        field = ".".join(str(part) for part in detail.get("loc", ())) or None
        return invalid_request_response(detail["msg"], field)

    await apply_fixture_delay()
    created_ts = int(time.time())
    scenario_response = await respond_to_scenario(request, req, created_ts)
    if scenario_response is not None:
        return scenario_response
    if chat_contains(req, openrouter_fixture.MARKER):
        if req.stream:
            return StreamingResponse(
                openrouter_fixture.streamed_reply(req, created_ts),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return openrouter_fixture.buffered_reply(req, created_ts)
    if req.tools and chat_contains(req, ollama_fixture.MARKER):
        if req.stream:
            return StreamingResponse(
                ollama_fixture.streamed_tool_call(req, created_ts),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return ollama_fixture.buffered_tool_call(req, created_ts)
    if (
        req.stream
        and req.tools
        and any(
            tool.get("type") == "custom"
            and isinstance(tool.get("custom"), dict)
            and tool["custom"].get("name") == "apply_patch"
            for tool in req.tools
        )
    ):
        for marker, variant in CUSTOM_TOOL_KIND_STREAM_MARKERS.items():
            if chat_contains(req, marker):
                return StreamingResponse(
                    generate_chat_custom_tool_kind_stream(req, created_ts, variant),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
    control_response = mock_chat_control_response(req, created_ts)
    if control_response is not None:
        return control_response

    shadow_control: ShadowControl | None = request.app.state.shadow_control
    if shadow_control is not None:
        shadow_response = await shadow_control.respond(
            model=req.model, request_id=request.headers.get("x-request-id", "")
        )
        if shadow_response is not None:
            return shadow_response
        content = f"Hello from {req.model}."
    elif is_hallucination_detection_request(req):
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
