"""Native Responses API HTTP route."""

import asyncio
import re
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .provider_boundary import SESSION_HEADER, parse_provider_request
from .responses_wire import (
    build_responses_image_generation_response,
    build_responses_response,
    build_responses_tool_response,
    generate_responses_image_generation_stream,
    generate_responses_midstream_error,
    generate_responses_stream,
    generate_responses_tool_stream,
    response_has_tool_result,
    response_input_contains,
    response_input_messages,
    response_requests_image_generation,
    response_texts,
)
from .settings import apply_fixture_delay

router = APIRouter()


def response_extract_mock_header_delay(body: dict[str, Any]) -> float:
    for item in response_input_messages(body):
        for text in response_texts(item):
            m = re.search(r"__mock_header_delay_(\d+(?:\.\d+)?)s?__", text)
            if m:
                return float(m.group(1))
    return 0.0


def response_extract_mock_frame_stall(body: dict[str, Any]) -> float:
    for item in response_input_messages(body):
        for text in response_texts(item):
            m = re.search(r"__mock_frame_stall_(\d+(?:\.\d+)?)s?__", text)
            if m:
                return float(m.group(1))
    return 0.0


@router.post("/v1/responses")
async def responses(request: Request):
    body, error_response = await parse_provider_request(request, "openai_responses")
    if error_response is not None:
        return error_response
    assert body is not None
    await apply_fixture_delay()
    session_id = request.headers.get(SESSION_HEADER) or "__global__"
    request.app.state.request_store.record(session_id, body, request.headers)
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
    delay = response_extract_mock_header_delay(body)
    if delay > 0:
        await asyncio.sleep(delay)
    if not body.get("stream"):
        return response
    if response_input_contains(body, "__mock_midstream_error__"):
        return StreamingResponse(
            generate_responses_midstream_error(response, item_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    resp_stall_sec = response_extract_mock_frame_stall(body)
    return StreamingResponse(
        generate_responses_stream(
            response,
            item_id,
            complete=not response_input_contains(body, "__mock_incomplete_stream__"),
            stall_seconds=resp_stall_sec,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
