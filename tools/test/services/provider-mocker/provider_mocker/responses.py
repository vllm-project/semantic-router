"""Native Responses API HTTP route."""

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .fault_schedule import FAULT_INJECTED_HEADER, get_fault_keys
from .provider_boundary import SESSION_HEADER, parse_provider_request
from .responses_wire import (
    build_responses_custom_tool_response,
    build_responses_image_generation_response,
    build_responses_response,
    build_responses_tool_response,
    generate_responses_custom_tool_stream,
    generate_responses_image_generation_stream,
    generate_responses_midstream_error,
    generate_responses_stream,
    generate_responses_tool_stream,
    response_has_tool_result,
    response_input_contains,
    response_requests_image_generation,
)
from .settings import apply_fixture_delay

router = APIRouter()


@router.post("/v1/responses")
async def responses(request: Request):
    raw_body = await request.body()
    body, error_response = await parse_provider_request(request, "openai_responses")
    if error_response is not None:
        return error_response
    assert body is not None
    await apply_fixture_delay()
    session_id = request.headers.get(SESSION_HEADER) or "__global__"
    request.app.state.request_store.record(
        session_id, body, request.headers, raw_body, request.url.path
    )

    fault_tracker = getattr(request.app.state, "fault_tracker", None)
    fault = None
    if fault_tracker is not None:
        schedule_key, counter_key = get_fault_keys(request.headers)
        if schedule_key or counter_key:
            _, fault = fault_tracker.record_call_and_match(schedule_key, counter_key)
            if fault is not None:
                if fault.delay and fault.delay > 0:
                    await asyncio.sleep(fault.delay)
                if fault.status is not None:
                    return JSONResponse(
                        status_code=fault.status,
                        content={
                            "error": {
                                "message": f"fault schedule injected {fault.status}",
                                "type": "fault_schedule_injected",
                                "param": None,
                                "code": f"fault_{fault.status}",
                            }
                        },
                        headers={FAULT_INJECTED_HEADER: "true"},
                    )
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
    if response_input_contains(body, "__mock_responses_custom_tool__") and body.get(
        "tools"
    ):
        if body.get("stream"):
            return StreamingResponse(
                generate_responses_custom_tool_stream(body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return build_responses_custom_tool_response(body)
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
    fault_headers = {FAULT_INJECTED_HEADER: "true"} if fault is not None else {}
    if not body.get("stream"):
        if fault is not None:
            return JSONResponse(content=response, headers=fault_headers)
        return response
    if response_input_contains(body, "__mock_midstream_error__"):
        return StreamingResponse(
            generate_responses_midstream_error(response, item_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                **fault_headers,
            },
        )
    is_complete = not response_input_contains(body, "__mock_incomplete_stream__")
    if fault is not None and fault.stream_cut_short:
        is_complete = False
    return StreamingResponse(
        generate_responses_stream(
            response,
            item_id,
            complete=is_complete,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            **fault_headers,
        },
    )
