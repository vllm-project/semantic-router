import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .cache import apply_cache_usage, cache_prefix_hash, has_cache_control
from .fault_schedule import FAULT_INJECTED_HEADER, get_fault_keys
from .messages_wire import build_message, contains_text, stream_message
from .provider_boundary import SESSION_HEADER, parse_provider_request
from .settings import apply_fixture_delay

router = APIRouter()


@router.post("/v1/messages")
async def messages(request: Request):
    raw_body = await request.body()
    body, error = await parse_provider_request(request, "anthropic_messages")
    if error is not None:
        return error
    session = request.headers.get(SESSION_HEADER) or "__global__"
    request.app.state.request_store.record(
        session, body, request.headers, raw_body, request.url.path
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
                            "type": "error",
                            "error": {
                                "type": "fault_schedule_injected",
                                "message": f"fault schedule injected {fault.status}",
                            },
                            "request_id": "req_mock_fault",
                        },
                        headers={FAULT_INJECTED_HEADER: "true"},
                    )

    await apply_fixture_delay()
    if contains_text(body["messages"], "__mock_provider_error__"):
        return JSONResponse(
            status_code=429,
            content={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "mock provider rate limit",
                },
                "request_id": "req_mock_rate_limit",
            },
        )
    response = build_message(body)
    if has_cache_control(body):
        seen = request.app.state.cache_tracker.mark(session, cache_prefix_hash(body))
        apply_cache_usage(response, request_had_cache_control=True, prefix_seen=seen)
    fault_headers = {FAULT_INJECTED_HEADER: "true"} if fault is not None else {}
    if not body.get("stream"):
        if fault is not None:
            return JSONResponse(content=response, headers=fault_headers)
        return response
    failure = ""
    if contains_text(body["messages"], "__mock_incomplete_stream__") or (
        fault is not None and fault.stream_cut_short
    ):
        failure = "incomplete"
    if contains_text(body["messages"], "__mock_midstream_error__"):
        failure = "error"
    return StreamingResponse(
        stream_message(response, failure),
        media_type="text/event-stream",
        headers=fault_headers if fault_headers else None,
    )
