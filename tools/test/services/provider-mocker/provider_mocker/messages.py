"""Native Messages endpoint with independent schema and cache observations."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .cache import apply_cache_usage, cache_prefix_hash, has_cache_control
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
    request.app.state.request_store.record(session, body, request.headers, raw_body)
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
    if not body.get("stream"):
        return response
    failure = ""
    if contains_text(body["messages"], "__mock_incomplete_stream__"):
        failure = "incomplete"
    if contains_text(body["messages"], "__mock_midstream_error__"):
        failure = "error"
    return StreamingResponse(
        stream_message(response, failure), media_type="text/event-stream"
    )
