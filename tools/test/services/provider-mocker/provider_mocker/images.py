"""Buffered native Images generation fixture with a real, fixed PNG payload."""

from fastapi import APIRouter, Request

from .provider_boundary import (
    SESSION_HEADER,
    invalid_request_response,
    parse_provider_request,
)
from .settings import apply_fixture_delay

# A valid 1x1 RGBA PNG. Image bytes are fixed; no model or image library is loaded.
FIXTURE_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
MAX_IMAGE_COUNT = 4
router = APIRouter()


@router.post("/v1/images/generations")
async def images(request: Request):
    body, error = await parse_provider_request(request, "openai_images")
    if error is not None:
        return error
    if body.get("response_format", "b64_json") != "b64_json":
        return invalid_request_response(
            "only b64_json image responses are supported", "response_format"
        )
    count = body.get("n", 1)
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not 1 <= count <= MAX_IMAGE_COUNT
    ):
        return invalid_request_response("n must be an integer between 1 and 4", "n")
    if not isinstance(body["prompt"], str) or not body["prompt"].strip():
        return invalid_request_response("prompt must be a non-empty string", "prompt")
    session = request.headers.get(SESSION_HEADER) or "__global__"
    request.app.state.request_store.record(session, body, request.headers)
    await apply_fixture_delay()
    return {
        "created": 1,
        "data": [
            {"b64_json": FIXTURE_PNG, "revised_prompt": body["prompt"]}
            for _ in range(count)
        ],
    }
