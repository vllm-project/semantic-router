"""Buffered native Speech API fixture with a real, fixed WAV payload."""

import struct

from fastapi import APIRouter, Request
from fastapi.responses import Response

from .provider_boundary import (
    SESSION_HEADER,
    invalid_request_response,
    parse_provider_request,
)
from .settings import apply_fixture_delay

SAMPLE_RATE = 24000
SAMPLE_COUNT = SAMPLE_RATE // 10


def _silent_wav(sample_rate: int, sample_count: int) -> bytes:
    data_size = sample_count * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + bytes(data_size)


# 0.1 s of 16-bit mono silence. Audio bytes are fixed; no TTS model is loaded.
FIXTURE_WAV = _silent_wav(SAMPLE_RATE, SAMPLE_COUNT)
router = APIRouter()


@router.post("/v1/audio/speech")
async def speech(request: Request):
    raw_body = await request.body()
    body, error = await parse_provider_request(request, "openai_speech")
    if error is not None:
        return error
    if body.get("stream") is True or body.get("stream_format", "audio") != "audio":
        return invalid_request_response(
            "only non-streaming speech responses are supported", "stream"
        )
    if body.get("response_format", "wav") != "wav":
        return invalid_request_response(
            "only wav speech responses are supported", "response_format"
        )
    if not isinstance(body["input"], str) or not body["input"].strip():
        return invalid_request_response("input must be a non-empty string", "input")
    session = request.headers.get(SESSION_HEADER) or "__global__"
    request.app.state.request_store.record(session, body, request.headers, raw_body)
    await apply_fixture_delay()
    return Response(content=FIXTURE_WAV, media_type="audio/wav")
