import json
import math
import time

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.2
    stream: bool | None = False


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def build_chat_content(req: ChatRequest) -> str:
    roles = [m.role for m in req.messages]
    system_messages = [m.content for m in req.messages if m.role == "system"]
    user_messages = [m.content for m in req.messages if m.role == "user"]

    return json.dumps(
        {
            "mock": "mock-vllm",
            "model": req.model,
            "roles": roles,
            "system": system_messages,
            "user": user_messages,
            "total_messages": len(req.messages),
        },
        separators=(",", ":"),
        sort_keys=True,
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
    req: ChatRequest, response: dict, content: str, usage: dict, created_ts: int
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
    yield build_chat_stream_chunk(req, response_id, created_ts, {}, "stop", usage)
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "openai/gpt-oss-20b", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request, http_response: Response):
    created_ts = int(time.time())
    content = build_chat_content(req)
    usage = build_chat_usage(req, content)
    response = build_chat_response(req, content, usage, created_ts)
    # Reflect the received Authorization back to the caller so end-to-end tests can
    # assert what actually reached the upstream: a forward_authorization_header
    # backend must receive the caller's verbatim token, while a static-key backend
    # must receive the router-injected key. This is a test-only echo.
    echoed_auth = request.headers.get("authorization", "")
    if not req.stream:
        http_response.headers["x-echo-authorization"] = echoed_auth
        return response

    return StreamingResponse(
        generate_chat_stream(req, response, content, usage, created_ts),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "x-echo-authorization": echoed_auth,
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
