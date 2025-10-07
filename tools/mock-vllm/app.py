import math
import time
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "openai/gpt-oss-20b", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Very simple echo-like behavior
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    content = f"[mock-{req.model}] You said: {last_user}"

    # Rough token estimation: ~1 token per 4 characters (ceil)
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    prompt_text = "\n".join(
        m.content for m in req.messages if isinstance(m.content, str)
    )
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(content)
    total_tokens = prompt_tokens + completion_tokens

    created_ts = int(time.time())

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        # Optional details fields some clients read when using caching/reasoning
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }

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
        # Some SDKs look for token_usage; keep it as an alias for convenience.
        "token_usage": usage,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
