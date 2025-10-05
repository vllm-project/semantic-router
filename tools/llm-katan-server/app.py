import math
import time
import os
import requests
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Configuration
MODEL = os.getenv("MODEL", "Qwen/Qwen2-0.5B-Instruct")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", MODEL)
LLM_KATAN_URL = os.getenv("LLM_KATAN_URL", "http://localhost:8001")

# Check if HuggingFace token is set
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    print("Warning: HUGGINGFACE_HUB_TOKEN not set. Some models may require authentication.")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": SERVED_MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        # Forward request to llm-katan backend
        llm_katan_request = {
            "model": MODEL,
            "messages": [{"role": msg.role, "content": msg.content} for msg in req.messages],
            "temperature": req.temperature,
        }
        
        if req.max_tokens:
            llm_katan_request["max_tokens"] = req.max_tokens
        
        # Make request to llm-katan
        response = requests.post(
            f"{LLM_KATAN_URL}/v1/chat/completions",
            json=llm_katan_request,
            timeout=30
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LLM Katan error: {response.text}"
            )
        
        result = response.json()
        
        # Update the model name in response to match our served model name
        result["model"] = req.model
        
        return result
        
    except requests.exceptions.RequestException as e:
        # Fallback to simple echo behavior if llm-katan is not available
        print(f"Warning: LLM Katan not available ({e}), using fallback response")
        
        # Simple echo-like behavior as fallback
        last_user = next(
            (m.content for m in reversed(req.messages) if m.role == "user"), ""
        )
        content = f"[katan-{req.model}] You said: {last_user}"

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
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }

        return {
            "id": "cmpl-katan-123",
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model,
            "system_fingerprint": "llm-katan-server",
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
