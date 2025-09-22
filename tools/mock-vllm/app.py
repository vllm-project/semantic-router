from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

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
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    content = f"[mock-{req.model}] You said: {last_user}"
    return {
        "id": "cmpl-mock-123",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
