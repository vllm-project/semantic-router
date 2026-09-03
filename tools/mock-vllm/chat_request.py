"""Typed Chat Completions view and deterministic request echo behavior."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.2
    stream: bool | None = False
    response_format: dict | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    logprobs: bool | None = False
    top_logprobs: int | None = None


def build_chat_content(request: ChatRequest) -> str:
    roles = [message.role for message in request.messages]
    developer_messages = [
        str(message.content)
        for message in request.messages
        if message.role == "developer"
    ]
    system_messages = [
        str(message.content) for message in request.messages if message.role == "system"
    ]
    user_messages = [
        str(message.content) for message in request.messages if message.role == "user"
    ]

    echo = {
        "mock": "mock-vllm",
        "protocol": "chat_completions",
        "model": request.model,
        "roles": roles,
        "developer": developer_messages,
        "system": system_messages,
        "user": user_messages,
        "total_messages": len(request.messages),
        "request_fields": sorted(request.model_fields_set),
    }
    if request.response_format is not None:
        echo["structured_output"] = request.response_format
    return json.dumps(echo, separators=(",", ":"), sort_keys=True)
