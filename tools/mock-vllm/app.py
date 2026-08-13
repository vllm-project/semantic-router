import json
import math
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.2
    stream: bool | None = False
    response_format: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


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


def is_hallucination_detection_request(req: ChatRequest) -> bool:
    if not req.response_format or req.response_format.get("type") != "json_schema":
        return False
    schema = req.response_format.get("json_schema", {})
    return schema.get("name") == "hallucination_detection"


def extract_answer_under_review(req: ChatRequest) -> str:
    for m in req.messages:
        if m.role == "user" and m.content and "Answer to verify:\n" in m.content:
            return m.content.split("Answer to verify:\n")[-1]
    return ""


def is_workflow_planner_request(req: ChatRequest) -> bool:
    if not req.response_format:
        return False
    if req.response_format.get("type") != "json_object":
        return False
    for message in req.messages:
        if message.content and "You are the Router Flow planner" in message.content:
            return True
    return False


def is_workflow_final_synthesis_request(req: ChatRequest) -> bool:
    for message in req.messages:
        if message.content and "Router Flow final synthesizer" in message.content:
            return True
    return False


def is_workflow_worker_request(req: ChatRequest) -> bool:
    for message in req.messages:
        if message.content and "Router Flow step" in message.content:
            return True
    return False


def has_assistant_tool_calls(req: ChatRequest) -> bool:
    return any(
        message.role == "assistant" and message.tool_calls for message in req.messages
    )


def extract_worker_models_from_planner_prompt(req: ChatRequest) -> list[str]:
    for message in req.messages:
        if message.role != "user" or not message.content:
            continue
        marker = "Available worker models, and the only worker models you may use:"
        if marker not in message.content:
            continue
        section = message.content.split(marker, 1)[1]
        section = section.split("\n\nLimits:", 1)[0]
        models = [line.strip() for line in section.splitlines() if line.strip()]
        if models:
            return models
    return ["openai/gpt-oss-20b"]


def build_workflow_plan_content(req: ChatRequest) -> str:
    worker_models = extract_worker_models_from_planner_prompt(req)
    worker_model = worker_models[0]
    return json.dumps(
        {
            "steps": [
                {
                    "id": "calculate",
                    "role": "worker",
                    "models": [worker_model],
                    "prompt": "Use the calculate tool to solve the user request.",
                }
            ],
            "final": {"prompt": "Return the final answer from the worker."},
        },
        separators=(",", ":"),
    )


def has_tool_message(req: ChatRequest) -> bool:
    return any(m.role == "tool" for m in req.messages)


def build_workflow_completion(_req: ChatRequest) -> str:
    return "The sum of 2 and 2 is 4."


def build_hallucination_detection_content(req: ChatRequest) -> str:
    answer = extract_answer_under_review(req)
    flagged_text = answer[:80] if answer else "mocked hallucination"
    return json.dumps(
        {
            "hallucinated_spans": [
                {
                    "text": flagged_text,
                    "category": "unsupported_addition",
                    "subcategory": "claim",
                }
            ]
        }
    )


def build_chat_usage(req: ChatRequest, content: str) -> dict:
    prompt_parts = []
    for m in req.messages:
        if isinstance(m.content, str):
            prompt_parts.append(m.content)
    prompt_text = "\n".join(prompt_parts)
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(content)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def should_emit_tool_call(req: ChatRequest) -> bool:
    if not req.tools:
        return False
    if not is_workflow_worker_request(req):
        return False
    if has_tool_message(req):
        return False
    if is_workflow_final_synthesis_request(req):
        return False
    return not has_assistant_tool_calls(req)


def build_tool_call_response(req: ChatRequest, usage: dict, created_ts: int) -> dict:
    tool_name = "lookup"
    if req.tools and len(req.tools) > 0:
        func = req.tools[0].get("function", {})
        tool_name = func.get("name", "lookup")
    tool_call = {
        "id": "call_mock_123",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps({"query": "sample query"}),
        },
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
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                },
                "finish_reason": "tool_calls",
                "logprobs": None,
            }
        ],
        "usage": usage,
        "token_usage": usage,
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
    return {
        "data": [
            {"id": "openai/gpt-oss-20b", "object": "model"},
            {"id": "openai/workflow-planner", "object": "model"},
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    created_ts = int(time.time())
    if is_hallucination_detection_request(req):
        content = build_hallucination_detection_content(req)
        usage = build_chat_usage(req, content)
        response = build_chat_response(req, content, usage, created_ts)
    elif is_workflow_planner_request(req):
        content = build_workflow_plan_content(req)
        usage = build_chat_usage(req, content)
        response = build_chat_response(req, content, usage, created_ts)
    elif is_workflow_final_synthesis_request(req) or has_tool_message(req):
        content = build_workflow_completion(req)
        usage = build_chat_usage(req, content)
        response = build_chat_response(req, content, usage, created_ts)
    elif should_emit_tool_call(req):
        # TODO: handle streaming tool_calls if a streaming workflow E2E test is added.
        usage = build_chat_usage(req, "")
        response = build_tool_call_response(req, usage, created_ts)
        return response
    else:
        content = build_chat_content(req)
        usage = build_chat_usage(req, content)
        response = build_chat_response(req, content, usage, created_ts)
    if not req.stream:
        return response

    return StreamingResponse(
        generate_chat_stream(req, response, content, usage, created_ts),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
