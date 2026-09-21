"""Router Flow planner/worker chat replies for the mock provider."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

from chat_request import ChatMessage, ChatRequest
from fastapi.responses import JSONResponse, StreamingResponse

ChatContains = Callable[[ChatRequest, str], bool]
ChatHasToolResult = Callable[[ChatRequest], bool]
ChatRequestsMockTool = Callable[[ChatRequest], bool]
BuildChatUsage = Callable[[ChatRequest, str], dict]
BuildChatResponse = Callable[[ChatRequest, str, dict, int], dict]
MockChatToolResponse = Callable[[ChatRequest, int], dict[str, Any]]
GenerateChatStream = Callable[..., Iterator[str]]
GenerateChatToolStream = Callable[[ChatRequest, int], Iterator[str]]


class ChatControlHelpers:
    def __init__(
        self,
        *,
        chat_contains: ChatContains,
        chat_has_tool_result: ChatHasToolResult,
        chat_requests_mock_tool: ChatRequestsMockTool,
        build_chat_usage: BuildChatUsage,
        build_chat_response: BuildChatResponse,
        mock_chat_tool_response: MockChatToolResponse,
        generate_chat_stream: GenerateChatStream,
        generate_chat_tool_stream: GenerateChatToolStream,
    ) -> None:
        self.chat_contains = chat_contains
        self.chat_has_tool_result = chat_has_tool_result
        self.chat_requests_mock_tool = chat_requests_mock_tool
        self.build_chat_usage = build_chat_usage
        self.build_chat_response = build_chat_response
        self.mock_chat_tool_response = mock_chat_tool_response
        self.generate_chat_stream = generate_chat_stream
        self.generate_chat_tool_stream = generate_chat_tool_stream


def mock_chat_control_response(
    req: ChatRequest, created_ts: int, helpers: ChatControlHelpers
) -> Any | None:
    if helpers.chat_contains(req, "__mock_provider_error__"):
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "mock provider rate limit",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
        )
    workflow_response = mock_workflow_chat_response(req, created_ts, helpers)
    if workflow_response is not None:
        return workflow_response
    if helpers.chat_has_tool_result(req):
        content = "tool result accepted"
        usage = helpers.build_chat_usage(req, content)
        response = helpers.build_chat_response(req, content, usage, created_ts)
        if req.stream:
            return StreamingResponse(
                helpers.generate_chat_stream(req, response, content, usage, created_ts),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return response
    if helpers.chat_requests_mock_tool(req):
        if req.stream:
            return StreamingResponse(
                helpers.generate_chat_tool_stream(req, created_ts),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return helpers.mock_chat_tool_response(req, created_ts)
    return None


def mock_workflow_chat_response(
    req: ChatRequest, created_ts: int, helpers: ChatControlHelpers
) -> dict[str, Any] | None:
    if is_workflow_planner_request(req):
        content = build_workflow_plan_content(req)
        usage = helpers.build_chat_usage(req, content)
        return helpers.build_chat_response(req, content, usage, created_ts)
    if is_workflow_final_synthesis_request(req) or (
        is_workflow_worker_request(req) and helpers.chat_has_tool_result(req)
    ):
        content = "The sum of 2 and 2 is 4."
        usage = helpers.build_chat_usage(req, content)
        return helpers.build_chat_response(req, content, usage, created_ts)
    if (
        is_workflow_worker_request(req)
        and req.tools
        and not helpers.chat_has_tool_result(req)
    ):
        return mock_workflow_tool_response(req, created_ts, helpers)
    return None


def mock_workflow_tool_response(
    req: ChatRequest, created_ts: int, helpers: ChatControlHelpers
) -> dict[str, Any]:
    tool_name = "lookup"
    if req.tools:
        func = req.tools[0].get("function", {})
        if (
            isinstance(func, dict)
            and isinstance(func.get("name"), str)
            and func["name"]
        ):
            tool_name = func["name"]
    response = helpers.mock_chat_tool_response(req, created_ts)
    response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = tool_name
    return response


def message_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return ""


def is_workflow_planner_request(req: ChatRequest) -> bool:
    if not req.response_format or req.response_format.get("type") != "json_object":
        return False
    return any(
        "You are the Router Flow planner" in message_text(message)
        for message in req.messages
    )


def is_workflow_final_synthesis_request(req: ChatRequest) -> bool:
    return any(
        "Router Flow final synthesizer" in message_text(message)
        for message in req.messages
    )


def is_workflow_worker_request(req: ChatRequest) -> bool:
    return any("Router Flow step" in message_text(message) for message in req.messages)


def extract_worker_models_from_planner_prompt(req: ChatRequest) -> list[str]:
    marker = "Available worker models, and the only worker models you may use:"
    for message in req.messages:
        content = message_text(message)
        if marker not in content:
            continue
        section = content.split(marker, 1)[1]
        section = section.split("\n\nLimits:", 1)[0]
        models = [line.strip() for line in section.splitlines() if line.strip()]
        if models:
            return models
    return ["openai/gpt-oss-20b"]


def build_workflow_plan_content(req: ChatRequest) -> str:
    worker_model = extract_worker_models_from_planner_prompt(req)[0]
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
