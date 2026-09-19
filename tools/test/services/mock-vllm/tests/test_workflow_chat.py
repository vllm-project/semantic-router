from chat_request import ChatMessage, ChatRequest
from workflow_chat import (
    build_workflow_plan_content,
    is_workflow_planner_request,
    is_workflow_worker_request,
)


def test_planner_request_is_detected_from_system_prompt():
    req = ChatRequest(
        model="openai/workflow-planner",
        messages=[
            ChatMessage(role="system", content="You are the Router Flow planner."),
            ChatMessage(role="user", content="What is 2+2?"),
        ],
        response_format={"type": "json_object"},
    )
    assert is_workflow_planner_request(req)


def test_worker_request_is_detected_from_step_prompt():
    req = ChatRequest(
        model="openai/gpt-oss-20b",
        messages=[ChatMessage(role="system", content="Router Flow step calculate")],
    )
    assert is_workflow_worker_request(req)


def test_planner_plan_uses_available_worker_model():
    req = ChatRequest(
        model="openai/workflow-planner",
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are the Router Flow planner.\n"
                    "Available worker models, and the only worker models you may use:\n"
                    "openai/gpt-oss-20b\n\n"
                    "Limits:\nkeep the plan short"
                ),
            )
        ],
        response_format={"type": "json_object"},
    )
    content = build_workflow_plan_content(req)
    assert "openai/gpt-oss-20b" in content
    assert "calculate" in content
