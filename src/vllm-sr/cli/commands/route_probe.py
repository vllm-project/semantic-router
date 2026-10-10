"""Assess final assistant delivery in non-streaming route probe responses."""

from __future__ import annotations

import json
from typing import Any


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _function_call(value: Any) -> bool:
    if not (
        isinstance(value, dict)
        and _nonempty_text(value.get("name"))
        and isinstance(value.get("arguments"), str)
    ):
        return False
    try:
        return isinstance(json.loads(value["arguments"]), dict)
    except ValueError:
        return False


def _delivery_kind(message: Any) -> str | None:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    if _nonempty_text(message.get("refusal")):
        return "refusal"
    content = message.get("content")
    if _nonempty_text(content):
        return "text"
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and _nonempty_text(part.get("text")):
                return "text"
            if part.get("type") == "refusal" and _nonempty_text(part.get("refusal")):
                return "refusal"
    calls = message.get("tool_calls")
    if (
        isinstance(calls, list)
        and calls
        and all(
            isinstance(call, dict)
            and _nonempty_text(call.get("id"))
            and call.get("type") == "function"
            and _function_call(call.get("function"))
            for call in calls
        )
    ):
        return "tool_calls"
    if _function_call(message.get("function_call")):
        return "function_call"
    return None


def _choice_delivery(choice: Any, index: int) -> dict[str, Any]:
    choice = choice if isinstance(choice, dict) else {}
    kind = _delivery_kind(choice.get("message"))
    finish_reason = choice.get("finish_reason")
    result = {"index": index, "finish_reason": finish_reason, "delivery": kind}
    if finish_reason == "length":
        result["error"] = "Completion was truncated at the token or context limit."
    elif finish_reason not in ("stop", "tool_calls", "function_call", "content_filter"):
        result["error"] = "Completion has no recognized terminal finish reason."
    elif kind is None:
        result["error"] = "No final assistant text, valid tool call, or refusal."
    elif finish_reason == "content_filter" and kind != "refusal":
        result["error"] = "Content filtering ended the completion without a refusal."
    return result


def delivery_assertion(response_body: Any) -> dict[str, Any]:
    """Require complete delivery, retaining diagnostics without copying reasoning."""
    choices = response_body.get("choices") if isinstance(response_body, dict) else None
    if not isinstance(choices, list) or not choices:
        actual = {
            "choices": [],
            "error": "Expected a nonempty chat completion choices array.",
        }
        passed = False
    else:
        reports = [
            _choice_delivery(choice, index) for index, choice in enumerate(choices)
        ]
        actual = {"choices": reports}
        passed = all("error" not in report for report in reports)
    return {
        "field": "response.body.delivery",
        "expected": "Complete assistant text, tool calls, or refusal in every choice.",
        "actual": actual,
        "passed": passed,
    }
