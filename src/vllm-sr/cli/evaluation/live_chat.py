"""Strict OpenAI Chat Completions adapter for live evaluation probes."""

from __future__ import annotations

from typing import Any

from cli.evaluation.contract_validation import derived_portable_id
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.http_client import EvaluationHTTPClient, HTTPResult
from cli.evaluation.target_arm_resolution import resolve_target_arm
from cli.evaluation.target_contracts import EvaluationTargetArm


def message_payloads(case: CaseVisible) -> list[dict[str, Any]]:
    return [
        message.model_dump(mode="json", exclude_none=True) for message in case.messages
    ]


def response_content(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    for field in ("content", "reasoning", "reasoning_content"):
        value = message.get(field)
        if isinstance(value, str):
            return value
    return None


def token_usage(payload: dict[str, Any] | None) -> tuple[int | None, int | None]:
    """Read the exact Chat Completions usage contract."""

    usage = payload.get("usage") if payload else None
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    return (
        input_tokens if isinstance(input_tokens, int) and input_tokens >= 0 else None,
        (
            output_tokens
            if isinstance(output_tokens, int) and output_tokens >= 0
            else None
        ),
    )


def grade_response(result: HTTPResult, labels: CaseGrading) -> float | None:
    content = response_content(result.payload)
    if labels.expected_answer is None or content is None:
        return None
    actual = " ".join(content.split())
    expected = " ".join(labels.expected_answer.split())
    return float(actual == expected)


def chat_request(
    client: EvaluationHTTPClient,
    endpoint: str,
    case: CaseVisible,
    model: str,
    track_id: str,
    attempt_id: str,
) -> HTTPResult:
    return client.post(
        endpoint,
        {
            "model": model,
            "messages": message_payloads(case),
            "temperature": 0,
            "stream": False,
        },
        track_id=track_id,
        case_id=case.id,
        attempt_id=attempt_id,
    )


def execute_chat_cases(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    model: str,
) -> dict[str, HTTPResult]:
    return {
        case.id: chat_request(
            client,
            endpoint,
            case,
            model,
            "multimodal",
            derived_portable_id("attempt", case.id),
        )
        for case in cases
    }


def arm_accounting(
    arm: EvaluationTargetArm,
    result: HTTPResult,
) -> tuple[int | None, int | None, float | None]:
    input_tokens, output_tokens = token_usage(result.payload)
    if input_tokens is None or output_tokens is None:
        return input_tokens, output_tokens, None
    runtime_cost = (
        input_tokens * arm.input_cost_per_million_tokens_usd
        + output_tokens * arm.output_cost_per_million_tokens_usd
    ) / 1_000_000
    return input_tokens, output_tokens, runtime_cost


def result_accounting(
    result: HTTPResult,
    arms: tuple[EvaluationTargetArm, ...],
) -> tuple[int | None, int | None, float | None]:
    arm = resolve_target_arm(result.headers.get("x-vsr-selected-model"), arms)
    if arm is None:
        input_tokens, output_tokens = token_usage(result.payload)
        return input_tokens, output_tokens, None
    return arm_accounting(arm, result)
