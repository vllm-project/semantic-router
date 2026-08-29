"""OpenAI chat adapters for direct-arm and correlated routed execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cli.evaluation.contracts import CaseGrading, CaseVisible, EvaluationTargetArm
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.http_client import EvaluationHTTPClient, HTTPResult


@dataclass(frozen=True)
class RoutedChatEvidence:
    records: list[ExecutionRecord]
    results: dict[str, HTTPResult]


@dataclass(frozen=True)
class DirectArmEvidence:
    records: list[ExecutionRecord]
    results: dict[tuple[str, str], HTTPResult]


def _messages(case: CaseVisible) -> list[dict[str, Any]]:
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
    value = message.get("content")
    return value if isinstance(value, str) else None


def token_usage(payload: dict[str, Any] | None) -> tuple[int | None, int | None]:
    usage = payload.get("usage") if payload else None
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
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
    actual = " ".join(content.casefold().split())
    expected = " ".join(labels.expected_answer.casefold().split())
    return float(actual == expected)


def chat_request(
    client: EvaluationHTTPClient,
    endpoint: str,
    case: CaseVisible,
    model: str,
) -> HTTPResult:
    return client.post(
        endpoint,
        {
            "model": model,
            "messages": _messages(case),
            "temperature": 0,
            "stream": False,
        },
    )


def _runtime_cost(
    arm: EvaluationTargetArm,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    return (
        input_tokens * arm.input_cost_per_million_tokens_usd
        + output_tokens * arm.output_cost_per_million_tokens_usd
    ) / 1_000_000


def direct_arm_evidence(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    arms: tuple[EvaluationTargetArm, ...],
) -> DirectArmEvidence:
    records: list[ExecutionRecord] = []
    results: dict[tuple[str, str], HTTPResult] = {}
    for case in cases:
        for arm in arms:
            result = chat_request(client, endpoint, case, arm.model)
            results[(case.id, arm.id)] = result
            input_tokens, output_tokens = token_usage(result.payload)
            records.append(
                ExecutionRecord(
                    id=f"model-pool-{case.id}-{arm.id}",
                    track_id="model_pool",
                    case_id=case.id,
                    attempt_id=f"attempt-{case.id}-{arm.id}",
                    status="succeeded" if result.success else "failed",
                    arm_id=arm.id,
                    success=result.success,
                    latency_ms=result.latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    runtime_cost=_runtime_cost(arm, input_tokens, output_tokens),
                    error=result.error,
                )
            )
    return DirectArmEvidence(records=records, results=results)


def _selected_arm(
    result: HTTPResult, arms: tuple[EvaluationTargetArm, ...]
) -> EvaluationTargetArm | None:
    selected = result.headers.get("x-vsr-selected-model")
    return next(
        (arm for arm in arms if selected in {arm.id, arm.model}),
        None,
    )


def result_accounting(
    result: HTTPResult,
    arms: tuple[EvaluationTargetArm, ...],
) -> tuple[int | None, int | None, float | None]:
    input_tokens, output_tokens = token_usage(result.payload)
    arm = _selected_arm(result, arms)
    return (
        input_tokens,
        output_tokens,
        _runtime_cost(arm, input_tokens, output_tokens) if arm else None,
    )


def _joint_record(
    case: CaseVisible,
    result: HTTPResult,
    arms: tuple[EvaluationTargetArm, ...],
) -> ExecutionRecord:
    arm = _selected_arm(result, arms)
    input_tokens, output_tokens = token_usage(result.payload)
    if arm is None:
        return ExecutionRecord(
            id=f"joint-{case.id}",
            track_id="joint",
            case_id=case.id,
            attempt_id=f"attempt-{case.id}",
            status="unavailable",
            success=result.success,
            latency_ms=result.latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=(
                result.error
                or "x-vsr-selected-model is absent or not in the server-owned arm set"
            ),
        )
    return ExecutionRecord(
        id=f"joint-{case.id}",
        track_id="joint",
        case_id=case.id,
        attempt_id=f"attempt-{case.id}",
        status="succeeded" if result.success else "failed",
        selected_arm_id=arm.id,
        algorithm=result.headers.get("x-vsr-selected-algorithm"),
        success=result.success,
        latency_ms=result.latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        runtime_cost=_runtime_cost(arm, input_tokens, output_tokens),
        error=result.error,
    )


def routed_chat_records(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    arms: tuple[EvaluationTargetArm, ...],
    entrypoint_model: str,
) -> RoutedChatEvidence:
    records: list[ExecutionRecord] = []
    results: dict[str, HTTPResult] = {}
    for case in cases:
        result = chat_request(client, endpoint, case, entrypoint_model)
        results[case.id] = result
        records.append(_joint_record(case, result, arms))
    return RoutedChatEvidence(records=records, results=results)
