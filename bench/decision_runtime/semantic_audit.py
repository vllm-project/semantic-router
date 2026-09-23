"""Untimed, request-relative parity gate for synthetic Decision workflows.

The old preview projection is used only to validate its wire contract. Numeric
comparisons use the original old probabilities and token counts, never the
projected confidence statistic.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any

from pydantic import ValidationError

# semantic_cases initializes the source checkout's runtime-contract path.
# isort: off
from .semantic_cases import RequestSpec, WorkloadCase
from .legacy_projection import project_legacy_preview
from .report import percentile
from .transport import Endpoint, OPENER, _consume
from decision_runtime.contracts import (
    ResponseContractError,
    SystemOneBatchRequest,
    SystemOneBatchResponse,
    SystemOneResponse,
    validate_batch_response_for_request,
    validate_response_for_request,
)

# isort: on

NOUL_DECISION_THRESHOLD = 0.5


@dataclass(frozen=True)
class AuditExchange:
    request_sha256: str
    response_sha256: str | None
    status_code: int | None
    error_code: str | None
    response: SystemOneResponse | SystemOneBatchResponse | None
    raw_question_input_tokens: dict[str, int] | None
    raw_legacy_answers: dict[str, dict[str, object]] | None

    def public_record(self) -> dict[str, object]:
        return {
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "status_code": self.status_code,
            "error_code": self.error_code,
        }


def _exchange(endpoint: Endpoint, spec: RequestSpec, timeout: float) -> AuditExchange:
    """Issue one untimed call and retain validated answers only in memory."""

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if endpoint.token is not None:
        headers["Authorization"] = f"Bearer {endpoint.token}"
    request = urllib.request.Request(
        endpoint.url, data=spec.body, headers=headers, method="POST"
    )
    status_code: int | None = None
    content: bytes | None = None
    response_sha256: str | None = None
    error_code: str | None = None
    response: SystemOneResponse | SystemOneBatchResponse | None = None
    raw_tokens: dict[str, int] | None = None
    raw_legacy_answers: dict[str, dict[str, object]] | None = None
    try:
        with OPENER.open(request, timeout=timeout) as wire_response:
            status_code = wire_response.status
            content, response_sha256 = _consume(wire_response)
    except urllib.error.HTTPError as error:
        status_code = error.code
        with error:
            content, response_sha256 = _consume(error)
    except (urllib.error.URLError, TimeoutError, OSError):
        error_code = "transport_error"

    if error_code is None and status_code != HTTPStatus.OK:
        error_code = f"http_{status_code}"
    elif error_code is None and content is None:
        error_code = "response_too_large"
    elif error_code is None:
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            error_code = "invalid_json"
        else:
            try:
                if isinstance(spec.request, SystemOneBatchRequest):
                    if endpoint.response_mode != "decision_v1":
                        raise ValueError("legacy preview supports singles only")
                    response = SystemOneBatchResponse.model_validate(parsed)
                    validate_batch_response_for_request(spec.request, response)
                else:
                    if endpoint.response_mode == "legacy_preview":
                        projected = project_legacy_preview(parsed, spec.request)
                        raw_legacy_answers = parsed["answers"]
                        raw_tokens = {
                            question_id: answer["input_tokens"]
                            for question_id, answer in parsed["answers"].items()
                        }
                        parsed = projected
                    elif endpoint.response_mode != "decision_v1":
                        raise ValueError("unsupported benchmark response mode")
                    response = SystemOneResponse.model_validate(parsed)
                    validate_response_for_request(spec.request, response)
            except (
                ValidationError,
                ResponseContractError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                error_code = "response_contract"
                response = None
                raw_tokens = None
                raw_legacy_answers = None

    return AuditExchange(
        request_sha256=spec.sha256,
        response_sha256=response_sha256,
        status_code=status_code,
        error_code=error_code,
        response=response,
        raw_question_input_tokens=raw_tokens,
        raw_legacy_answers=raw_legacy_answers,
    )


def _request_semantics_match(case: WorkloadCase) -> bool:
    new = case.new_request.request
    if len(case.old_singles) != case.state_count:
        return False
    if isinstance(new, SystemOneBatchRequest):
        if [item.id for item in new.states] != [
            spec.state_id for spec in case.old_singles
        ]:
            return False
        states = [item.state for item in new.states]
    else:
        if (
            case.state_count != 1
            or case.new_request.state_id != case.old_singles[0].state_id
        ):
            return False
        states = [new.state]
    return all(
        spec.request.questions == new.questions and spec.request.state == state
        for spec, state in zip(case.old_singles, states, strict=True)
    )


def _mismatch(
    code: str, *, state_id: str | None = None, question_id: str | None = None
) -> dict[str, str]:
    row = {"code": code}
    if state_id is not None:
        row["state_id"] = state_id
    if question_id is not None:
        row["question_id"] = question_id
    return row


def _compare_answer(
    old: Any,
    new: Any,
    *,
    tolerance: float,
    state_id: str,
    question_id: str,
) -> tuple[dict[str, object], list[dict[str, str]], list[float]]:
    """Compare Decision meaning without treating old confidence as equivalent."""

    mismatches: list[dict[str, str]] = []
    deltas: list[float] = []
    row: dict[str, object] = {"question_id": question_id, "type": old.type}
    if old.type != new.type:
        mismatches.append(
            _mismatch(
                "answer_type_mismatch", state_id=state_id, question_id=question_id
            )
        )
        row["new_type"] = new.type
        return row, mismatches, deltas

    if old.type == "noul":
        delta = abs(old.noul - new.noul)
        deltas.append(delta)
        row.update(
            old_probability=old.noul,
            new_probability=new.noul,
            absolute_probability_delta=delta,
            old_threshold_0_5_outcome=old.noul >= NOUL_DECISION_THRESHOLD,
            new_threshold_0_5_outcome=new.noul >= NOUL_DECISION_THRESHOLD,
            threshold_0_5_outcome_mismatch_diagnostic=(
                (old.noul >= NOUL_DECISION_THRESHOLD)
                != (new.noul >= NOUL_DECISION_THRESHOLD)
            ),
        )
        if delta > tolerance:
            mismatches.append(
                _mismatch(
                    "probability_tolerance", state_id=state_id, question_id=question_id
                )
            )
    else:
        probability_deltas = {
            key: abs(value - new.probabilities[key])
            for key, value in old.probabilities.items()
        }
        deltas.extend(probability_deltas.values())
        row.update(
            old_probabilities=old.probabilities,
            new_probabilities=new.probabilities,
            absolute_probability_deltas=probability_deltas,
            maximum_probability_delta=max(probability_deltas.values()),
        )
        if any(delta > tolerance for delta in probability_deltas.values()):
            mismatches.append(
                _mismatch(
                    "probability_tolerance", state_id=state_id, question_id=question_id
                )
            )
        if old.type == "choice":
            row.update(old_outcome=old.choice, new_outcome=new.choice)
            if old.choice != new.choice:
                mismatches.append(
                    _mismatch(
                        "categorical_outcome_mismatch",
                        state_id=state_id,
                        question_id=question_id,
                    )
                )
        else:
            score_delta = abs(old.score - new.score)
            score_tolerance = tolerance * (len(old.probabilities) - 1)
            row.update(
                old_score=old.score,
                new_score=new.score,
                absolute_score_delta=score_delta,
                score_tolerance=score_tolerance,
            )
            if score_delta > score_tolerance:
                mismatches.append(
                    _mismatch(
                        "score_tolerance", state_id=state_id, question_id=question_id
                    )
                )
    return row, mismatches, deltas


def audit_case(
    case: WorkloadCase,
    old: Endpoint,
    new_single: Endpoint,
    new_batch: Endpoint,
    *,
    timeout: float,
    probability_tolerance: float,
) -> tuple[dict[str, object], list[float]]:
    """Audit one generated logical case with old singles and one new call."""

    request_match = _request_semantics_match(case)
    old_exchanges = [_exchange(old, spec, timeout) for spec in case.old_singles]
    new_exchange = _exchange(
        new_single if case.state_count == 1 else new_batch,
        case.new_request,
        timeout,
    )
    mismatches: list[dict[str, str]] = []
    if not request_match:
        mismatches.append(_mismatch("request_semantics_mismatch"))
    for spec, exchange in zip(case.old_singles, old_exchanges, strict=True):
        if exchange.error_code is not None:
            mismatches.append(
                _mismatch(f"old_{exchange.error_code}", state_id=spec.state_id)
            )
    if new_exchange.error_code is not None:
        mismatches.append(_mismatch(f"new_{new_exchange.error_code}"))

    state_rows: list[dict[str, object]] = []
    probability_deltas: list[float] = []
    old_input_total = 0
    new_input_total = 0
    old_output_total = 0
    new_output_total = 0
    if (
        request_match
        and all(item.response is not None for item in old_exchanges)
        and new_exchange.response is not None
    ):
        if isinstance(new_exchange.response, SystemOneBatchResponse):
            new_results = new_exchange.response.results
        else:
            new_results = [new_exchange.response]
        for spec, old_exchange, new_result in zip(
            case.old_singles, old_exchanges, new_results, strict=True
        ):
            old_response = old_exchange.response
            assert isinstance(old_response, SystemOneResponse)
            old_tokens = old_response.usage.input_tokens
            new_tokens = new_result.usage.input_tokens
            old_input_total += old_tokens
            new_input_total += new_tokens
            old_output_total += old_response.usage.output_tokens
            new_output_total += new_result.usage.output_tokens
            state_id = spec.state_id
            assert state_id is not None
            state_row: dict[str, object] = {
                "state_id": state_id,
                "old_input_tokens": old_tokens,
                "new_input_tokens": new_tokens,
                "input_token_delta": new_tokens - old_tokens,
                "old_output_tokens": old_response.usage.output_tokens,
                "new_output_tokens": new_result.usage.output_tokens,
                "output_token_delta": (
                    new_result.usage.output_tokens - old_response.usage.output_tokens
                ),
                "old_question_input_tokens": old_exchange.raw_question_input_tokens,
                "old_answer_source": (
                    "raw_legacy_preview"
                    if old_exchange.raw_legacy_answers is not None
                    else "strict_decision_v1"
                ),
                "answers": [],
            }
            if old_tokens != new_tokens:
                mismatches.append(_mismatch("input_token_mismatch", state_id=state_id))
            for question_id in spec.request.questions:
                old_answer = (
                    SimpleNamespace(**old_exchange.raw_legacy_answers[question_id])
                    if old_exchange.raw_legacy_answers is not None
                    else old_response.answers[question_id]
                )
                answer_row, answer_mismatches, deltas = _compare_answer(
                    old_answer,
                    new_result.answers[question_id],
                    tolerance=probability_tolerance,
                    state_id=state_id,
                    question_id=question_id,
                )
                state_row["answers"].append(answer_row)
                mismatches.extend(answer_mismatches)
                probability_deltas.extend(deltas)
            state_rows.append(state_row)

    record: dict[str, object] = {
        "case_id": case.id,
        "question_count": case.question_count,
        "state_count": case.state_count,
        "request_semantics_identical_except_model_id_and_batch_envelope": request_match,
        "old_http": [item.public_record() for item in old_exchanges],
        "new_http": new_exchange.public_record(),
        "old_input_tokens_total": old_input_total if state_rows else None,
        "new_input_tokens_total": new_input_total if state_rows else None,
        "input_token_total_delta": (
            new_input_total - old_input_total if state_rows else None
        ),
        "old_output_tokens_total": old_output_total if state_rows else None,
        "new_output_tokens_total": new_output_total if state_rows else None,
        "output_token_total_delta": (
            new_output_total - old_output_total if state_rows else None
        ),
        "states": state_rows,
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    return record, probability_deltas


def audit_cohorts(
    cohorts: dict[tuple[int, int], tuple[WorkloadCase, ...]],
    old: Endpoint,
    new_single: Endpoint,
    new_batch: Endpoint,
    *,
    timeout: float,
    probability_tolerance: float,
    output_handle: Any,
) -> dict[str, object]:
    """Audit every unique generated case before any measured workflow."""

    if probability_tolerance < 0 or not math.isfinite(probability_tolerance):
        raise ValueError("probability tolerance must be nonnegative and finite")
    case_count = 0
    passed_count = 0
    deltas: list[float] = []
    errors: Counter[str] = Counter()
    noul_threshold_0_5_disagreements = 0
    for cases in cohorts.values():
        for case in cases:
            record, case_deltas = audit_case(
                case,
                old,
                new_single,
                new_batch,
                timeout=timeout,
                probability_tolerance=probability_tolerance,
            )
            output_handle.write(json.dumps(record, sort_keys=True) + "\n")
            output_handle.flush()
            case_count += 1
            passed_count += bool(record["passed"])
            deltas.extend(case_deltas)
            errors.update(item["code"] for item in record["mismatches"])
            noul_threshold_0_5_disagreements += sum(
                bool(answer.get("threshold_0_5_outcome_mismatch_diagnostic"))
                for state in record["states"]
                for answer in state["answers"]
            )
    return {
        "status": "passed" if case_count == passed_count else "failed",
        "cases": case_count,
        "passed_cases": passed_count,
        "failed_cases": case_count - passed_count,
        "mismatch_counts": dict(sorted(errors.items())),
        "probability_tolerance_absolute": probability_tolerance,
        "score_tolerance_rule": "probability_tolerance * (rubric_level_count - 1)",
        "input_token_tolerance": 0,
        "choice_label_tolerance": 0,
        "noul_threshold_0_5_disagreements_diagnostic_only": (
            noul_threshold_0_5_disagreements
        ),
        "confidence_numeric_equivalence_claimed": False,
        "absolute_probability_delta": {
            "scope": "validated old/new answer pairs only",
            "count": len(deltas),
            "p50": percentile(deltas, 0.50),
            "p95": percentile(deltas, 0.95),
            "p99": percentile(deltas, 0.99),
            "max": max(deltas) if deltas else None,
        },
    }
