"""Post-timing semantic checks and bounded wire evidence for arrival waves.

Both arms are checked against the sealed untimed audit. This module does not
run in an HTTP worker or contribute to a measured completion timestamp.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from typing import Any, TextIO

from tools.ci import decision_timed_semantics as timed_semantics

from .historical_preview_projection import project_legacy_preview
from .semantic_cases import WorkloadCase
from .semantic_transport import HttpSample

MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _new_audit_reference(audited: dict[str, Any]) -> dict[str, Any]:
    """Project new audit values into the comparator's reference fields."""

    states = []
    for state in audited["states"]:
        answers = []
        for answer in state["answers"]:
            expected = dict(answer)
            if answer["type"] == "noul":
                expected["old_probability"] = answer["new_probability"]
            else:
                expected["old_probabilities"] = answer["new_probabilities"]
                if answer["type"] == "choice":
                    expected["old_outcome"] = answer["new_outcome"]
                elif answer["type"] == "score":
                    expected["old_score"] = answer["new_score"]
                else:
                    raise ValueError("timed audit answer type is invalid")
            answers.append(expected)
        states.append({**state, "answers": answers})
    return {"states": states}


def _compare_sample(
    sample: HttpSample,
    case: WorkloadCase,
    audited: dict[str, Any],
    *,
    old_model_id: str,
    model_id: str,
    old_response_mode: str,
) -> None:
    """Compare one captured timed call with its exact case and audit state."""

    if sample.arm == "old":
        matches = [
            spec for spec in case.old_singles if spec.state_id == sample.state_id
        ]
        if len(matches) != 1:
            raise ValueError("timed old state inventory differs")
        spec = matches[0]
        states = [
            state for state in audited["states"] if state["state_id"] == spec.state_id
        ]
        if len(states) != 1:
            raise ValueError("timed old audit state inventory differs")
        expected_state = {
            **states[0],
            "new_output_tokens": states[0]["old_output_tokens"],
        }
        reference = {"states": [expected_state]}
        expected_model = old_model_id
        state_count = 1
    elif sample.arm == "new":
        spec = case.new_request
        reference = audited
        expected_model = model_id
        state_count = case.state_count
    else:
        raise ValueError("timed arm is invalid")

    request_body = sample.request_body
    response_body = sample.response_body
    if request_body is None or response_body is None:
        raise ValueError("timed wire body is missing")
    if (
        request_body != spec.body
        or sample.request_sha256 != spec.sha256
        or hashlib.sha256(request_body).hexdigest() != sample.request_sha256
        or hashlib.sha256(response_body).hexdigest() != sample.response_sha256
    ):
        raise ValueError("timed wire body or hash differs")
    parsed = timed_semantics._json(response_body)
    if sample.arm == "old" and old_response_mode == "legacy_preview":
        expected_tokens = states[0]["old_question_input_tokens"]
        if not isinstance(expected_tokens, dict):
            raise ValueError("timed legacy audit tokens are missing")
        answers = parsed.get("answers")
        if (
            not isinstance(answers, dict)
            or {
                key: value.get("input_tokens") if isinstance(value, dict) else None
                for key, value in answers.items()
            }
            != expected_tokens
        ):
            raise ValueError("timed legacy question tokens differ")
        parsed = project_legacy_preview(parsed, spec.request)
    request = timed_semantics._json(request_body)
    timed_semantics._compare_body(
        parsed,
        request,
        reference,
        expected_model,
        case.question_count,
        state_count,
    )
    if sample.arm == "new":
        timed_semantics._compare_body(
            parsed,
            request,
            _new_audit_reference(audited),
            expected_model,
            case.question_count,
            state_count,
        )


class TimedSemanticEvidence:
    """Validate throughput samples and archive exact bodies within a hard cap."""

    def __init__(self, handle: TextIO) -> None:
        self.handle = handle
        self.uncompressed_bytes = 0
        self.archived = 0

    def validate_wave(
        self,
        samples: list[HttpSample],
        trace: tuple[Any, ...],
        audited_cases: dict[str, dict[str, Any]],
        *,
        arm: str,
        model_id: str,
        old_model_id: str,
        old_response_mode: str,
    ) -> tuple[dict[tuple[int, str | None], str], dict[str, Any]]:
        statuses: dict[tuple[int, str | None], str] = {}
        counts: Counter[str] = Counter()
        for sample in samples:
            key = sample.sequence, sample.state_id
            status = "passed"
            if sample.error_code is not None:
                status = "http_failure"
            elif sample.phase != "throughput" or sample.arm != arm:
                status = "timed_evidence_identity_mismatch"
            elif sample.sequence >= len(trace) or sample.sequence < 0:
                status = "timed_evidence_identity_mismatch"
            elif sample.request_body is None or sample.response_body is None:
                status = "timed_evidence_missing_body"
            else:
                case = trace[sample.sequence].case
                audited = audited_cases.get(case.id)
                if sample.case_id != case.id or audited is None:
                    status = "timed_evidence_identity_mismatch"
                else:
                    try:
                        _compare_sample(
                            sample,
                            case,
                            audited,
                            old_model_id=old_model_id,
                            model_id=model_id,
                            old_response_mode=old_response_mode,
                        )
                    except (KeyError, TypeError, ValueError):
                        status = "timed_semantic_mismatch"
            if sample.request_body is not None and sample.response_body is not None:
                if (
                    len(sample.request_body) > MAX_REQUEST_BYTES
                    or len(sample.response_body) > MAX_RESPONSE_BYTES
                ):
                    status = "timed_evidence_body_limit"
                else:
                    record = {
                        "arm": sample.arm,
                        "phase": sample.phase,
                        "round": sample.round,
                        "sequence": sample.sequence,
                        "case_id": sample.case_id,
                        "state_id": sample.state_id,
                        "concurrency": sample.concurrency,
                        "request_sha256": sample.request_sha256,
                        "response_sha256": sample.response_sha256,
                        "request_base64": base64.b64encode(sample.request_body).decode(
                            "ascii"
                        ),
                        "response_base64": base64.b64encode(
                            sample.response_body
                        ).decode("ascii"),
                        "validation_status": status,
                    }
                    line = (
                        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    size = len(line.encode("utf-8"))
                    if self.uncompressed_bytes + size <= MAX_ARCHIVE_BYTES:
                        self.handle.write(line)
                        self.uncompressed_bytes += size
                        self.archived += 1
                    else:
                        status = "timed_evidence_archive_limit"
            elif status == "passed":
                status = "timed_evidence_missing_body"
            statuses[key] = status
            counts[status] += 1
        self.handle.flush()
        return statuses, {
            "attempted_http": len(samples),
            "validated_http": counts["passed"],
            "status_counts": dict(sorted(counts.items())),
        }
