"""Check exact timed response bodies against the sealed sequential audit.

The protected benchmark retains every new-arm throughput wave at c8 and c32.
Its compressed records contain the original request and response
bodies, so the gate independently checks both wire hashes and the full
request-relative response contract before comparing every answer.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import random
import zlib
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
MAX_COMPRESSED_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
PROBABILITY_TOLERANCE = 0.01
RESPONSE_MATH_TOLERANCE = 2e-5
_TOPICS = ("billing", "shipping", "account", "service", "returns", "access")
_CHANNELS = ("chat", "email", "phone")


@lru_cache(maxsize=32)
def canonical_request_bodies(model_id: str, q: int, s: int) -> Mapping[str, bytes]:
    """Independently reconstruct the protected seed-17, four-variant cohort.

    This intentionally duplicates the small wire generator rather than trusting
    request hashes supplied by the producer. Benchmark tests pin it to
    ``semantic_cases.generate_cases`` so generator drift fails before release.
    """

    rng = random.Random(17 + q * 1_000_003 + s * 10_007)
    result: dict[str, bytes] = {}
    for variant in range(4):
        questions: dict[str, dict[str, Any]] = {}
        for index in range(q):
            kind = (index + variant) % 3
            if kind == 0:
                question = {
                    "type": "noul",
                    "instructions": f"Does the message require an action for check {index}?",
                    "criteria": {
                        "true": "An action is needed",
                        "false": "No action is needed",
                    },
                }
            elif kind == 1:
                question = {
                    "type": "choice",
                    "instructions": f"Choose the handling queue for check {index}.",
                    "criteria": {
                        "billing": "Payment and refund requests",
                        "service": "Product and account requests",
                        "delivery": "Shipment requests",
                    },
                }
            else:
                question = {
                    "type": "score",
                    "instructions": f"Rate the urgency for check {index}.",
                    "criteria": ["Routine", "Same day", "Immediate"],
                }
            questions[f"q{index:04d}"] = question
        states = [
            {
                "id": f"s{index:04d}",
                "state": {
                    "message": (
                        f"Synthetic {_TOPICS[rng.randrange(len(_TOPICS))]} request "
                        f"{variant}-{index}; please review the next action."
                    ),
                    "channel": _CHANNELS[rng.randrange(len(_CHANNELS))],
                    "priority_hint": index % 3,
                },
            }
            for index in range(s)
        ]
        payload = (
            {"model": model_id, "state": states[0]["state"], "questions": questions}
            if s == 1
            else {"model": model_id, "states": states, "questions": questions}
        )
        result[f"q{q:04d}_s{s:04d}_v{variant:03d}"] = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    return MappingProxyType(result)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("timed semantic evidence has a duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"timed semantic evidence has a nonfinite number: {value}")


def _json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("timed semantic evidence has invalid JSON") from error
    return _object(value, "timed semantic JSON")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _probability(value: Any, label: str) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{label} must be a finite probability")
    return float(value)


def _score(value: Any, label: str, levels: int) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= levels - 1
    ):
        raise ValueError(f"{label} is outside the rubric")
    return float(value)


def _content(value: Any, label: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if isinstance(value, str) and value.strip():
        return
    if type(value) in (dict, list):
        return
    raise ValueError(f"timed semantic {label} is invalid")


def _request_questions(
    request: dict[str, Any],
    audited: dict[str, Any],
    *,
    model_id: str,
    q: int,
    s: int,
) -> dict[str, dict[str, Any]]:
    expected_keys = (
        {"model", "state", "questions"} if s == 1 else {"model", "states", "questions"}
    )
    if set(request) != expected_keys or request.get("model") != model_id:
        raise ValueError("timed semantic request envelope differs")
    if s == 1:
        _content(request["state"], "request state")
    else:
        states = _array(request["states"], "timed request states")
        if (
            len(states) != s
            or any(
                not isinstance(state, dict) or set(state) != {"id", "state"}
                for state in states
            )
            or [state["id"] for state in states]
            != [state["state_id"] for state in audited["states"]]
        ):
            raise ValueError("timed semantic request state identity differs")
        for state in states:
            _content(state["state"], "request state")
    questions = _object(request["questions"], "timed request questions")
    if len(questions) != q or any(not key.strip() for key in questions):
        raise ValueError("timed semantic request question inventory differs")
    result: dict[str, dict[str, Any]] = {}
    for question_id, value in questions.items():
        question = _object(value, "timed request question")
        kind = question.get("type")
        if kind == "noul":
            if set(question) not in (
                {"type", "instructions"},
                {"type", "instructions", "criteria"},
            ) or (
                "criteria" in question
                and (
                    question["criteria"] is not None
                    and (
                        not isinstance(question["criteria"], dict)
                        or not set(question["criteria"]) <= {"true", "false"}
                    )
                )
            ):
                raise ValueError("timed semantic Noul criteria are invalid")
            if isinstance(question.get("criteria"), dict):
                for criterion in question["criteria"].values():
                    _content(criterion, "Noul criterion", nullable=True)
        elif kind == "choice":
            criteria = _object(question.get("criteria"), "timed Choice criteria")
            if (
                set(question) != {"type", "instructions", "criteria"}
                or not 2 <= len(criteria) <= 255
                or any(not key.strip() for key in criteria)
            ):
                raise ValueError("timed semantic Choice criteria are invalid")
            for criterion in criteria.values():
                _content(criterion, "Choice criterion", nullable=True)
        elif kind == "score":
            criteria = _array(question.get("criteria"), "timed Score criteria")
            if (
                set(question) != {"type", "instructions", "criteria"}
                or not 2 <= len(criteria) <= 10
            ):
                raise ValueError("timed semantic Score criteria are invalid")
            for criterion in criteria:
                _content(criterion, "Score criterion")
        else:
            raise ValueError("timed semantic question type is invalid")
        _content(question.get("instructions"), "question instructions")
        result[question_id] = question
    return result


def _response_distribution(
    answer: dict[str, Any], question: dict[str, Any]
) -> dict[str, float]:
    kind = question["type"]
    criteria = question["criteria"]
    keys = (
        list(criteria) if kind == "choice" else [str(i) for i in range(len(criteria))]
    )
    distribution = _object(answer.get("probabilities"), "timed distribution")
    if set(distribution) != set(keys):
        raise ValueError("timed semantic probability keys differ from request")
    values = {key: _probability(distribution[key], "timed probability") for key in keys}
    if not math.isclose(
        math.fsum(values.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=RESPONSE_MATH_TOLERANCE,
    ):
        raise ValueError("timed semantic probability distribution does not sum to one")
    probabilities = list(values.values())
    if kind == "choice":
        first, second = sorted(probabilities, reverse=True)[:2]
        expected_confidence = min(1.0, max(0.0, first - second))
        choice = answer.get("choice")
        if choice not in values or not math.isclose(
            values[choice],
            first,
            rel_tol=0.0,
            abs_tol=RESPONSE_MATH_TOLERANCE,
        ):
            raise ValueError("timed semantic Choice winner is inconsistent")
    else:
        if answer.get("legend") != {
            str(index): criterion for index, criterion in enumerate(criteria)
        }:
            raise ValueError("timed semantic Score legend differs from request")
        mean = math.fsum(index * value for index, value in enumerate(probabilities))
        if not math.isclose(
            _score(answer.get("score"), "timed score", len(probabilities)),
            mean,
            rel_tol=0.0,
            abs_tol=RESPONSE_MATH_TOLERANCE,
        ):
            raise ValueError("timed semantic Score weighted mean is inconsistent")
        variance = math.fsum(
            value * (index - mean) ** 2 for index, value in enumerate(probabilities)
        )
        uniform_variance = (len(probabilities) ** 2 - 1) / 12
        expected_confidence = min(1.0, max(0.0, 1.0 - variance / uniform_variance))
    if not math.isclose(
        _probability(answer.get("confidence"), "timed confidence"),
        expected_confidence,
        rel_tol=0.0,
        abs_tol=RESPONSE_MATH_TOLERANCE,
    ):
        raise ValueError("timed semantic confidence is inconsistent")
    return values


def _reference(
    audit_rows: list[dict[str, Any]], q: int, s: int
) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in cases:
            raise ValueError("timed semantic audit has duplicate or invalid cases")
        states = _array(row.get("states"), "audited states")
        if len(states) != s:
            raise ValueError("timed semantic audit state count differs")
        identifiers: set[str] = set()
        for state in states:
            state = _object(state, "audited state")
            state_id = state.get("state_id")
            if not isinstance(state_id, str) or not state_id or state_id in identifiers:
                raise ValueError("timed semantic audit state IDs are invalid")
            identifiers.add(state_id)
            _integer(state.get("old_input_tokens"), "audited input tokens")
            _integer(state.get("new_output_tokens"), "audited output tokens")
            answers = _array(state.get("answers"), "audited answers")
            if len(answers) != q:
                raise ValueError("timed semantic audit question count differs")
            question_ids: set[str] = set()
            for answer in answers:
                answer = _object(answer, "audited answer")
                question_id = answer.get("question_id")
                if (
                    not isinstance(question_id, str)
                    or not question_id
                    or question_id in question_ids
                ):
                    raise ValueError("timed semantic audit question IDs are invalid")
                question_ids.add(question_id)
        cases[case_id] = row
    return cases


def _compare_answers(
    actual: Any, expected: Any, questions: dict[str, dict[str, Any]]
) -> None:
    answers = _object(actual, "timed answers")
    audited = {row["question_id"]: row for row in _array(expected, "audited answers")}
    if set(answers) != set(audited) or set(answers) != set(questions):
        raise ValueError("timed semantic question inventory differs")
    for question_id, row in audited.items():
        answer = _object(answers[question_id], "timed answer")
        kind = row.get("type")
        question = questions[question_id]
        if answer.get("type") != kind or kind != question["type"]:
            raise ValueError("timed semantic answer type differs")
        if kind == "noul":
            if set(answer) != {"type", "noul"}:
                raise ValueError("timed semantic Noul response shape differs")
            old = _probability(row.get("old_probability"), "audited noul")
            new = _probability(answer.get("noul"), "timed noul")
            if abs(new - old) > PROBABILITY_TOLERANCE:
                raise ValueError("timed semantic probability tolerance exceeded")
            continue
        if kind not in ("choice", "score"):
            raise ValueError("timed semantic answer type is invalid")
        expected_fields = (
            {"type", "choice", "confidence", "probabilities"}
            if kind == "choice"
            else {"type", "score", "confidence", "legend", "probabilities"}
        )
        if set(answer) != expected_fields:
            raise ValueError("timed semantic answer response shape differs")
        validated_distribution = _response_distribution(answer, question)
        old_distribution = _object(row.get("old_probabilities"), "audited distribution")
        if set(validated_distribution) != set(old_distribution):
            raise ValueError("timed semantic probability keys differ")
        for key, old_value in old_distribution.items():
            old = _probability(old_value, "audited probability")
            new = validated_distribution[key]
            if abs(new - old) > PROBABILITY_TOLERANCE:
                raise ValueError("timed semantic probability tolerance exceeded")
        if kind == "choice":
            if answer.get("choice") != row.get("old_outcome"):
                raise ValueError("timed semantic Choice outcome differs")
        elif abs(
            _score(answer.get("score"), "timed score", len(validated_distribution))
            - _score(row.get("old_score"), "audited score", len(validated_distribution))
        ) > PROBABILITY_TOLERANCE * (len(validated_distribution) - 1):
            raise ValueError("timed semantic Score tolerance exceeded")


def _compare_body(
    body: dict[str, Any],
    request: dict[str, Any],
    audited: dict[str, Any],
    model_id: str,
    q: int,
    s: int,
) -> None:
    questions = _request_questions(request, audited, model_id=model_id, q=q, s=s)
    if body.get("model") != model_id:
        raise ValueError("timed semantic model differs")
    expected_states = _array(audited.get("states"), "audited states")
    if s == 1:
        if set(body) != {"model", "answers", "usage"}:
            raise ValueError("timed semantic single response shape differs")
        actual_states = [(expected_states[0], body)]
    else:
        if set(body) != {"model", "results", "usage"}:
            raise ValueError("timed semantic batch response shape differs")
        results = _array(body.get("results"), "timed results")
        if len(results) != s:
            raise ValueError("timed semantic result count differs")
        actual_states = list(zip(expected_states, results, strict=True))
    total_input = total_output = 0
    for expected, result in actual_states:
        result = _object(result, "timed result")
        if s > 1 and (
            set(result) != {"id", "answers", "usage"}
            or result.get("id") != expected["state_id"]
        ):
            raise ValueError("timed semantic state identity differs")
        usage = _object(result.get("usage"), "timed usage")
        if set(usage) != {"input_tokens", "output_tokens"}:
            raise ValueError("timed semantic token usage shape differs")
        input_tokens = _integer(usage["input_tokens"], "timed input tokens")
        output_tokens = _integer(usage["output_tokens"], "timed output tokens")
        if input_tokens != expected["old_input_tokens"]:
            raise ValueError("timed semantic input token usage differs")
        if output_tokens != expected["new_output_tokens"]:
            raise ValueError("timed semantic output token usage differs from audit")
        total_input += input_tokens
        total_output += output_tokens
        _compare_answers(result.get("answers"), expected.get("answers"), questions)
    usage = _object(body.get("usage"), "timed batch usage")
    if set(usage) != {"input_tokens", "output_tokens"} or (
        _integer(usage["input_tokens"], "timed total input tokens") != total_input
        or _integer(usage["output_tokens"], "timed total output tokens") != total_output
    ):
        raise ValueError("timed semantic total token usage differs")


def _read_archive(path: Path) -> bytes:
    """Read one complete gzip member with a strict decompressed size limit."""

    try:
        if path.stat().st_size > MAX_COMPRESSED_ARCHIVE_BYTES:
            raise ValueError("timed semantic compressed archive is too large")
        with path.open("rb") as handle:
            compressed = handle.read(MAX_COMPRESSED_ARCHIVE_BYTES + 1)
        if len(compressed) > MAX_COMPRESSED_ARCHIVE_BYTES:
            raise ValueError("timed semantic compressed archive is too large")
        decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        content = decoder.decompress(compressed, MAX_ARCHIVE_BYTES + 1)
    except (OSError, zlib.error) as error:
        raise ValueError("timed semantic archive is invalid") from error
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError("timed semantic archive is too large")
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("timed semantic archive has trailing or incomplete data")
    if not content or not content.endswith(b"\n"):
        raise ValueError("timed semantic archive has an incomplete record")
    return content


def _wire_body(encoded: Any, *, limit: int, label: str) -> bytes:
    if not isinstance(encoded, str) or len(encoded) > 4 * (limit // 3 + 1):
        raise ValueError(f"timed semantic {label} body is too large")
    try:
        body = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"timed semantic {label} encoding is invalid") from error
    if len(body) > limit:
        raise ValueError(f"timed semantic {label} body is too large")
    return body


def validate_timed_semantics(
    path: Path,
    samples: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    q: int,
    s: int,
    model_id: str,
    rounds: int,
) -> int:
    """Require one exact body for every new c8/c32 measured workflow."""

    references = _reference(audit_rows, q, s)
    canonical_requests = canonical_request_bodies(model_id, q, s)
    expected: dict[tuple[int, int, int], dict[str, Any]] = {}
    for sample in samples:
        if (
            sample.get("arm") == "new"
            and sample.get("phase") == "throughput"
            and sample.get("concurrency") in (8, 32)
        ):
            key = (sample["concurrency"], sample["round"], sample["sequence"])
            if key in expected:
                raise ValueError("timed semantic samples are duplicated")
            expected[key] = sample
    if not expected or {key[:2] for key in expected} != {
        (concurrency, round_number)
        for concurrency in (8, 32)
        for round_number in range(rounds)
    }:
        raise ValueError("timed semantic c8/c32 sample inventory is incomplete")
    content = _read_archive(path)
    seen: set[tuple[int, int, int]] = set()
    for line in content.splitlines():
        record = _json(line)
        if set(record) != {
            "case_id",
            "concurrency",
            "round",
            "sequence",
            "request_sha256",
            "request_base64",
            "response_sha256",
            "response_base64",
        }:
            raise ValueError("timed semantic record shape differs")
        if (
            type(record["concurrency"]) is not int
            or record["concurrency"] not in (8, 32)
            or type(record["sequence"]) is not int
            or record["sequence"] < 0
            or type(record["round"]) is not int
            or record["round"] not in range(rounds)
        ):
            raise ValueError("timed semantic record identity is invalid")
        key = (record["concurrency"], record["round"], record["sequence"])
        if key not in expected or key in seen:
            raise ValueError("timed semantic record is missing or duplicated")
        seen.add(key)
        sample = expected[key]
        for field in (
            "case_id",
            "concurrency",
            "round",
            "sequence",
            "request_sha256",
            "response_sha256",
        ):
            if record[field] != sample[field]:
                raise ValueError("timed semantic record identity differs from sample")
        request_bytes = _wire_body(
            record["request_base64"], limit=MAX_REQUEST_BYTES, label="request"
        )
        if hashlib.sha256(request_bytes).hexdigest() != sample["request_sha256"]:
            raise ValueError("timed semantic request hash differs from sample")
        if request_bytes != canonical_requests.get(sample["case_id"]):
            raise ValueError("timed semantic request differs from canonical case")
        body_bytes = _wire_body(
            record["response_base64"], limit=MAX_RESPONSE_BYTES, label="response"
        )
        if hashlib.sha256(body_bytes).hexdigest() != sample["response_sha256"]:
            raise ValueError("timed semantic response hash differs from sample")
        reference = references.get(sample["case_id"])
        if reference is None:
            raise ValueError("timed semantic case has no audited reference")
        _compare_body(
            _json(body_bytes), _json(request_bytes), reference, model_id, q, s
        )
    if seen != set(expected):
        raise ValueError("timed semantic c8/c32 archive is incomplete")
    return len(seen) * q * s
