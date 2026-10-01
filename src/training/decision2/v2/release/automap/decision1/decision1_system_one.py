# Copyright 2026 The vLLM Semantic Router Authors.
# SPDX-License-Identifier: Apache-2.0
"""System One requests and answers for Decision 1.0 models.

Answers have the schema of the System One API that the vLLM Semantic Router
Decision runtime serves for these models; Choice and Score confidence is its
``decision_type_aware_v1`` statistic. Question IDs are bookkeeping only and never
enter model text. Nothing is generated.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any

MAX_QUESTIONS = 1024
PROBABILITY_SUM_TOLERANCE = 2e-5
KINDS = ("noul", "choice", "score")


class DecisionInputError(ValueError):
    """The request or one of its questions does not follow System One."""


class DecisionInputTooLongError(DecisionInputError):
    """A complete question exceeds the model's input limit; nothing is truncated."""


@dataclass(frozen=True)
class Candidate:
    key: str
    description: Any


@dataclass(frozen=True)
class Row:
    """One typed decision: the rendered state, instructions and ordered candidates."""

    question_id: str
    type: str
    state: str
    instructions: Any
    candidates: tuple[Candidate, ...]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_text(value: Any) -> str:
    """Strings stay as written; objects and arrays use the canonical JSON form."""
    return value if isinstance(value, str) else canonical_json(value)


def _json_value(value: Any, where: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DecisionInputError(f"{where} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, where)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DecisionInputError(f"{where} object keys must be strings")
            _json_value(item, where)
        return
    raise DecisionInputError(f"{where} must contain JSON values only")


def _content(value: Any, where: str, *, nullable: bool = False) -> Any:
    if value is None and nullable:
        return None
    if isinstance(value, str):
        if not value.strip():
            raise DecisionInputError(f"{where} must not be empty or whitespace")
        return value
    if isinstance(value, (dict, list)):
        _json_value(value, where)
        return copy.deepcopy(value)
    raise DecisionInputError(f"{where} must be text, an object or an array")


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionInputError(f"{where} must be a nonempty string")
    return value


def validate_request(state: Any, questions: Any) -> Any:
    """Request-level checks; returns a detached copy of the state."""
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise DecisionInputError(
            f"questions must be a mapping of 1 to {MAX_QUESTIONS} named questions"
        )
    for question_id in questions:
        _identifier(question_id, "question ID")
    return _content(state, "state")


def validate_question(question_id: str, question: Any) -> dict[str, Any]:
    """A detached, validated copy of one System One question."""
    where = f"questions.{question_id}"
    if not isinstance(question, dict):
        raise DecisionInputError(f"{where} must be an object")
    kind = question.get("type")
    if kind not in KINDS:
        raise DecisionInputError(f"{where}.type must be noul, choice or score")
    unknown = set(question) - {"type", "instructions", "criteria"}
    if unknown:
        raise DecisionInputError(f"{where} has unsupported fields {sorted(unknown)}")
    if "instructions" not in question:
        raise DecisionInputError(f"{where}.instructions is required")
    item = {
        "type": kind,
        "instructions": _content(question["instructions"], f"{where}.instructions"),
    }
    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise DecisionInputError(
                    f"{where}.criteria may contain only true and false"
                )
            item["criteria"] = {
                key: _content(value, f"{where}.criteria.{key}", nullable=True)
                for key, value in criteria.items()
            }
    elif kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise DecisionInputError(f"{where}.criteria must name 2 to 255 options")
        item["criteria"] = {
            _identifier(name, f"{where} option name"): _content(
                value, f"{where}.criteria.{name}", nullable=True
            )
            for name, value in criteria.items()
        }
    else:
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise DecisionInputError(f"{where}.criteria must list 2 to 10 levels")
        item["criteria"] = [
            _content(value, f"{where}.criteria[{index}]")
            for index, value in enumerate(criteria)
        ]
    return item


def build_row(
    question_id: str,
    state: Any,
    question: dict[str, Any],
    *,
    noul_default_false: str,
    noul_default_true: str,
    noul_explicit_null: str,
) -> Row:
    """Ordered candidates of one validated question; Noul is always (false, true)."""
    kind = question["type"]
    if kind == "noul":
        criteria = question.get("criteria") or {}
        candidates = []
        for key, default in (
            ("false", noul_default_false),
            ("true", noul_default_true),
        ):
            if key not in criteria:
                description = default
            elif criteria[key] is None and noul_explicit_null == "use_default":
                description = default
            else:
                description = criteria[key]
            candidates.append(Candidate(key, description))
    elif kind == "choice":
        candidates = [
            Candidate(key, value) for key, value in question["criteria"].items()
        ]
    else:
        candidates = [
            Candidate(str(index), value)
            for index, value in enumerate(question["criteria"])
        ]
    return Row(
        question_id,
        kind,
        content_text(state),
        question["instructions"],
        tuple(candidates),
    )


def error_answer(question: Any, code: str) -> dict[str, Any]:
    kind = question.get("type") if isinstance(question, dict) else None
    return {"type": kind if kind in KINDS else None, "error": code}


def choice_confidence(probabilities: list[float]) -> float:
    """Top-two margin; a sole available choice has concentration one."""
    if len(probabilities) == 1:
        return 1.0
    first, second = sorted(probabilities, reverse=True)[:2]
    return min(1.0, max(0.0, first - second))


def score_confidence(probabilities: list[float]) -> float:
    """Concentration around the expected ordered Score, relative to uniform."""
    if len(probabilities) == 1:
        return 1.0
    mean = math.fsum(index * value for index, value in enumerate(probabilities))
    variance = math.fsum(
        value * (index - mean) ** 2 for index, value in enumerate(probabilities)
    )
    uniform_variance = (len(probabilities) ** 2 - 1) / 12
    return min(1.0, max(0.0, 1.0 - variance / uniform_variance))


def answer(row: Row, probabilities: list[float]) -> dict[str, Any]:
    """Typed answer for one row's ordered candidate distribution."""
    values = [float(value) for value in probabilities]
    if (
        len(values) != len(row.candidates)
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
        or not math.isclose(
            math.fsum(values), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE
        )
    ):
        return {"type": row.type, "error": "invalid_model_output"}
    if row.type == "noul":
        return {"type": "noul", "noul": values[1]}
    keys = [candidate.key for candidate in row.candidates]
    distribution = dict(zip(keys, values))
    if row.type == "choice":
        winner = max(range(len(keys)), key=values.__getitem__)
        return {
            "type": "choice",
            "choice": keys[winner],
            "confidence": choice_confidence(values),
            "probabilities": distribution,
        }
    return {
        "type": "score",
        "score": math.fsum(index * value for index, value in enumerate(values)),
        "confidence": score_confidence(values),
        "legend": {
            candidate.key: candidate.description for candidate in row.candidates
        },
        "probabilities": distribution,
    }
