"""System One request semantics shared by every decisions family.

A request carries a state and named, typed questions. This module validates
states and questions and turns each question into its ordered options:
Choice and Noul options from ``criteria`` (or the ordered ``choices``
superset field) and Score levels from ``criteria`` (or ``levels``). Families
render the options into their own prompt formats.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import INVALID_QUESTION, QuestionError
from .registry.artifacts import canonical_json

MIN_OPTIONS = 2
MAX_OPTIONS = 255
QUESTION_TYPES = ("choice", "noul", "score")
MIN_LEVELS, MAX_LEVELS = 2, 10


canonical = canonical_json


def json_payload(
    value: Any, *, nullable: bool = False, require_nonempty_text: bool = False
) -> bool:
    """Text, an object or an array of JSON values (Choice descriptions may be null)."""
    if value is None:
        return nullable
    if not isinstance(value, (str, dict, list)) or (
        require_nonempty_text and value == ""
    ):
        return False

    def json_value(part: Any) -> bool:
        if part is None or isinstance(part, (str, bool, int)):
            return True
        if isinstance(part, float):
            return math.isfinite(part)
        if isinstance(part, list):
            return all(json_value(child) for child in part)
        if isinstance(part, dict):
            return all(
                isinstance(key, str) and json_value(child)
                for key, child in part.items()
            )
        return False

    try:
        return json_value(value)
    except RecursionError:
        return False


def valid_state(state: Any) -> bool:
    if not json_payload(state):
        return False
    try:
        canonical(state)
    except (TypeError, ValueError):
        return False
    return True


def question_options(question: Any) -> tuple[str, Any, list[dict[str, Any]]]:
    """Validate one question; its type, instructions and ordered options.

    Accepts System One ``criteria`` and the superset fields ``choices``
    (ordered Choice / Noul options) and ``levels`` (ordered Score levels).
    """
    if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
        raise QuestionError(INVALID_QUESTION, "unsupported or malformed question type")
    kind = question["type"]
    instructions = question.get("instructions")
    if not json_payload(instructions, require_nonempty_text=True):
        raise QuestionError(INVALID_QUESTION, "missing question instructions")
    criteria = question.get("criteria")
    choices = question.get("choices")
    levels = question.get("levels")
    if kind == "score":
        if choices is not None:
            raise QuestionError(
                INVALID_QUESTION, "score questions take criteria or levels, not choices"
            )
        if levels is not None:
            if criteria is not None:
                raise QuestionError(
                    INVALID_QUESTION, "use criteria or levels, not both"
                )
            criteria = levels
        if (
            not isinstance(criteria, list)
            or not MIN_LEVELS <= len(criteria) <= MAX_LEVELS
        ):
            raise QuestionError(
                INVALID_QUESTION,
                "score criteria must be an ordered list of 2..10 levels",
            )
        if any(not json_payload(description) for description in criteria):
            raise QuestionError(
                INVALID_QUESTION,
                "score criteria descriptions must be text or structured data",
            )
        return (
            kind,
            instructions,
            [
                {"key": str(index), "description": description}
                for index, description in enumerate(criteria)
            ],
        )
    if levels is not None:
        raise QuestionError(INVALID_QUESTION, "levels apply only to score questions")
    if choices is not None:
        if criteria is not None:
            raise QuestionError(INVALID_QUESTION, "use criteria or choices, not both")
        criteria = _choices_to_criteria(choices)
    if kind == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
            raise QuestionError(
                INVALID_QUESTION, "noul requires only false and true criteria"
            )
        if len(criteria) < MIN_OPTIONS:
            criteria = {
                "false": criteria.get("false", "No"),
                "true": criteria.get("true", "Yes"),
            }
    if (
        not isinstance(criteria, dict)
        or not MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS
    ):
        raise QuestionError(
            INVALID_QUESTION,
            "choice/noul criteria must be an object with 2..255 options",
        )
    if any(
        not isinstance(key, str)
        or not key
        or not json_payload(description, nullable=kind == "choice")
        for key, description in criteria.items()
    ):
        raise QuestionError(
            INVALID_QUESTION,
            "choice/noul criteria need nonempty string keys and valid descriptions",
        )
    if kind == "noul" and set(criteria) != {"false", "true"}:
        raise QuestionError(INVALID_QUESTION, "noul requires false and true criteria")
    return (
        kind,
        instructions,
        [
            {"key": key, "description": description}
            for key, description in criteria.items()
        ],
    )


def _choices_to_criteria(choices: Any) -> dict[str, Any]:
    if not isinstance(choices, list):
        raise QuestionError(
            INVALID_QUESTION, "choices must be a list of {key, description}"
        )
    criteria: dict[str, Any] = {}
    for choice in choices:
        if (
            not isinstance(choice, dict)
            or set(choice) - {"key", "description"}
            or "key" not in choice
        ):
            raise QuestionError(INVALID_QUESTION, "each choice is {key, description}")
        key = choice["key"]
        if not isinstance(key, str) or not key or key in criteria:
            raise QuestionError(
                INVALID_QUESTION, "choice keys must be unique nonempty strings"
            )
        criteria[key] = choice.get("description")
    return criteria
