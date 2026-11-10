"""System One request semantics shared by every decisions family.

A request carries a state and named, typed questions. This module validates
states and questions with one set of rules for every family: text is never
blank, a question takes only its type's fields (plus those its family reads
itself), and keys are unique non-blank strings. Each question becomes its
ordered options: Choice and Noul options from ``criteria`` (or the ordered
``choices`` superset field) and Score levels from ``criteria`` (or
``levels``). Families add their own question types and presets and render
the options into their own prompt formats.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .errors import INVALID_QUESTION, QuestionError
from .registry.artifacts import canonical_json

MIN_OPTIONS = 2
MAX_OPTIONS = 255
QUESTION_TYPES = ("choice", "noul", "score")
# Questions that locate what their instructions describe: in an image (box), at a moment of a video (box
# with ``at``), or at every sampled moment of a video (track).
BOX_TYPES = ("box", "track")
# A track's updates per second (``rate``), the default first.
TRACK_RATES = (1, 2)
MIN_LEVELS, MAX_LEVELS = 2, 10
NOUL_KEYS = ("false", "true")
FIELDS = {
    "choice": frozenset({"type", "instructions", "criteria", "choices"}),
    "noul": frozenset({"type", "instructions", "criteria", "choices"}),
    "score": frozenset({"type", "instructions", "criteria", "levels"}),
    "box": frozenset({"type", "instructions", "at"}),
    "track": frozenset({"type", "instructions", "rate"}),
}
# What a question without instructions asks (System One lets Choice, Score and a described Noul omit them).
DEFAULT_INSTRUCTIONS = "Choose the best matching option."


canonical = canonical_json

GOLDEN_STATE = (
    "Write a Python function that merges two sorted lists and explain its running time."
)


def golden_questions(other: Any) -> dict[str, Any]:
    """The readiness request's questions, one per System One type; ``other`` describes Choice's catch-all option."""
    return {
        "domain": {
            "type": "choice",
            "instructions": "Which domain does this request belong to?",
            "criteria": {"code": "Programming", "math": "Mathematics", "other": other},
        },
        "reasoning": {
            "type": "noul",
            "instructions": "Does answering this request need multi-step reasoning?",
        },
        "difficulty": {
            "type": "score",
            "instructions": "How difficult is this request?",
            "criteria": ["Trivial", "Moderate", "Hard"],
        },
    }


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


@dataclass(frozen=True)
class Question:
    """One valid System One question.

    ``criteria`` is in request order: Choice ``{key: description}``
    (descriptions may be null), Noul the given subset of ``false`` / ``true``
    (null or absent leaves the family's default) and Score the levels.
    """

    kind: str
    instructions: Any
    criteria: Any


def _invalid(message: str) -> QuestionError:
    return QuestionError(INVALID_QUESTION, message)


def content(value: Any, where: str, *, nullable: bool = False) -> Any:
    """Non-blank text, or an object or array of JSON values (null where ``nullable``)."""
    if value is None and nullable:
        return None
    if isinstance(value, str):
        if not value.strip():
            raise _invalid(f"{where} must not be empty or whitespace")
        return value
    if isinstance(value, (dict, list)) and json_payload(value):
        return value
    raise _invalid(f"{where} must be text, an object or an array")


def require_full_input(question: dict[str, Any]) -> bool:
    """Validate the opt-in completeness contract without changing ordinary questions."""
    value = question.get("require_full_input", False)
    if not isinstance(value, bool):
        raise _invalid("require_full_input must be a boolean")
    if value and question.get("overflow") == "truncate":
        raise _invalid("require_full_input cannot be combined with overflow truncate")
    return value


def read_question(
    question: Any, *, extra_fields: frozenset[str] = frozenset()
) -> Question:
    """Validate one System One question; raises ``QuestionError(invalid_question)``.

    A question takes ``type``, ``instructions`` and ``criteria``, plus the
    superset fields ``choices`` (ordered Choice / Noul options) or ``levels``
    (ordered Score levels), and the ``extra_fields`` its family reads itself
    (``over``, ``threshold``); any other field is invalid.
    """
    if not isinstance(question, dict):
        raise _invalid("a question must be an object")
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        raise QuestionError(
            INVALID_QUESTION,
            f"type must be one of {list(QUESTION_TYPES)}",
            reason="type_unsupported",
        )
    require_full_input(question)
    unknown = set(question) - FIELDS[kind] - extra_fields - {"require_full_input"}
    if unknown:
        raise _invalid(f"{kind} questions do not take {sorted(unknown)}")
    criteria = question.get("criteria")
    if question.get("instructions") is None:
        described = isinstance(criteria, dict) and any(
            value is not None for value in criteria.values()
        )
        listed = isinstance(question.get("choices"), list) and any(
            isinstance(option, dict) and option.get("description") is not None
            for option in question["choices"]
        )
        if kind == "noul" and not (described or listed):
            raise _invalid(
                "a noul question needs instructions or criteria describing an outcome"
            )
        instructions: Any = DEFAULT_INSTRUCTIONS
    else:
        instructions = content(question["instructions"], "instructions")
    superset = "levels" if kind == "score" else "choices"
    if question.get(superset) is not None:
        if criteria is not None:
            raise _invalid(f"use criteria or {superset}, not both")
        criteria = question[superset]
        if kind != "score":
            criteria = _options(criteria, NOUL_KEYS if kind == "noul" else None)
    if kind == "score":
        if (
            not isinstance(criteria, list)
            or not MIN_LEVELS <= len(criteria) <= MAX_LEVELS
        ):
            raise _invalid(f"criteria must list {MIN_LEVELS} to {MAX_LEVELS} levels")
        levels = [
            content(level, f"criteria[{index}]") for index, level in enumerate(criteria)
        ]
        return Question(kind, instructions, levels)
    if kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - set(NOUL_KEYS)
        ):
            raise _invalid("noul criteria may contain only false and true")
        given = {
            key: content(value, f"criteria.{key}", nullable=True)
            for key, value in (criteria or {}).items()
        }
        return Question(kind, instructions, given)
    return Question(kind, instructions, named_options(criteria))


@dataclass(frozen=True)
class BoxQuestion:
    """One valid box or track question: its type, instructions, ``at`` for a box at a moment of a video and
    ``rate`` for a track."""

    kind: str
    instructions: Any
    at: float | None = None
    rate: int | None = None


def read_box(question: Any) -> BoxQuestion:
    """Validate a box or track question; raises ``QuestionError(invalid_question)``."""
    if not isinstance(question, dict) or question.get("type") not in BOX_TYPES:
        raise _invalid(f"type must be one of {list(BOX_TYPES)}")
    kind = question["type"]
    unknown = set(question) - FIELDS[kind]
    if unknown:
        raise _invalid(f"{kind} questions do not take {sorted(unknown)}")
    if question.get("instructions") is None:
        raise _invalid(f"a {kind} question needs instructions: what to locate")
    instructions = content(question["instructions"], "instructions")
    at = question.get("at")
    if at is not None and (
        isinstance(at, bool)
        or not isinstance(at, (int, float))
        or not math.isfinite(at)
        or at < 0
    ):
        raise _invalid(
            "at must be a number of seconds from the start of the video, at least 0"
        )
    return BoxQuestion(
        kind,
        instructions,
        None if at is None else float(at),
        track_rate(question) if kind == "track" else None,
    )


def track_rate(question: dict[str, Any]) -> int:
    """A track question's ``rate``: 1 or 2 updates per second, 1 when absent or null; raises ``QuestionError``."""
    rate = question.get("rate")
    if rate is None:
        return TRACK_RATES[0]
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or rate not in TRACK_RATES
    ):
        raise _invalid(
            f"rate must be {TRACK_RATES[0]} or {TRACK_RATES[1]} updates per second"
        )
    return int(rate)


def choice_confidence(probabilities: Sequence[float]) -> float:
    """System One's Choice confidence: how far the top probability sits above an even split, 0 to 1."""
    count = len(probabilities)
    if count <= 1:
        return 1.0
    value = (max(probabilities) - 1 / count) / (1 - 1 / count)
    return max(0.0, min(1.0, value))


def score_confidence(probabilities: Sequence[float]) -> float:
    """System One's Score confidence: the spread around the most likely level against an even spread's, 0 to 1."""
    count = len(probabilities)
    if count <= 1:
        return 1.0
    values = list(probabilities)
    mode = values.index(max(values))
    spread = sum(p * abs(level - mode) for level, p in enumerate(values))
    even = sum(abs(level - (count - 1) / 2) for level in range(count)) / count
    return max(0.0, min(1.0, 1 - spread / even))


def named_options(criteria: Any, minimum: int = MIN_OPTIONS) -> dict[str, Any]:
    """``{name: description}`` criteria: ``minimum`` to 255 non-blank names, descriptions content or null.

    Choice options; families with their own labelled types (Set, Span) pass
    their ``minimum``.
    """
    if not isinstance(criteria, dict) or not minimum <= len(criteria) <= MAX_OPTIONS:
        raise _invalid(f"criteria must name {minimum} to {MAX_OPTIONS} options")
    options = {}
    for key, value in criteria.items():
        if not isinstance(key, str) or not key.strip():
            raise _invalid("option names must be nonempty strings")
        options[key] = content(value, f"criteria.{key}", nullable=True)
    return options


def _options(choices: Any, keys: tuple[str, ...] | None) -> dict[str, Any]:
    """``choices`` ([{key, description}]) as criteria; keys unique and non-blank (or from ``keys``)."""
    return listed_options(choices, "choices", keys)


def listed_options(
    items: Any, field: str, keys: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """An ordered ``[{key, description}]`` field (``choices``, a family's ``labels``) as criteria."""
    if not isinstance(items, list):
        raise _invalid(f"{field} must be a list of {{key, description}}")
    criteria: dict[str, Any] = {}
    for option in items:
        if (
            not isinstance(option, dict)
            or "key" not in option
            or set(option) - {"key", "description"}
        ):
            raise _invalid(f"each entry of {field} is {{key, description}}")
        key = option["key"]
        if not isinstance(key, str) or not key.strip() or key in criteria:
            raise _invalid(f"{field} keys must be unique nonempty strings")
        if keys is not None and key not in keys:
            raise _invalid(f"{field} may contain only {list(keys)}")
        criteria[key] = option.get("description")
    return criteria


def question_options(
    question: Any, noul_defaults: tuple[str, str] = ("No", "Yes")
) -> tuple[str, Any, list[dict[str, Any]]]:
    """A valid question's type, instructions and ordered ``{key, description}`` options.

    Score keys are level indices; Noul is ``false`` then ``true``, a null or
    absent description taking ``noul_defaults``.
    """
    parsed = read_question(question)
    if parsed.kind == "score":
        options = [
            {"key": str(index), "description": level}
            for index, level in enumerate(parsed.criteria)
        ]
    elif parsed.kind == "noul":
        options = [
            {
                "key": key,
                "description": (
                    default
                    if parsed.criteria.get(key) is None
                    else parsed.criteria[key]
                ),
            }
            for key, default in zip(NOUL_KEYS, noul_defaults, strict=True)
        ]
    else:
        options = [
            {"key": key, "description": description}
            for key, description in parsed.criteria.items()
        ]
    return parsed.kind, parsed.instructions, options
