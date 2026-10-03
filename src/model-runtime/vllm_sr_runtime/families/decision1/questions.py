"""System One requests for Decision 1.0, validated as its bundled runtime validates them.

Decision 1.0 is stricter than the shared superset rules (``systemone.py``):
text must not be blank and a question takes only its own fields. A question
may also use the superset fields ``choices`` (ordered Choice or Noul options)
and ``levels`` (ordered Score levels), or name one of the package's presets.
Each valid question becomes a ``Row``: its type, instructions and ordered
candidates, Noul always ``(false, true)`` with the runtime's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...errors import INVALID_QUESTION, QuestionError
from ...systemone import MAX_LEVELS, MAX_OPTIONS, MIN_LEVELS, MIN_OPTIONS, json_payload

MAX_QUESTIONS = 1024
KINDS = ("choice", "noul", "score")
FIELDS = frozenset({"type", "instructions", "criteria", "choices", "levels"})
NOUL_KEYS = ("false", "true")


@dataclass(frozen=True)
class Candidate:
    key: str
    description: Any


@dataclass(frozen=True)
class Row:
    """One valid question: its type, instructions and candidates in model order."""

    question_id: str
    kind: str
    instructions: Any
    candidates: tuple[Candidate, ...]

    @property
    def keys(self) -> list[str]:
        return [candidate.key for candidate in self.candidates]


@dataclass(frozen=True)
class NoulDefaults:
    """A runtime's Noul criteria when a request leaves them out (or, ``null_is_default``, sends null)."""

    false: str
    true: str
    null_is_default: bool


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


def check_request(state: Any, questions: Any) -> None:
    """Request-level rules; a violation is a malformed request (``ValueError``), not a question error."""
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(
            f"questions must be a mapping of 1 to {MAX_QUESTIONS} named questions"
        )
    for question_id in questions:
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("question IDs must be nonempty strings")
    try:
        content(state, "state")
    except QuestionError as exc:
        raise ValueError(str(exc)) from exc


def _options(value: Any, where: str, keys: tuple[str, ...] | None) -> dict[str, Any]:
    """``choices`` ([{key, description}]) as criteria; keys unique and non-blank (or from ``keys``)."""
    if not isinstance(value, list):
        raise _invalid(f"{where} must be a list of {{key, description}}")
    criteria: dict[str, Any] = {}
    for option in value:
        if (
            not isinstance(option, dict)
            or "key" not in option
            or set(option) - {"key", "description"}
        ):
            raise _invalid(f"each of {where} is {{key, description}}")
        key = option["key"]
        if not isinstance(key, str) or not key.strip() or key in criteria:
            raise _invalid(f"{where} keys must be unique nonempty strings")
        if keys is not None and key not in keys:
            raise _invalid(f"{where} may contain only {list(keys)}")
        criteria[key] = option.get("description")
    return criteria


def parse(
    question_id: str,
    question: Any,
    defaults: NoulDefaults,
    presets: dict[str, dict[str, Any]] | None = None,
) -> Row:
    """One question as a ``Row``; raises ``QuestionError(invalid_question)``."""
    where = f"questions.{question_id}"
    if isinstance(question, dict) and "preset" in question:
        if set(question) != {"preset"} or question["preset"] not in (presets or {}):
            raise _invalid(
                f"{where}.preset must name one of the model's presets, alone"
            )
        question = presets[question["preset"]]  # type: ignore[index]
    if not isinstance(question, dict):
        raise _invalid(f"{where} must be an object")
    kind = question.get("type")
    if kind not in KINDS:
        raise _invalid(f"{where}.type must be noul, choice or score")
    if set(question) - FIELDS:
        raise _invalid(
            f"{where} has unsupported fields {sorted(set(question) - FIELDS)}"
        )
    if "instructions" not in question:
        raise _invalid(f"{where}.instructions is required")
    instructions = content(question["instructions"], f"{where}.instructions")
    criteria = question.get("criteria")
    choices, levels = question.get("choices"), question.get("levels")
    if (choices is not None or levels is not None) and criteria is not None:
        raise _invalid(f"{where}: use criteria, choices or levels, not several")
    if (levels is not None and kind != "score") or (
        choices is not None and kind == "score"
    ):
        raise _invalid(
            f"{where}: levels apply to score questions, choices to the others"
        )
    if kind == "score":
        criteria = levels if levels is not None else criteria
        if (
            not isinstance(criteria, list)
            or not MIN_LEVELS <= len(criteria) <= MAX_LEVELS
        ):
            raise _invalid(
                f"{where}.criteria must list {MIN_LEVELS} to {MAX_LEVELS} levels"
            )
        candidates = [
            Candidate(str(index), content(value, f"{where}.criteria[{index}]"))
            for index, value in enumerate(criteria)
        ]
        return Row(question_id, kind, instructions, tuple(candidates))
    if choices is not None:
        criteria = _options(
            choices, f"{where}.choices", NOUL_KEYS if kind == "noul" else None
        )
    if kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - set(NOUL_KEYS)
        ):
            raise _invalid(f"{where}.criteria may contain only true and false")
        given = {
            key: content(value, f"{where}.criteria.{key}", nullable=True)
            for key, value in (criteria or {}).items()
        }
        candidates = []
        for key, default in zip(
            NOUL_KEYS, (defaults.false, defaults.true), strict=True
        ):
            description = given.get(key, default)
            if description is None and defaults.null_is_default:
                description = default
            candidates.append(Candidate(key, description))
        return Row(question_id, kind, instructions, tuple(candidates))
    if (
        not isinstance(criteria, dict)
        or not MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS
    ):
        raise _invalid(
            f"{where}.criteria must name {MIN_OPTIONS} to {MAX_OPTIONS} options"
        )
    candidates = []
    for key, value in criteria.items():
        if not isinstance(key, str) or not key.strip():
            raise _invalid(f"{where} option names must be nonempty strings")
        candidates.append(
            Candidate(key, content(value, f"{where}.criteria.{key}", nullable=True))
        )
    return Row(question_id, kind, instructions, tuple(candidates))
