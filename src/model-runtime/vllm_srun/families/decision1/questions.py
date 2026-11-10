"""System One requests for Decision 1.0, validated as its bundled runtime validates them.

Questions follow the shared System One rules (``systemone.read_question``);
a question may also name one of the package's presets. Each valid question
becomes a ``Row``: its type, instructions and ordered candidates, Noul always
``(false, true)`` with the runtime's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...errors import INVALID_QUESTION, QuestionError
from ...systemone import NOUL_KEYS, QUESTION_TYPES, content, read_question

MAX_QUESTIONS = 1024
KINDS = QUESTION_TYPES


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


def parse(
    question_id: str,
    question: Any,
    defaults: NoulDefaults,
    presets: dict[str, dict[str, Any]] | None = None,
) -> Row:
    """One question (or a preset, alone) as a ``Row``; raises ``QuestionError(invalid_question)``."""
    if isinstance(question, dict) and "preset" in question:
        if set(question) - {"preset", "require_full_input"} or question[
            "preset"
        ] not in (presets or {}):
            raise QuestionError(
                INVALID_QUESTION, "preset must name one of the model's presets, alone"
            )
        assert presets is not None
        question = {
            **presets[question["preset"]],
            **{
                key: question[key] for key in ("require_full_input",) if key in question
            },
        }
    parsed = read_question(question)
    if parsed.kind == "score":
        candidates = tuple(
            Candidate(str(index), level) for index, level in enumerate(parsed.criteria)
        )
    elif parsed.kind == "noul":
        candidates = tuple(
            Candidate(key, _noul(parsed.criteria, key, default, defaults))
            for key, default in zip(
                NOUL_KEYS, (defaults.false, defaults.true), strict=True
            )
        )
    else:
        candidates = tuple(Candidate(*option) for option in parsed.criteria.items())
    return Row(question_id, parsed.kind, parsed.instructions, candidates)


def _noul(given: dict[str, Any], key: str, default: str, defaults: NoulDefaults) -> Any:
    description = given.get(key, default)
    if description is None and defaults.null_is_default:
        return default
    return description
