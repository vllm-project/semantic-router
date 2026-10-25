"""Validate System One answers collected outside a timed section."""

from __future__ import annotations

from typing import Any


def validate_answers(
    answers: Any, questions: dict[str, Any], side: str, prompt_id: Any
) -> None:
    """Reject empty answers, missing question IDs, and item-level error objects."""
    where = f"{side} prompt {prompt_id}"
    if not isinstance(answers, dict) or not answers:
        raise SystemExit(f"{where}: empty or missing answers")
    if not questions:
        raise SystemExit(f"{where}: request has no questions")
    unexpected = answers.keys() - questions.keys()
    if unexpected:
        raise SystemExit(f"{where}: unexpected answer IDs {sorted(unexpected)}")
    for qid in questions:
        if qid not in answers:
            raise SystemExit(f"{where}: missing answer for {qid!r}")
        value = answers[qid]
        if not value:
            raise SystemExit(f"{where}: empty answer for {qid!r}")
        if not isinstance(value, dict):
            raise SystemExit(f"{where}: {qid!r} is not an answer object")
        if "error" in value:
            raise SystemExit(f"{where}: {qid!r} error {value['error']}")


def validate_panel(
    collected: list[Any], bodies: list[dict[str, Any]], side: str
) -> None:
    if len(collected) != len(bodies):
        raise SystemExit(f"{side}: {len(collected)} answers for {len(bodies)} requests")
    for body, answers in zip(bodies, collected, strict=True):
        validate_answers(answers, body["questions"], side, body.get("id"))
