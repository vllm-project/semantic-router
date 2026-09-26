"""Contracts for the fixed, gold-blind generative architecture screen."""

from __future__ import annotations

from research.qwen38_generative_screen import choose_prompts, parse_content


def _row(number: int, kind: str) -> dict:
    question = {"type": kind, "instructions": "Decide."}
    if kind == "score":
        question["criteria"] = ["low", "medium", "high"]
    elif kind == "noul":
        question["criteria"] = {"false": "No", "true": "Yes"}
    else:
        question["criteria"] = {"alpha": "A", "beta": "B"}
    return {
        "id": f"item-{number}-{kind}",
        "state": "Evidence",
        "questions": {"q": question},
    }


def test_sample_depends_on_prompt_ids_and_types_only() -> None:
    rows = [
        _row(number, kind)
        for kind in ("choice", "noul", "score")
        for number in range(60)
    ]
    first = choose_prompts(rows)
    assert len(first) == 120
    assert {
        kind: sum(kind in row["id"] for row in first)
        for kind in ("choice", "noul", "score")
    } == {
        "choice": 40,
        "noul": 40,
        "score": 40,
    }
    # A private answer-bearing field would never be written to prompts, and
    # even if passed here cannot influence the sample.
    for row in rows:
        row["gold"] = {"value": "secret"}
    assert [row["id"] for row in choose_prompts(rows)] == [row["id"] for row in first]


def test_invalid_generation_is_not_repaired() -> None:
    choice = _row(0, "choice")
    assert parse_content(choice, '{"q":"alpha"}')["q"] == {
        "type": "choice",
        "choice": "alpha",
    }
    assert (
        parse_content(choice, '{"q":"ALPHA"}')["q"]["error"]
        == "invalid_generated_answer"
    )
    assert (
        parse_content(choice, '```json {"q":"alpha"} ```')["q"]["error"]
        == "invalid_generated_json"
    )
    assert (
        parse_content(choice, '{"q":"alpha","extra":0}')["q"]["error"]
        == "invalid_generated_json"
    )
    assert (
        parse_content(_row(0, "noul"), '{"q":1}')["q"]["error"]
        == "invalid_generated_answer"
    )
    assert (
        parse_content(_row(0, "score"), '{"q":true}')["q"]["error"]
        == "invalid_generated_answer"
    )
