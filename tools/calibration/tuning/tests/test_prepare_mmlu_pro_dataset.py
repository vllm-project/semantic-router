"""Offline tests for deterministic MMLU-Pro preparation."""

from __future__ import annotations

from tuning.prepare_mmlu_pro_dataset import _prepare_rows


def _rows() -> list[dict[str, object]]:
    return [
        {
            "question_id": f"q{index}",
            "category": "math" if index < 3 else "history",
            "question": f"Question {index}",
            "options": ["first", "second"],
            "answer": "A",
        }
        for index in range(6)
    ]


def test_preparation_is_deterministic_and_creates_all_splits():
    first = _prepare_rows(_rows(), 10, None, seed=42)
    second = _prepare_rows(_rows(), 10, None, seed=42)

    assert first == second
    assert {item["split"] for item in first} == {
        "train",
        "calibration",
        "held_out",
    }
    assert all(
        "Respond with exactly 'Answer: [letter]'" in item["prompt"] for item in first
    )


def test_preparation_can_filter_categories():
    prepared = _prepare_rows(_rows(), 10, {"math"}, seed=42)

    assert len(prepared) == 3
    assert {item["category"] for item in prepared} == {"math"}
