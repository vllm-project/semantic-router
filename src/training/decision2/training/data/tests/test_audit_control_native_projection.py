"""Tests of source-group quarantine and typed native NLI projections."""

from __future__ import annotations

from training.data.audit_control_native_projection import _row, quarantined_groups
from training.data.audit_nli_evidence_score import Pair
from training.model.data import validate_row


def test_projection_preserves_three_way_oracle_and_binary_meaning() -> None:
    for label, choice_key in (
        ("entailment", "supports"),
        ("neutral", "undetermined"),
        ("contradiction", "contradicts"),
    ):
        pair = Pair(
            "The agreement was signed.", "The agreement is signed.", label, "g", "x", 1
        )
        rows = {
            task: _row(pair, task=task)
            for task in ("choice", "supports", "contradicts")
        }
        for row in rows.values():
            validate_row(row, "train")
            assert row["state"] == pair.premise
            assert row["instructions"]["claim"] == pair.hypothesis
            assert row["task_type"] in {"choice", "noul"}
        assert rows["choice"]["options"][rows["choice"]["label"]]["key"] == choice_key
        for task in ("supports", "contradicts"):
            expected = "true" if choice_key == task else "false"
            assert rows[task]["options"][rows[task]["label"]]["key"] == expected


def test_near_hit_quarantines_entire_premise_group() -> None:
    long_premise = "aaaaabbbbbcccccddddd zzzzzwwwwwvvvvvuuuuu"
    pairs = [
        Pair(long_premise, "Claim one", "entailment", "group-one", "x", 0),
        Pair(long_premise, "Claim two", "neutral", "group-one", "x", 1),
        Pair(
            "A separate source passage with enough characters.",
            "Other claim",
            "neutral",
            "group-two",
            "x",
            2,
        ),
    ]
    blocked, overlap = quarantined_groups(
        pairs, [("typed_final_goldfree", "aaaaabbbbbcccccddddd")]
    )
    assert blocked == {"group-one"}
    assert overlap["matched_source_group_count"] == 1
