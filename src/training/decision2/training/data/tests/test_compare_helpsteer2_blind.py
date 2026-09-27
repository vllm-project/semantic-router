"""Synthetic tests for the sealed HelpSteer2 pilot comparison."""

from __future__ import annotations

import pytest

from training.data.compare_helpsteer2_blind import compare, weighted_kappa


def fixture_rows():
    blind = []
    key = []
    judgments = []
    for group in range(12):
        for ordinal, grade in enumerate((0, 4)):
            item_id = f"case-{group}-{ordinal}"
            length = 200 if (group < 6) == (grade == 4) else 100
            blind.append(
                {
                    "id": item_id,
                    "group_id": str(group),
                    "state": {"candidate_response": "x" * length},
                }
            )
            key.append({"id": item_id, "group_id": str(group), "correctness": grade})
            judgments.append(
                {
                    "id": item_id,
                    "group_id": str(group),
                    "correctness_grade_0_to_4": grade,
                    "flags": [],
                }
            )
    sealed = {
        "gold_accessed": False,
        "aggregate": {"sample_count": 24, "independent_groups": 12},
        "rows": judgments,
    }
    return blind, key, sealed


def test_exact_review_and_length_balance():
    result = compare(*fixture_rows())
    assert result["exact"] == 24
    assert result["within_one"] == 24
    assert result["quadratic_weighted_kappa"] == 1
    assert result["pair_outcomes"] == {"concordant": 12}
    assert result["pair_by_length_direction"] == {
        "gold_higher_longer": {"concordant": 6},
        "gold_higher_shorter": {"concordant": 6},
    }


def test_tie_and_flag_counts():
    blind, key, sealed = fixture_rows()
    sealed["rows"][1]["correctness_grade_0_to_4"] = 0
    sealed["rows"][1]["flags"] = ["ambiguity"]
    result = compare(blind, key, sealed)
    assert result["exact"] == 23
    assert result["large_errors_ge_2"] == 1
    assert result["pair_outcomes"] == {"tie": 1, "concordant": 11}
    assert result["flag_counts"] == {"ambiguity": 1}
    assert result["pair_by_flag"]["any_flag"] == {"tie": 1}


def test_reject_nonblind_or_missing_rows():
    blind, key, sealed = fixture_rows()
    sealed["gold_accessed"] = True
    with pytest.raises(ValueError, match="answer blindness"):
        compare(blind, key, sealed)
    sealed["gold_accessed"] = False
    sealed["rows"].pop()
    with pytest.raises(ValueError, match="lengths differ"):
        compare(blind, key, sealed)


def test_quadratic_kappa_is_five_level():
    assert weighted_kappa([0, 4], [0, 4]) == 1
    assert weighted_kappa([0, 4], [4, 0]) == -1
