"""The bounded typed-head gate cannot silently change its rows or probabilities."""

from __future__ import annotations

import pytest
from scripts.preflight_qwen06_type_head import _compare, _phase_status, _roster


def test_preflight_roster_is_stratified_but_preserves_original_order() -> None:
    rows = [
        {"id": "c0", "task_type": "choice"},
        {"id": "s0", "task_type": "score"},
        {"id": "n0", "task_type": "noul"},
        {"id": "c1", "task_type": "choice"},
        {"id": "s1", "task_type": "score"},
    ]
    assert [
        row["id"] for row in _roster(rows, {"choice": 2, "noul": 1, "score": 1})
    ] == ["c0", "s0", "n0", "c1"]
    with pytest.raises(ValueError, match="Too few score"):
        _roster(rows, {"score": 3})


def test_preflight_reload_comparison_checks_identity_and_probability() -> None:
    left = [{"id": "c", "tokens_sha256": "a", "probabilities": [0.9, 0.1], "winner": 0}]
    right = [
        {"id": "c", "tokens_sha256": "a", "probabilities": [0.899, 0.101], "winner": 0}
    ]
    result = _compare(left, right)
    assert result["category_changes"] == 0
    assert result["max_probability_drift"] == pytest.approx(0.001)
    right[0]["tokens_sha256"] = "b"
    with pytest.raises(ValueError, match="identity"):
        _compare(left, right)


def test_zero_and_one_step_caps_are_independent() -> None:
    comparison = {"rows": 3, "category_changes": 0, "max_probability_drift": 1e-5}
    assert (
        _phase_status(
            comparison,
            expected_rows=3,
            max_drift=1e-4,
            elapsed_seconds=540,
            seconds_cap=540,
        )
        == "PASS"
    )
    assert (
        _phase_status(
            comparison,
            expected_rows=3,
            max_drift=1e-4,
            elapsed_seconds=540.01,
            seconds_cap=540,
        )
        == "HOLD"
    )
    comparison["rows"] = 32
    assert (
        _phase_status(
            comparison,
            expected_rows=32,
            max_drift=1e-3,
            elapsed_seconds=180,
            seconds_cap=180,
        )
        == "PASS"
    )
    assert (
        _phase_status(
            comparison,
            expected_rows=32,
            max_drift=1e-3,
            elapsed_seconds=180.01,
            seconds_cap=180,
        )
        == "HOLD"
    )
