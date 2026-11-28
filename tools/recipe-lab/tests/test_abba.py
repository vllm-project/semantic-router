"""AB/BA schedule and paired summary tests."""

from __future__ import annotations

from recipe_lab.abba import paired_summary, schedule


def test_schedule_alternates_starting_with_ab():
    assert schedule(3) == [
        ("baseline", 0),
        ("candidate", 0),
        ("candidate", 1),
        ("baseline", 1),
        ("baseline", 2),
        ("candidate", 2),
    ]


def test_schedule_zero_repetitions():
    assert schedule(0) == []


def test_paired_summary_medians_and_missing():
    results = [
        {"label": "baseline", "metrics": {"accuracy": 0.80}},
        {"label": "candidate", "metrics": {"accuracy": 0.85}},
        {"label": "baseline", "metrics": {"accuracy": 0.82}},
        {"label": "candidate", "metrics": {}},  # missing -> counted, not zeroed
    ]
    summary = paired_summary(results, "accuracy")
    assert summary["baseline"]["median"] == 0.81
    assert summary["candidate"]["median"] == 0.85
    assert summary["missing"] == {"baseline": 0, "candidate": 1}
    assert "not established" in summary["measurement_method"]
