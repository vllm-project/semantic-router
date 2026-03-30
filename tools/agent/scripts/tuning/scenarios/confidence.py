"""Selective Confidence Threshold Scenario — offline per-category optimization.

Uses OfflineAnalyzer to find per-category optimal confidence thresholds for
model escalation.  This is an offline analysis scenario (no live router needed)
that classifies categories into ESCALATE/SELECTIVE/AVOID strategies.

This scenario doesn't use TuningLoop (it's offline), but follows the same
severity-weighted analytical framework.

Usage:
    from tuning import OfflineAnalyzer
    from tuning.scenarios.confidence import question_severity, normalize_logprob

    analyzer = OfflineAnalyzer(severity_fn=question_severity)
    result = analyzer.find_optimal_threshold(
        items=questions,
        confidence_fn=lambda q: q.confidence,
        quadrant_fn=lambda q: q.quadrant,
        correct_small_fn=lambda q: q.correct_small,
        correct_large_fn=lambda q: q.correct_large,
        id_fn=lambda q: q.qid,
    )
"""

from __future__ import annotations


def normalize_logprob(avg_logprob: float) -> float:
    """Mirror Go router's normalizeLogprob: (avgLogprob + 3.0) / 3.0, clamped [0, 1]."""
    return min(1.0, max(0.0, (avg_logprob + 3.0) / 3.0))


def question_severity(q) -> int:
    """Severity weight by category uplift magnitude."""
    cat = getattr(q, "category", "")
    net = abs(NET_UPLIFTS.get(cat, 0))
    if net >= 5:
        return 10
    if net >= 3:
        return 5
    if net >= 1:
        return 3
    return 1


NET_UPLIFTS = {
    "computer_science": 8,
    "other": 8,
    "psychology": 7,
    "biology": 6,
    "math": 5,
    "business": 3,
    "philosophy": 3,
    "economics": 2,
    "engineering": 2,
    "law": 2,
    "chemistry": 1,
    "physics": 1,
    "history": 0,
    "health": -2,
}
