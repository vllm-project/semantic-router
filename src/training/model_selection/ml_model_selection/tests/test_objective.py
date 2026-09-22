"""Selector objective: configured weights must actually decide the winner."""

import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

from objective import SelectorObjective, label_snapshots  # noqa: E402
from query_outcome_set import CandidateOutcome, QueryOutcomeSet  # noqa: E402

FAST_MS = 100.0
SLOW_MS = 30_000.0
# A latency far past the scale should leave essentially nothing of the speed term.
NEGLIGIBLE = 0.01


def _outcome(model_ref, quality, latency_ms=FAST_MS, cost=0.0, success=True):
    return CandidateOutcome(
        model_ref=model_ref,
        success=success,
        quality=quality,
        latency_ms=latency_ms,
        cost=cost,
    )


def _snapshot(*outcomes):
    return QueryOutcomeSet(
        query="q", source="bench", category="math", outcomes=tuple(outcomes)
    )


def test_quality_wins_under_the_default_weighting():
    snapshot = _snapshot(
        _outcome("weak-fast", quality=0.50, latency_ms=FAST_MS),
        _outcome("strong-slow", quality=0.95, latency_ms=SLOW_MS),
    )
    assert SelectorObjective().best(snapshot)[0] == "strong-slow"


def test_latency_weighting_changes_the_winner():
    """The bug this replaces: three of four call sites pinned 0.9/0.1 and a
    configured weight could not move the decision."""
    snapshot = _snapshot(
        _outcome("weak-fast", quality=0.50, latency_ms=FAST_MS),
        _outcome("strong-slow", quality=0.95, latency_ms=SLOW_MS),
    )
    latency_first = SelectorObjective(quality_weight=0.1, latency_weight=0.9)
    assert latency_first.best(snapshot)[0] == "weak-fast"


def test_cost_participates_only_when_weighted():
    snapshot = _snapshot(
        _outcome("cheap", quality=0.80, cost=0.01),
        _outcome("pricey", quality=0.82, cost=5.00),
    )
    assert SelectorObjective().best(snapshot)[0] == "pricey"
    cost_aware = SelectorObjective(
        quality_weight=0.5, latency_weight=0.0, cost_weight=0.5
    )
    assert cost_aware.best(snapshot)[0] == "cheap"


def test_failure_is_a_disqualification_not_a_quality_trade():
    """A failed call must not win on a high recorded quality."""
    snapshot = _snapshot(
        _outcome("broke", quality=1.0, success=False),
        _outcome("worked", quality=0.20, success=True),
    )
    assert SelectorObjective().best(snapshot)[0] == "worked"


@pytest.mark.parametrize(
    "weights",
    [
        {"quality_weight": 1.0, "latency_weight": 1.0},
        {"quality_weight": 1.0, "latency_weight": 1.0, "cost_weight": 1.0},
        {"quality_weight": 0.0, "latency_weight": 5.0},
        {"quality_weight": 10.0, "latency_weight": 0.0},
        {"quality_weight": 0.9, "latency_weight": 0.1},
    ],
)
def test_success_outranks_failure_under_every_accepted_weighting(weights):
    """A fixed penalty only disqualified failures for small weights: with
    quality_weight=latency_weight=1.0 a failed candidate scored 0.99 against a
    successful 0.45 and won (review on #4022)."""
    snapshot = _snapshot(
        _outcome("broke", quality=1.0, latency_ms=FAST_MS, success=False),
        _outcome("worked", quality=0.20, latency_ms=SLOW_MS, success=True),
    )
    objective = SelectorObjective(**weights)
    assert objective.best(snapshot)[0] == "worked"
    assert [name for name, _ in objective.rank(snapshot)] == ["worked", "broke"]


def test_failures_still_order_among_themselves():
    """Every candidate failed, so the query still needs a deterministic winner."""
    snapshot = _snapshot(
        _outcome("bad", quality=0.10, success=False),
        _outcome("less-bad", quality=0.80, success=False),
    )
    assert SelectorObjective().best(snapshot)[0] == "less-bad"


def test_scoring_is_independent_of_the_other_candidates():
    """Label generation scaled latency by the slowest peer, so a candidate's score
    moved when an unrelated model joined the group and could not be reproduced at
    inference, where one candidate arrives alone."""
    contender = _outcome("m", quality=0.7, latency_ms=1_000.0)
    alone = SelectorObjective().score(contender)
    crowded = SelectorObjective().rank(
        _snapshot(contender, _outcome("slowpoke", quality=0.1, latency_ms=600_000.0))
    )
    assert dict(crowded)["m"] == alone


def test_ties_break_on_model_ref_for_reproducibility():
    snapshot = _snapshot(_outcome("b", quality=0.8), _outcome("a", quality=0.8))
    assert [name for name, _ in SelectorObjective().rank(snapshot)] == ["a", "b"]


def test_objective_id_tracks_the_weights():
    """An artifact has to pin the rule it was trained under."""
    base = SelectorObjective()
    assert base.objective_id == SelectorObjective().objective_id
    assert base.objective_id != SelectorObjective(quality_weight=0.5).objective_id
    assert base.objective_id != SelectorObjective(latency_scale_ms=500.0).objective_id


def test_latency_saturates_rather_than_going_negative():
    ruinous = SelectorObjective().score(_outcome("m", quality=0.0, latency_ms=10**9))
    assert 0.0 <= ruinous < NEGLIGIBLE


def test_labels_are_keyed_by_query_identity():
    snapshots = [
        _snapshot(_outcome("a", quality=0.9), _outcome("b", quality=0.1)),
    ]
    labels = label_snapshots(snapshots)
    assert labels[snapshots[0].query_id][0] == "a"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quality_weight": -0.1},
        {"latency_scale_ms": 0.0},
        {"cost_scale": -1.0},
        {"quality_weight": 0.0, "latency_weight": 0.0, "cost_weight": 0.0},
    ],
)
def test_incoherent_objectives_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SelectorObjective(**kwargs)


def test_empty_snapshot_has_no_winner():
    with pytest.raises(ValueError):
        SelectorObjective().best(_snapshot())
