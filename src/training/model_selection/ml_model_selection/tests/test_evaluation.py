"""Held-out evaluation: baselines learn from train only and regret is measured against the oracle."""

import sys
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

from evaluation import (  # noqa: E402
    ORACLE,
    baseline_selectors,
    evaluate_against_baselines,
    evaluate_policy,
)
from objective import SelectorObjective  # noqa: E402
from query_outcome_set import CandidateOutcome, QueryOutcomeSet  # noqa: E402

OBJECTIVE = SelectorObjective(quality_weight=1.0, latency_weight=0.0, cost_weight=0.0)


def _outcome(model_ref, quality, cost=0.0, success=True):
    return CandidateOutcome(
        model_ref=model_ref,
        success=success,
        quality=quality,
        latency_ms=100.0,
        cost=cost,
    )


def _snapshot(query, category, *outcomes):
    return QueryOutcomeSet(
        query=query, source="bench", category=category, outcomes=tuple(outcomes)
    )


# big is the strongest and the most expensive, small the cheapest, mid wins on utility
# once failures count as zero.
TRAIN = (
    _snapshot(
        "t1",
        "math",
        _outcome("big", 0.9, 5.0),
        _outcome("mid", 0.6, 2.0),
        _outcome("small", 0.2, 0.5),
    ),
    _snapshot(
        "t2",
        "math",
        _outcome("big", 0.8, 5.0, success=False),
        _outcome("mid", 0.7, 2.0),
        _outcome("small", 0.3, 0.5),
    ),
    _snapshot(
        "t3",
        "code",
        _outcome("big", 0.9, 5.0, success=False),
        _outcome("mid", 0.5, 2.0),
        _outcome("small", 0.1, 0.5),
    ),
)

TEST = (
    _snapshot(
        "q1",
        "math",
        _outcome("big", 0.9, 5.0),
        _outcome("mid", 0.6, 2.0),
        _outcome("small", 0.2, 0.5),
    ),
    _snapshot(
        "q2",
        "code",
        _outcome("big", 0.4, 5.0),
        _outcome("mid", 0.8, 2.0),
        _outcome("small", 0.1, 0.5),
    ),
)


def _fixed(ref):
    return lambda query, eligible: ref if ref in eligible else None


def test_oracle_has_no_regret_and_the_best_recorded_utility():
    report = evaluate_policy(ORACLE, None, TEST, OBJECTIVE)

    assert report.overall.mean_regret == 0.0
    assert report.overall.mean_utility == pytest.approx((0.9 + 0.8) / 2)
    assert report.overall.coverage == 1.0


def test_regret_is_the_oracle_score_minus_the_pick():
    report = evaluate_policy("always_small", _fixed("small"), TEST, OBJECTIVE)

    # oracle: 0.9 and 0.8; small: 0.2 and 0.1 -> regrets 0.7 and 0.7
    assert report.overall.mean_regret == pytest.approx(0.7)
    assert report.overall.mean_quality == pytest.approx(0.15)


def test_baselines_pick_by_train_statistics_not_test_outcomes():
    picks = {
        name: select("q", ("big", "mid", "small"))
        for name, select in baseline_selectors(TRAIN, OBJECTIVE).items()
    }

    assert picks["strongest"] == "big"  # mean train quality 0.866
    assert picks["cheapest"] == "small"
    assert (
        picks["global_best"] == "mid"
    )  # big's two failures score 0 under the objective


def test_a_model_unseen_in_train_ranks_last():
    select = baseline_selectors(TRAIN, OBJECTIVE)["strongest"]

    assert select("q", ("new-model", "small")) == "small"
    assert select("q", ("new-model",)) == "new-model"


def test_random_is_deterministic_and_independent_of_candidate_order():
    select = baseline_selectors(TRAIN, OBJECTIVE, seed=3)["random"]
    again = baseline_selectors(TRAIN, OBJECTIVE, seed=3)["random"]

    first = select("same query", ("a", "b", "c"))
    assert first == again("same query", ("a", "b", "c"))
    assert first == select("same query", ("c", "a", "b"))
    picks = {
        baseline_selectors(TRAIN, OBJECTIVE, seed=s)["random"](
            "same query", ("a", "b", "c")
        )
        for s in range(20)
    }
    assert len(picks) > 1


def test_abstaining_lowers_coverage_and_is_left_out_of_the_averages():
    def only_math(query, eligible):
        return "big" if query == "q1" else None

    report = evaluate_policy("partial", only_math, TEST, OBJECTIVE)

    assert report.overall.queries == 2
    assert report.overall.answered == 1
    assert report.overall.coverage == 0.5
    assert report.overall.mean_regret == 0.0
    assert report.route_share == {"big": 1.0}


def test_slices_split_the_same_queries_by_category():
    report = evaluate_policy("always_mid", _fixed("mid"), TEST, OBJECTIVE)

    assert set(report.slices) == {"code", "math"}
    assert report.slices["math"].mean_regret == pytest.approx(0.3)
    assert report.slices["code"].mean_regret == 0.0
    assert sum(m.queries for m in report.slices.values()) == report.overall.queries


def test_route_share_adds_up_to_one():
    def mixed(query, eligible):
        return "big" if query == "q1" else "mid"

    report = evaluate_policy("mixed", mixed, TEST, OBJECTIVE)

    assert report.route_share == {"big": 0.5, "mid": 0.5}


def test_a_pick_outside_the_queries_candidates_is_rejected():
    with pytest.raises(ValueError, match="not a candidate"):
        evaluate_policy("bad", _fixed_unchecked("gpt-9"), TEST, OBJECTIVE)


def _fixed_unchecked(ref):
    return lambda query, eligible: ref


def test_the_selector_sees_only_the_eligible_models_of_each_query():
    seen = []

    def spy(query, eligible):
        seen.append(eligible)
        return eligible[0]

    narrow = (_snapshot("n1", "math", _outcome("big", 0.9), _outcome("mid", 0.6)),)
    evaluate_policy("spy", spy, narrow, OBJECTIVE)

    assert seen == [("big", "mid")]


def test_empty_test_split_is_an_error():
    with pytest.raises(ValueError, match="no queries"):
        evaluate_policy("x", _fixed("big"), (), OBJECTIVE)
    with pytest.raises(ValueError, match="non-empty train"):
        baseline_selectors((), OBJECTIVE)


def test_report_covers_the_selector_every_baseline_and_the_oracle():
    reports = evaluate_against_baselines(
        _fixed("mid"),
        TRAIN,
        TEST,
        OBJECTIVE,
        selector_name="kmeans",
        current_router=_fixed("big"),
    )

    assert set(reports) == {
        "kmeans",
        "strongest",
        "cheapest",
        "global_best",
        "random",
        "current_router",
        ORACLE,
    }
    assert reports[ORACLE].overall.mean_regret == 0.0
    assert all(r.overall.mean_regret >= 0.0 for r in reports.values())
    assert reports["kmeans"].overall.mean_regret == pytest.approx(0.15)


def test_a_selector_cannot_take_a_baseline_name():
    with pytest.raises(ValueError, match="reserved"):
        evaluate_against_baselines(
            _fixed("mid"), TRAIN, TEST, OBJECTIVE, selector_name="oracle"
        )
