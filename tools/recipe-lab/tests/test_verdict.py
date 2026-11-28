"""Three-state verdict tests: priority rules, guard veto, honesty."""

from __future__ import annotations

import math

import pytest
from recipe_lab.verdict import (
    GuardResult,
    PairedStats,
    compute_paired_stats,
    evaluate_verdict,
    evidence_quality_reasons,
    resolve_constraint,
    weighted_macro_average,
    workload_guard_results,
)


def guard(name: str, value: float | None, floor: float | None = 0.95) -> GuardResult:
    return GuardResult(name=name, metric=f"guard.{name}", value=value, floor=floor)


BASE = {
    "baseline_metric": 0.80,
    "direction": "maximize",
    "min_improvement_pct": 1.0,
    "guards": [],
}


def test_pass_when_improvement_meets_bar():
    result = evaluate_verdict(**BASE, candidate_metric=0.85, incomplete_reasons=[])
    assert result.verdict == "PASS"
    assert result.improvement_pct == 6.25


def test_fail_when_below_bar():
    result = evaluate_verdict(**BASE, candidate_metric=0.805, incomplete_reasons=[])
    assert result.verdict == "FAIL"
    assert "below min" in result.reasons[0]


def test_inconclusive_outranks_fail():
    # a broken run must never be recorded as "the recipe is worse"
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.70,  # much worse
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=["trial 2 timed out", "snapshot hash changed"],
    )
    assert result.verdict == "INCONCLUSIVE"
    assert len(result.reasons) == 2


def test_guard_veto_forces_fail_despite_improvement():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.90,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[guard("pii-recall", 0.90, 0.95)],
        incomplete_reasons=[],
    )
    assert result.verdict == "FAIL"
    assert "guard floor violated" in result.reasons[0]


def test_unevaluated_guard_is_incomplete_not_fail():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.90,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[guard("pii-recall", None)],
        incomplete_reasons=[],
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "pii-recall" in result.reasons[0]


def test_missing_metrics_are_inconclusive():
    for kwargs in ({"candidate_metric": None}, {"baseline_metric": None}):
        merged = {
            **BASE,
            **kwargs,
            "candidate_metric": kwargs.get("candidate_metric", 0.9),
            "baseline_metric": kwargs.get("baseline_metric", 0.8),
        }
        result = evaluate_verdict(**merged, incomplete_reasons=[])
        assert result.verdict == "INCONCLUSIVE"


def test_minimize_direction():
    result = evaluate_verdict(
        baseline_metric=10.0,
        candidate_metric=9.0,
        direction="minimize",
        min_improvement_pct=5.0,
        guards=[],
        incomplete_reasons=[],
    )
    assert result.verdict == "PASS"  # 10% lower latency


def test_guard_passing_keeps_pass():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[guard("pii-recall", 0.97), guard("jailbreak-recall", 0.99)],
        incomplete_reasons=[],
    )
    assert result.verdict == "PASS"


# ---- M2: paired statistics decide from the confidence interval ----


def paired(
    *,
    paired_cases: int = 400,
    quality_delta: float = 0.05,
    ci: tuple[float, float] | None = (0.03, 0.07),
) -> PairedStats:
    half = paired_cases // 4
    return PairedStats(
        paired_cases=paired_cases,
        quality_delta=quality_delta,
        quality_delta_ci95=ci,
        quality_delta_bootstrap_ci95=ci,
        wins=half + 1,
        losses=max(half - 1, 0),
        ties=paired_cases - 2 * half,
    )


def test_paired_pass_when_ci_lower_bound_meets_bar():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.05, ci=(0.03, 0.07)),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "PASS"
    assert "lower bound 3.00pp" in result.reasons[0]


def test_paired_pass_boundary_exact_lower_bound():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.05, ci=(0.02, 0.07)),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "PASS"


def test_paired_fail_when_ci_upper_bound_below_bar():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.82,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.01, ci=(0.005, 0.015)),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "FAIL"
    assert "upper bound 1.50pp below min 2.00pp" in result.reasons[0]


def test_paired_inconclusive_when_interval_spans_threshold():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.83,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.03, ci=(0.01, 0.05)),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "spans the threshold 2.00pp" in result.reasons[0]


def test_paired_inconclusive_when_interval_spans_zero():
    # pp bar 0: the interval covering 0 means the improvement could be noise
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.81,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(quality_delta=0.005, ci=(-0.02, 0.03)),
        min_cases=200,
        min_improvement_pp=0.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "spans the threshold 0.00pp" in result.reasons[0]


def test_paired_inconclusive_when_insufficient_cases():
    result = evaluate_verdict(
        baseline_metric=17 / 60,
        candidate_metric=18 / 60,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(paired_cases=60, quality_delta=1 / 60),
        min_cases=200,
        min_improvement_pp=1.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "insufficient paired sample: 60 paired cases < min_cases 200" in (
        result.reasons[0]
    )


def test_paired_inconclusive_with_relative_threshold():
    # a relative bar cannot be applied to an absolute interval
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(),
        min_cases=200,
        min_improvement_pp=None,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "min_improvement_pp" in result.reasons[0]


def test_paired_minimize_direction_is_inconclusive():
    result = evaluate_verdict(
        baseline_metric=10.0,
        candidate_metric=9.0,
        direction="minimize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(),
        min_cases=200,
        min_improvement_pp=1.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "maximize objectives only" in result.reasons[0]


def test_paired_inconclusive_without_confidence_interval():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=paired(ci=None),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "no confidence interval" in result.reasons[0]


def test_pp_bar_without_paired_stats_is_inconclusive():
    # a pp bar is defined on the paired CI; point estimates cannot carry it
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.90,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[],
        incomplete_reasons=[],
        stats=None,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "requires paired CI evidence" in result.reasons[0]


def test_guard_veto_outranks_paired_sample_shortfall():
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.85,
        direction="maximize",
        min_improvement_pct=0.0,
        guards=[guard("pii-recall", 0.90, 0.95)],
        incomplete_reasons=[],
        stats=paired(paired_cases=60),
        min_cases=200,
        min_improvement_pp=2.0,
    )
    assert result.verdict == "FAIL"
    assert "guard floor violated" in result.reasons[0]


# ---- M2: compute_paired_stats (sr-bench field shape) ----


def test_compute_paired_stats_plain_matches_hoeffding():
    diffs = [1.0] * 120 + [0.0] * 80
    s = compute_paired_stats(diffs)
    expected_delta = 0.6
    radius = math.sqrt(2 * math.log(2 / 0.05) / 200)
    assert s.paired_cases == 200
    assert s.quality_delta == pytest.approx(expected_delta)
    assert s.quality_delta_ci95[0] == pytest.approx(expected_delta - radius)
    assert s.quality_delta_ci95[1] == pytest.approx(expected_delta + radius)
    assert (s.wins, s.losses, s.ties) == (120, 0, 80)


def test_compute_paired_stats_grouped_weighted_macro():
    groups = {
        "a": [1.0] * 50 + [0.0] * 50,  # n=100, mean 0.5
        "b": [1.0] * 10 + [0.0] * 10,  # n=20, mean 0.5
    }
    weights = {"a": 3.0, "b": 1.0}  # normalized: a=0.75, b=0.25
    diffs = list(groups["a"]) + list(groups["b"])
    s = compute_paired_stats(diffs, groups=groups, weights=weights)
    radius = math.sqrt(2 * math.log(2 / 0.05) * (0.75**2 / 100 + 0.25**2 / 20))
    assert s.paired_cases == 120
    assert s.quality_delta == pytest.approx(0.5)
    assert s.quality_delta_ci95[0] == pytest.approx(0.5 - radius)
    assert s.quality_delta_ci95[1] == pytest.approx(0.5 + radius)
    assert (s.wins, s.losses, s.ties) == (60, 0, 60)


def test_compute_paired_stats_bootstrap_deterministic_and_srbench_aligned():
    diffs = [1.0, 0.0] * 100
    s1 = compute_paired_stats(diffs)
    s2 = compute_paired_stats(diffs)
    assert s1 == s2  # same fixed seed -> identical intervals
    # degenerate diffs pin the bootstrap quantiles (2000 samples: [49]/[1949])
    sure = compute_paired_stats([1.0] * 200)
    assert sure.quality_delta_bootstrap_ci95 == (1.0, 1.0)


def test_compute_paired_stats_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least one"):
        compute_paired_stats([])
    with pytest.raises(ValueError, match="together"):
        compute_paired_stats([1.0], groups={"a": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        compute_paired_stats([1.0], groups={"a": [1.0]}, weights={})
    with pytest.raises(ValueError, match="cover"):
        compute_paired_stats([1.0, 1.0], groups={"a": [1.0]}, weights={"a": 1.0})


# --- M3: evidence quality floors (truncation / no-answer rates) ---


def test_evidence_quality_truncation_over_ceiling_is_reported():
    reasons = evidence_quality_reasons(
        "baseline",
        truncation_rate=32 / 60,
        missing_result_rate=0.0,
        no_answer_rate=None,
        max_truncation_pct=2.0,
        max_no_answer_pct=2.0,
    )
    assert reasons == ["baseline truncation rate 53.33% exceeds max 2.00%"]


def test_evidence_quality_missing_results_over_ceiling_is_reported():
    # Missing results and missing answers are distinct failures with the
    # same consequence: the case contributes no judged answer.
    reasons = evidence_quality_reasons(
        "candidate",
        truncation_rate=0.0,
        missing_result_rate=8 / 60,
        no_answer_rate=None,
        max_truncation_pct=2.0,
        max_no_answer_pct=2.0,
    )
    assert reasons == ["candidate missing-result rate 13.33% exceeds max 2.00%"]


def test_evidence_quality_no_answer_over_ceiling_is_reported():
    reasons = evidence_quality_reasons(
        "candidate",
        truncation_rate=0.0,
        missing_result_rate=0.0,
        no_answer_rate=5 / 60,
        max_truncation_pct=2.0,
        max_no_answer_pct=2.0,
    )
    assert reasons == ["candidate no-answer rate 8.33% exceeds max 2.00%"]


def test_evidence_quality_within_ceilings_is_clean():
    assert (
        evidence_quality_reasons(
            "baseline",
            truncation_rate=0.01,
            missing_result_rate=0.01,
            no_answer_rate=0.02,
            max_truncation_pct=2.0,
            max_no_answer_pct=2.0,
        )
        == []
    )


def test_evidence_quality_unknown_rate_is_not_penalized():
    assert (
        evidence_quality_reasons(
            "candidate",
            truncation_rate=None,
            missing_result_rate=None,
            no_answer_rate=None,
            max_truncation_pct=2.0,
            max_no_answer_pct=2.0,
        )
        == []
    )


def test_evidence_quality_ceiling_breach_is_incomplete_evidence():
    # A breach lands in incomplete_reasons, so the verdict is INCONCLUSIVE -
    # a truncated arm is unreliable evidence, not a quality loss.
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.90,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=evidence_quality_reasons(
            "baseline",
            truncation_rate=0.10,
            missing_result_rate=None,
            no_answer_rate=None,
            max_truncation_pct=2.0,
            max_no_answer_pct=2.0,
        ),
    )
    assert result.verdict == "INCONCLUSIVE"
    assert "baseline truncation rate 10.00% exceeds max 2.00%" in result.reasons


# --- M4: reference-arm constraint bounds (ref:strong / ref:weak) ---


def m4_constraint(**overrides: object) -> dict:
    base: dict = {
        "name": "beat-strong",
        "metric": "capability.score",
        "op": "gte",
        "bound": "ref:strong",
        "tolerance_pct": 0.0,
        "tolerance_pp": 0.0,
        "factor": None,
        "baseline": None,
        "candidate": 0.9,
    }
    base.update(overrides)
    return base


def test_resolve_constraint_ref_bound_passes_and_violates():
    references = {"strong": 0.85, "weak": 0.70}
    resolved = resolve_constraint(m4_constraint(), references)
    assert resolved["status"] == "pass"
    assert resolved["bound_value"] == pytest.approx(0.85)
    assert resolved["ref_value"] == pytest.approx(0.85)
    resolved = resolve_constraint(m4_constraint(candidate=0.80), references)
    assert resolved["status"] == "violated"
    assert "beat-strong violated" in resolved["reason"]


def test_resolve_constraint_factor_scales_reference_bound():
    references = {"strong": 0.90, "weak": 0.80}
    resolved = resolve_constraint(
        m4_constraint(bound="ref:weak", factor=0.5, candidate=0.5), references
    )
    assert resolved["status"] == "pass"
    assert resolved["bound_value"] == pytest.approx(0.40)
    resolved = resolve_constraint(
        m4_constraint(bound="ref:weak", factor=0.5, candidate=0.39), references
    )
    assert resolved["status"] == "violated"


def test_resolve_constraint_tolerance_pp_relaxes_by_absolute_points():
    # lte: the candidate may spend up to bound + 2pp
    resolved = resolve_constraint(
        m4_constraint(
            name="cost-cap",
            metric="total_spend_usd",
            op="lte",
            bound=0.50,
            tolerance_pp=2.0,
            candidate=0.51,
        )
    )
    assert resolved["status"] == "pass"
    assert resolved["bound_value"] == pytest.approx(0.52)
    # gte: the candidate may score down to bound - 2pp
    resolved = resolve_constraint(
        m4_constraint(
            name="quality-floor",
            metric="capability.score",
            op="gte",
            bound=0.50,
            tolerance_pp=2.0,
            candidate=0.49,
        )
    )
    assert resolved["status"] == "pass"
    assert resolved["bound_value"] == pytest.approx(0.48)


def test_resolve_constraint_missing_reference_is_not_evaluated():
    # a reference value the evidence does not carry is unevaluated
    # (INCONCLUSIVE), never counted as zero.
    for references in (None, {"strong": None, "weak": None}):
        resolved = resolve_constraint(m4_constraint(candidate=0.0), references)
        assert resolved["status"] == "not_evaluated"
        assert resolved["bound_value"] is None
        reason = resolved["not_evaluated_reason"]
        assert "reference arm 'strong'" in reason
        assert "never counted as zero" in reason


def test_evaluate_verdict_ref_constraint_pass_and_fail():
    # the constraint's measured pair value rides in the dict ("candidate");
    # candidate_metric only feeds the improvement calculation
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.90,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=[],
        constraints=[m4_constraint(candidate=0.90)],
        references={"strong": 0.85, "weak": 0.70},
    )
    assert result.verdict == "PASS"
    assert result.constraints[0]["status"] == "pass"
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.84,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=[],
        constraints=[m4_constraint(candidate=0.84)],
        references={"strong": 0.85, "weak": 0.70},
    )
    assert result.verdict == "FAIL"
    assert "beat-strong violated" in result.reasons[0]


def test_evaluate_verdict_unknown_reference_is_inconclusive():
    # the improvement cleared the bar, but an unknown reference arm keeps
    # the whole verdict INCONCLUSIVE
    result = evaluate_verdict(
        baseline_metric=0.80,
        candidate_metric=0.90,
        direction="maximize",
        min_improvement_pct=1.0,
        guards=[],
        incomplete_reasons=[],
        constraints=[m4_constraint()],
        references={"strong": None, "weak": None},
    )
    assert result.verdict == "INCONCLUSIVE"
    assert any("reference arm 'strong'" in r for r in result.reasons)


# ---- M5: weighted macro average and the auto "not worse" guard ----


def test_weighted_macro_average_uses_contract_weights():
    value = weighted_macro_average(
        {"mmlu-pro": 0.62, "office": 0.71}, {"mmlu-pro": 2.0, "office": 1.0}
    )
    assert math.isclose(value, (2.0 * 0.62 + 1.0 * 0.71) / 3.0)


def test_weighted_macro_average_excludes_missing_evidence():
    # a workload without a measurement is never counted as zero: the weights
    # renormalize over the measured workloads
    value = weighted_macro_average(
        {"mmlu-pro": 0.60, "office": None}, {"mmlu-pro": 2.0, "office": 1.0}
    )
    assert math.isclose(value, 0.60)


def test_weighted_macro_average_rejects_all_missing_and_bad_weights():
    with pytest.raises(ValueError, match="at least one measured workload"):
        weighted_macro_average(
            {"mmlu-pro": None, "office": None}, {"mmlu-pro": 2.0, "office": 1.0}
        )
    with pytest.raises(ValueError, match="positive macro-average weight"):
        weighted_macro_average({"mmlu-pro": 0.6}, {"mmlu-pro": 0.0})
    with pytest.raises(ValueError, match="positive macro-average weight"):
        weighted_macro_average({"mmlu-pro": 0.6}, {"office": 1.0})


def m5_stats(ci_low: float) -> PairedStats:
    return PairedStats(
        paired_cases=300,
        quality_delta=(ci_low + 0.01),
        quality_delta_ci95=(ci_low, ci_low + 0.02),
        quality_delta_bootstrap_ci95=None,
        wins=150,
        losses=100,
        ties=50,
    )


def test_workload_guards_pass_when_ci_low_within_tolerance():
    rows = workload_guard_results(
        {"mmlu-pro": 1.0, "office": 2.0},
        {"mmlu-pro": m5_stats(-0.008), "office": m5_stats(-0.02)},
    )
    by_name = {row.workload: row for row in rows}
    # -0.8pp >= -1pp -> the default guard holds
    assert by_name["mmlu-pro"].passed is True
    assert by_name["mmlu-pro"].evaluated is True
    # -2pp < -2pp is false: exactly at the tolerance boundary still passes
    assert by_name["office"].passed is True


def test_workload_guards_fail_beyond_tolerance_and_stay_unevaluated_without_evidence():
    rows = workload_guard_results(
        {"mmlu-pro": 1.0, "office": 1.0},
        {"mmlu-pro": m5_stats(-0.015), "office": None},
    )
    by_name = {row.workload: row for row in rows}
    # -1.5pp < -1pp: the candidate got worse by more than the guard allows
    assert by_name["mmlu-pro"].passed is False
    assert by_name["mmlu-pro"].evaluated is True
    # no paired evidence: unevaluated, never counted as a loss
    assert by_name["office"].evaluated is False
    assert by_name["office"].passed is False
