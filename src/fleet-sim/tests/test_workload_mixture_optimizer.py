"""Tests for FleetOptimizer workload mixture reporting."""

from __future__ import annotations

from pathlib import Path

import pytest
from fleet_sim.gpu_profiles.profiles import A100_80GB
from fleet_sim.optimizer import (
    FleetOptimizer,
    MixtureOptimizationError,
    evaluate_mixture_scenario,
)
from fleet_sim.workload import load_mixture_scenario

_DATA = Path(__file__).parent.parent / "fleet_sim" / "data"


def _optimizer() -> FleetOptimizer:
    return FleetOptimizer(
        gpu_short=A100_80GB,
        gpu_long=A100_80GB,
        B_short=4096,
        t_slo_ms=500,
        long_max_ctx=65536,
    )


def test_mixture_report_compares_baselines_and_archetypes():
    scenario = load_mixture_scenario(_DATA / "workload_mixture_nominal.json")

    report = evaluate_mixture_scenario(
        optimizer=_optimizer(),
        scenario=scenario,
        lam=20,
        gammas=[1.0, 1.2],
    )

    case_ids = {case.case_id for case in report.cases}
    assert report.ok, report.diagnostics
    assert "aggregate-cdf" in case_ids
    assert "nominal-mixture" in case_ids
    assert "archetype:agent-heavy" in case_ids
    assert report.robust_recommendation is not None
    assert report.robust_recommendation.total_gpus >= 1
    assert any(
        case.cost_sensitivity_pct not in (None, 0.0)
        for case in report.cases
        if case.kind == "individual_archetype"
    )
    assert any("worst-case mixture" in item for item in report.diagnostics)


def test_mixture_report_includes_drift_and_burst_windows():
    for fixture in ("drift", "burst"):
        scenario = load_mixture_scenario(_DATA / f"workload_mixture_{fixture}.json")
        report = evaluate_mixture_scenario(
            optimizer=_optimizer(),
            scenario=scenario,
            lam=20,
            gammas=[1.0],
        )

        window_cases = [
            case for case in report.cases if case.kind == "composition_window"
        ]
        assert len(window_cases) == len(scenario.composition_schedule)
        assert report.robust_recommendation is not None


def test_mixture_report_fails_explicitly_when_gpu_budget_is_infeasible():
    scenario = load_mixture_scenario(_DATA / "workload_mixture_burst.json")

    report = evaluate_mixture_scenario(
        optimizer=_optimizer(),
        scenario=scenario,
        lam=200,
        gammas=[1.0],
        max_total_gpus=1,
    )

    assert not report.ok
    assert report.robust_recommendation is None
    assert any("max_total_gpus=1" in item for item in report.diagnostics)
    with pytest.raises(MixtureOptimizationError, match="max_total_gpus=1"):
        evaluate_mixture_scenario(
            optimizer=_optimizer(),
            scenario=scenario,
            lam=200,
            gammas=[1.0],
            max_total_gpus=1,
            fail_on_infeasible=True,
        )
