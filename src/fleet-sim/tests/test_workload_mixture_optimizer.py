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
from fleet_sim.optimizer.mixture import _empirical_cdf
from fleet_sim.workload import MixtureScenario, load_mixture_scenario

_DATA = Path(__file__).parent.parent / "fleet_sim" / "data"


def _optimizer() -> FleetOptimizer:
    return FleetOptimizer(
        gpu_short=A100_80GB,
        gpu_long=A100_80GB,
        B_short=4096,
        t_slo_ms=500,
        long_max_ctx=65536,
    )


def _incompatible_constraint_scenario() -> MixtureScenario:
    return MixtureScenario.from_dict(
        {
            "schema_version": "fleet-sim.workload-mixture/v1alpha1",
            "id": "incompatible-hard-constraints",
            "version": "2026-07-18",
            "archetypes": [
                {
                    "id": "model-a-global",
                    "version": "v1",
                    "source": {"kind": "cdf", "path": "lmsys_cdf.json"},
                    "arrival_process": {"kind": "poisson"},
                    "slo_class": "interactive",
                    "model_eligibility": ["model-a"],
                    "residency": ["global"],
                    "weight": 0.5,
                },
                {
                    "id": "model-b-us",
                    "version": "v1",
                    "source": {"kind": "cdf", "path": "lmsys_cdf.json"},
                    "arrival_process": {"kind": "poisson"},
                    "slo_class": "interactive",
                    "model_eligibility": ["model-b"],
                    "residency": ["us"],
                    "weight": 0.5,
                },
            ],
        },
        base_dir=_DATA,
    )


def test_empirical_cdf_aggregates_duplicate_trace_lengths():
    left_dup = _empirical_cdf([10, 10, 20])
    right_dup = _empirical_cdf([10, 20, 20])

    assert left_dup == [(10, pytest.approx(2 / 3)), (20, 1.0)]
    assert right_dup == [(10, pytest.approx(1 / 3)), (20, 1.0)]


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
    nominal = next(case for case in report.cases if case.case_id == "nominal-mixture")
    assert nominal.model_eligibility == ("llama-3.1-70b",)
    assert nominal.residency == ("us",)
    assert any(
        case.cost_sensitivity_pct not in (None, 0.0)
        for case in report.cases
        if case.kind == "individual_archetype"
    )
    assert any("worst-case mixture" in item for item in report.diagnostics)


def test_mixture_report_fails_closed_for_unrepresentable_constraints():
    scenario = _incompatible_constraint_scenario()

    report = evaluate_mixture_scenario(
        optimizer=_optimizer(),
        scenario=scenario,
        lam=20,
        gammas=[1.0],
    )

    nominal = next(case for case in report.cases if case.case_id == "nominal-mixture")
    assert not report.ok
    assert report.robust_recommendation is None
    assert nominal.best is None
    assert nominal.active_archetypes == ("model-a-global", "model-b-us")
    assert "model_eligibility" in nominal.infeasible_reason
    assert "residency" in nominal.infeasible_reason
    assert any("CDF-only optimizer" in item for item in report.diagnostics)
    with pytest.raises(MixtureOptimizationError, match="model_eligibility"):
        evaluate_mixture_scenario(
            optimizer=_optimizer(),
            scenario=scenario,
            lam=20,
            gammas=[1.0],
            fail_on_infeasible=True,
        )


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
