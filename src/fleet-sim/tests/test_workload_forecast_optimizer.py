"""Tests for FleetOptimizer workload forecast backtesting."""

from __future__ import annotations

from pathlib import Path

from fleet_sim.gpu_profiles.profiles import A100_80GB
from fleet_sim.optimizer import FleetOptimizer, evaluate_forecast_backtest
from fleet_sim.workload import load_forecast_scenario

_DATA = Path(__file__).parent.parent / "fleet_sim" / "data"


def _optimizer() -> FleetOptimizer:
    return FleetOptimizer(
        gpu_short=A100_80GB,
        gpu_long=A100_80GB,
        B_short=4096,
        t_slo_ms=500,
        long_max_ctx=65536,
    )


def _scenario(name: str):
    return load_forecast_scenario(_DATA / f"workload_forecast_{name}.json")


def test_forecast_backtest_reports_baselines_recommendations_and_no_actuation():
    report = evaluate_forecast_backtest(
        optimizer=_optimizer(),
        scenario=_scenario("seasonal"),
        gammas=[1.0],
    )

    methods = {method.method: method for method in report.methods}
    assert set(methods) == {
        "static_mean",
        "reactive_last_window",
        "moving_window",
        "seasonal_naive",
        "linear_trend",
    }
    assert report.actual_required is not None
    assert methods["seasonal_naive"].score == 0
    assert methods["seasonal_naive"].beats_static_control is True
    assert methods["seasonal_naive"].beats_reactive_control is True
    assert report.recommended_method == "seasonal_naive"
    assert report.actuation_records == ()

    data = report.to_dict()
    assert data["actuation_records"] == []
    assert data["methods"][0]["mixture_report"]["cases"]


def test_forecast_backtest_reports_drift_and_statistical_baseline():
    report = evaluate_forecast_backtest(
        optimizer=_optimizer(),
        scenario=_scenario("drift"),
        gammas=[1.0],
    )

    assert any("taxonomy drift" in item for item in report.diagnostics)
    assert any(method.method == "linear_trend" for method in report.methods)
    assert report.actual_required is not None


def test_forecast_backtest_reports_burst_oscillation_and_simple_control_loss():
    report = evaluate_forecast_backtest(
        optimizer=_optimizer(),
        scenario=_scenario("burst"),
        gammas=[1.0],
    )

    assert any("burst detected" in item for item in report.diagnostics)
    assert any("oscillation risk" in item for item in report.diagnostics)
    assert any(
        "forecasting did not beat simpler controls" in item
        for item in report.diagnostics
    )


def test_stale_data_rolls_back_to_reactive_fallback_without_actuation():
    report = evaluate_forecast_backtest(
        optimizer=_optimizer(),
        scenario=_scenario("burst"),
        gammas=[1.0],
        now_s=360,
    )

    assert report.fail_safe_triggered is True
    assert report.fallback_control == "reactive_last_window"
    assert report.rollback_reason is not None
    assert report.recommended_method == "reactive_last_window"
    assert [method.method for method in report.methods] == ["reactive_last_window"]
    assert report.actuation_records == ()
