"""Tests for workload archetype forecast aggregate scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fleet_sim.workload import (
    ForecastValidationError,
    LinearTrendForecaster,
    SeasonalNaiveForecaster,
    WorkloadForecastScenario,
    detect_aggregate_diagnostics,
    forecast_to_mixture_scenario,
    load_forecast_scenario,
    validate_forecast_scenario,
    validate_mixture_scenario,
)

_DATA = Path(__file__).parent.parent / "fleet_sim" / "data"


def _scenario(name: str):
    return load_forecast_scenario(_DATA / f"workload_forecast_{name}.json")


def test_forecast_fixture_loads_and_validates_schema_and_taxonomy():
    scenario = _scenario("seasonal")

    assert scenario.schema_version == "fleet-sim.workload-archetype-forecast/v1alpha1"
    assert scenario.taxonomy_version == "fleet-sim.workload-archetype-taxonomy/v1alpha1"
    assert scenario.privacy.content_free is True
    assert scenario.privacy.min_requests_per_window == 100
    assert scenario.load_source_mixture().archetype_ids == (
        "interactive-chat",
        "multiturn-chat",
        "agent-heavy",
    )


def test_high_cardinality_or_content_fields_fail_validation():
    data = json.loads((_DATA / "workload_forecast_seasonal.json").read_text())
    data["aggregate_windows"][0]["caller_id"] = "customer-123"
    scenario = WorkloadForecastScenario.from_dict(data, base_dir=_DATA)

    with pytest.raises(ForecastValidationError, match="caller_id"):
        validate_forecast_scenario(scenario, raw_data=data)


def test_low_count_window_fails_privacy_threshold():
    data = json.loads((_DATA / "workload_forecast_seasonal.json").read_text())
    data["aggregate_windows"][0]["request_count"] = 5
    scenario = WorkloadForecastScenario.from_dict(data, base_dir=_DATA)

    with pytest.raises(ForecastValidationError, match="min_requests_per_window"):
        validate_forecast_scenario(scenario, raw_data=data)


def test_seasonal_forecast_converts_to_reproducible_mixture_scenario():
    scenario = _scenario("seasonal")
    source_mixture = scenario.load_source_mixture()
    windows = SeasonalNaiveForecaster(season_length_windows=3).forecast(scenario)
    base_lam = sum(window.arrival_rate for window in scenario.aggregate_windows) / len(
        scenario.aggregate_windows
    )

    mixture = forecast_to_mixture_scenario(
        source_mixture=source_mixture,
        windows=windows,
        base_lam=base_lam,
        scenario_id="test-seasonal-forecast",
        version="test",
    )

    validate_mixture_scenario(mixture)
    assert len(mixture.composition_schedule) == 2
    assert windows[0].archetype_weights == scenario.holdout_windows[0].archetype_weights
    assert windows[1].request_count == scenario.holdout_windows[1].request_count


def test_linear_trend_projects_drift_and_normalizes_weights():
    scenario = _scenario("drift")
    windows = LinearTrendForecaster(window_count=4).forecast(scenario)

    assert (
        windows[0].archetype_weights["agent-heavy"]
        > scenario.aggregate_windows[-1].archetype_weights["agent-heavy"]
    )
    for window in windows:
        assert sum(window.archetype_weights.values()) == pytest.approx(1.0)


def test_burst_drift_and_oscillation_diagnostics_are_detected():
    burst = _scenario("burst")
    burst_diagnostics = detect_aggregate_diagnostics(
        burst.aggregate_windows, burst.holdout_windows
    )

    assert any("burst detected" in item for item in burst_diagnostics)
    assert any("oscillation risk" in item for item in burst_diagnostics)

    drift = _scenario("drift")
    drift_diagnostics = detect_aggregate_diagnostics(
        drift.aggregate_windows, drift.holdout_windows
    )
    assert any("taxonomy drift" in item for item in drift_diagnostics)


def test_stale_reason_supports_reactive_rollback_policy():
    scenario = _scenario("burst")

    assert scenario.stale_reason(now_s=240) is None
    assert "stale" in scenario.stale_reason(now_s=360)
