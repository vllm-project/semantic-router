"""FleetOptimizer backtests for workload archetype forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..workload.forecast import (
    AggregateWindow,
    ForecastedWindow,
    WorkloadForecastScenario,
    detect_aggregate_diagnostics,
    forecast_to_mixture_scenario,
    validate_forecast_scenario,
)
from ..workload.forecasting import ReactiveLastWindowForecaster, default_forecasters
from .base import FleetOptimizer
from .forecast_models import (
    ForecastBacktestError,
    ForecastBacktestReport,
    ForecastCapacityRecommendation,
    ForecastMethodBacktest,
    ForecastWindowBacktest,
)
from .mixture import evaluate_mixture_scenario
from .mixture_models import MixtureOptimizationReport


def evaluate_forecast_backtest(
    optimizer: FleetOptimizer,
    scenario: WorkloadForecastScenario,
    gammas: list[float] | None = None,
    moving_window_count: int = 3,
    season_length_windows: int = 3,
    trend_window_count: int = 4,
    forecasters: Sequence[Any] | None = None,
    n_sim_requests: int = 0,
    verify_top_n: int = 0,
    max_total_gpus: int | None = None,
    now_s: float | None = None,
    fail_on_stale: bool = False,
    verbose: bool = False,
) -> ForecastBacktestReport:
    """Backtest proactive forecast recommendations against observed holdout windows."""

    validate_forecast_scenario(scenario)
    if not scenario.holdout_windows:
        raise ForecastBacktestError("holdout_windows are required for backtesting")
    gammas = gammas or [round(1.0 + 0.1 * k, 1) for k in range(11)]
    actual_windows = scenario.holdout_windows[: scenario.forecast_horizon_windows]
    if not actual_windows:
        raise ForecastBacktestError("forecast_horizon_windows selects no holdout data")

    source_mixture = scenario.load_source_mixture()
    base_lam = _mean([window.arrival_rate for window in scenario.aggregate_windows])
    if base_lam <= 0:
        raise ForecastBacktestError("aggregate_windows have zero arrival rate")

    actual_report, actual_required = _actual_holdout_report(
        optimizer,
        scenario,
        source_mixture,
        actual_windows,
        base_lam,
        gammas,
        n_sim_requests,
        verify_top_n,
        max_total_gpus,
        verbose,
    )

    stale_reason = scenario.stale_reason(now_s)
    if stale_reason and fail_on_stale:
        raise ForecastBacktestError(stale_reason)
    active_forecasters = _active_forecasters(
        stale_reason,
        forecasters,
        moving_window_count,
        season_length_windows,
        trend_window_count,
    )

    methods = tuple(
        _evaluate_method(
            optimizer=optimizer,
            scenario=scenario,
            source_mixture=source_mixture,
            actual_windows=actual_windows,
            actual_required=actual_required,
            base_lam=base_lam,
            forecaster=forecaster,
            gammas=gammas,
            n_sim_requests=n_sim_requests,
            verify_top_n=verify_top_n,
            max_total_gpus=max_total_gpus,
            verbose=verbose,
        )
        for forecaster in active_forecasters
    )
    _annotate_control_comparison(methods)
    diagnostics, recommended_method, fallback_control, rollback_reason = (
        _final_decision(
            scenario,
            actual_windows,
            actual_required,
            methods,
            stale_reason,
        )
    )

    return ForecastBacktestReport(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        source_mixture_id=source_mixture.id,
        base_lam=base_lam,
        methods=methods,
        actual_report=actual_report,
        actual_required=actual_required,
        recommended_method=recommended_method,
        diagnostics=tuple(diagnostics),
        actuation_records=(),
        fail_safe_triggered=stale_reason is not None,
        fallback_control=fallback_control,
        rollback_reason=rollback_reason,
    )


def _final_decision(
    scenario: WorkloadForecastScenario,
    actual_windows: Sequence[AggregateWindow],
    actual_required: ForecastCapacityRecommendation | None,
    methods: Sequence[ForecastMethodBacktest],
    stale_reason: str | None,
) -> tuple[tuple[str, ...], str | None, str | None, str | None]:
    diagnostics = list(
        detect_aggregate_diagnostics(scenario.aggregate_windows, actual_windows)
    )
    diagnostics.extend(_recommendation_diagnostics(methods, actual_required))
    recommended_method = _recommended_method(methods, diagnostics)
    fallback_control = None
    rollback_reason = None
    if stale_reason:
        fallback_control = "reactive_last_window"
        rollback_reason = stale_reason
        diagnostics.append(
            "stale forecast data; rolling back recommendations to reactive_last_window"
        )
        recommended_method = "reactive_last_window"
    return tuple(diagnostics), recommended_method, fallback_control, rollback_reason


def _actual_holdout_report(
    optimizer: FleetOptimizer,
    scenario: WorkloadForecastScenario,
    source_mixture,
    actual_windows: Sequence[AggregateWindow],
    base_lam: float,
    gammas: list[float],
    n_sim_requests: int,
    verify_top_n: int,
    max_total_gpus: int | None,
    verbose: bool,
) -> tuple[MixtureOptimizationReport, ForecastCapacityRecommendation | None]:
    actual_mixture = forecast_to_mixture_scenario(
        source_mixture=source_mixture,
        windows=actual_windows,
        base_lam=base_lam,
        scenario_id=f"{scenario.id}:actual-holdout",
        version=scenario.version,
        description="Observed holdout aggregate windows for forecast backtesting.",
        metadata={"source": "holdout_aggregates"},
    )
    report = evaluate_mixture_scenario(
        optimizer=optimizer,
        scenario=actual_mixture,
        lam=base_lam,
        gammas=gammas,
        n_sim_requests=n_sim_requests,
        verify_top_n=verify_top_n,
        max_total_gpus=max_total_gpus,
        fail_on_infeasible=False,
        verbose=verbose,
    )
    return (
        report,
        _capacity_recommendation(
            "actual_holdout",
            report,
            optimizer,
            source="actual_holdout",
        ),
    )


def _active_forecasters(
    stale_reason: str | None,
    forecasters: Sequence[Any] | None,
    moving_window_count: int,
    season_length_windows: int,
    trend_window_count: int,
) -> Sequence[Any]:
    if stale_reason:
        return (ReactiveLastWindowForecaster(),)
    return forecasters or default_forecasters(
        moving_window_count=moving_window_count,
        season_length_windows=season_length_windows,
        trend_window_count=trend_window_count,
    )


def _evaluate_method(
    optimizer: FleetOptimizer,
    scenario: WorkloadForecastScenario,
    source_mixture,
    actual_windows: Sequence[AggregateWindow],
    actual_required: ForecastCapacityRecommendation | None,
    base_lam: float,
    forecaster: Any,
    gammas: list[float],
    n_sim_requests: int,
    verify_top_n: int,
    max_total_gpus: int | None,
    verbose: bool,
) -> ForecastMethodBacktest:
    method = str(getattr(forecaster, "name", forecaster.__class__.__name__))
    control_kind = str(getattr(forecaster, "control_kind", "forecast"))
    forecast_windows = tuple(forecaster.forecast(scenario))
    forecast_mixture = forecast_to_mixture_scenario(
        source_mixture=source_mixture,
        windows=forecast_windows,
        base_lam=base_lam,
        scenario_id=f"{scenario.id}:{method}",
        version=scenario.version,
        description=(
            "Forecast-derived workload mixture scenario from content-free "
            "aggregate windows."
        ),
        metadata={
            "forecast_method": method,
            "control_kind": control_kind,
            "source_forecast_scenario": scenario.id,
        },
    )
    mixture_report = evaluate_mixture_scenario(
        optimizer=optimizer,
        scenario=forecast_mixture,
        lam=base_lam,
        gammas=gammas,
        n_sim_requests=n_sim_requests,
        verify_top_n=verify_top_n,
        max_total_gpus=max_total_gpus,
        fail_on_infeasible=False,
        verbose=verbose,
    )
    recommendation = _capacity_recommendation(
        method,
        mixture_report,
        optimizer,
        source="forecast_window",
    )
    diagnostics = []
    if (
        recommendation is not None
        and actual_required is not None
        and recommendation.total_gpus < actual_required.total_gpus
    ):
        diagnostics.append(
            f"{method} under-provisions observed holdout by "
            f"{actual_required.total_gpus - recommendation.total_gpus} GPUs"
        )
    return ForecastMethodBacktest(
        method=method,
        control_kind=control_kind,
        forecast_windows=forecast_windows,
        window_results=_window_results(forecast_windows, actual_windows),
        mixture_report=mixture_report,
        recommendation=recommendation,
        actual_required=actual_required,
        diagnostics=tuple(diagnostics),
    )


def _capacity_recommendation(
    method: str,
    report: MixtureOptimizationReport,
    optimizer: FleetOptimizer,
    source: str,
) -> ForecastCapacityRecommendation | None:
    cases = [
        case
        for case in report.cases
        if case.kind == "composition_window" and case.best is not None
    ]
    if not cases:
        cases = [
            case
            for case in report.cases
            if case.case_id == report.nominal_case_id and case.best is not None
        ]
    if not cases:
        return None
    n_s = max(case.best.n_s for case in cases if case.best is not None)
    n_l = max(case.best.n_l for case in cases if case.best is not None)
    total = n_s + n_l
    cost_hr = (
        n_s * optimizer.gpu_short.cost_per_hr + n_l * optimizer.gpu_long.cost_per_hr
    )
    worst = max(
        cases,
        key=lambda case: case.best.annualised_cost_kusd if case.best else 0.0,
    )
    best = worst.best
    return ForecastCapacityRecommendation(
        method=method,
        n_s=n_s,
        n_l=n_l,
        total_gpus=total,
        cost_per_hr=cost_hr,
        annualised_cost_kusd=cost_hr * 8760 / 1000,
        worst_case_id=worst.case_id,
        slo_met=all(case.slo_met for case in cases),
        gamma=best.gamma if best else 1.0,
        source=source,
    )


def _window_results(
    forecast_windows: Sequence[ForecastedWindow],
    actual_windows: Sequence[AggregateWindow],
) -> tuple[ForecastWindowBacktest, ...]:
    rows = []
    for forecast, actual in zip(forecast_windows, actual_windows):
        actual_rate = actual.arrival_rate
        forecast_rate = forecast.arrival_rate
        error_pct = (
            (forecast_rate - actual_rate) / actual_rate * 100.0
            if actual_rate > 0
            else 0.0
        )
        uncertainty = float(forecast.uncertainty.get("arrival_rate_stddev", 0.0) or 0.0)
        rows.append(
            ForecastWindowBacktest(
                start_s=actual.start_s,
                duration_s=actual.duration_s,
                actual_arrival_rate=actual_rate,
                forecast_arrival_rate=forecast_rate,
                arrival_rate_error_pct=error_pct,
                weight_l1_error=_weight_l1(
                    forecast.archetype_weights, actual.archetype_weights
                ),
                actual_weights=dict(actual.archetype_weights),
                forecast_weights=dict(forecast.archetype_weights),
                covered_by_arrival_uncertainty=(
                    abs(forecast_rate - actual_rate) <= uncertainty
                ),
            )
        )
    return tuple(rows)


def _annotate_control_comparison(methods: Sequence[ForecastMethodBacktest]) -> None:
    static = next((item for item in methods if item.control_kind == "static"), None)
    reactive = next((item for item in methods if item.control_kind == "reactive"), None)
    for method in methods:
        method.beats_static_control = (
            method.score < static.score
            if static is not None and method is not static
            else None
        )
        method.beats_reactive_control = (
            method.score < reactive.score
            if reactive is not None and method is not reactive
            else None
        )


def _recommendation_diagnostics(
    methods: Sequence[ForecastMethodBacktest],
    actual_required: ForecastCapacityRecommendation | None,
) -> list[str]:
    diagnostics = [item for method in methods for item in method.diagnostics]
    if actual_required is None:
        diagnostics.append("observed holdout produced no capacity recommendation")
    forecast_methods = [item for item in methods if item.control_kind == "forecast"]
    simple_controls = [
        item for item in methods if item.control_kind in {"static", "reactive"}
    ]
    if forecast_methods and simple_controls:
        best_forecast = min(forecast_methods, key=lambda item: item.score)
        best_simple = min(simple_controls, key=lambda item: item.score)
        if best_forecast.score >= best_simple.score:
            diagnostics.append(
                "forecasting did not beat simpler controls: "
                f"best_forecast={best_forecast.method} score={best_forecast.score:.3f}, "
                f"best_simple={best_simple.method} score={best_simple.score:.3f}"
            )
    return diagnostics


def _recommended_method(
    methods: Sequence[ForecastMethodBacktest],
    diagnostics: list[str],
) -> str | None:
    eligible = [method for method in methods if method.recommendation is not None]
    if not eligible:
        return None
    forecast_methods = [item for item in eligible if item.control_kind == "forecast"]
    simple_controls = [
        item for item in eligible if item.control_kind in {"static", "reactive"}
    ]
    if not forecast_methods or not simple_controls:
        return min(eligible, key=lambda item: item.score).method
    best_forecast = min(forecast_methods, key=lambda item: item.score)
    best_simple = min(simple_controls, key=lambda item: item.score)
    if best_forecast.score < best_simple.score:
        return best_forecast.method
    diagnostics.append(
        "using simpler control because forecast error is not lower than baseline"
    )
    return best_simple.method


def _weight_l1(left: dict[str, float], right: dict[str, float]) -> float:
    ids = set(left) | set(right)
    return sum(
        abs(left.get(archetype_id, 0.0) - right.get(archetype_id, 0.0))
        for archetype_id in ids
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
