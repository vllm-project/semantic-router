"""Naive and statistical baselines for aggregate workload forecasts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .forecast import AggregateWindow, ForecastedWindow, WorkloadForecastScenario


class StaticMeanForecaster:
    """Predict the horizon from the mean historical aggregate window."""

    name = "static_mean"
    control_kind = "static"

    def forecast(
        self, scenario: WorkloadForecastScenario
    ) -> tuple[ForecastedWindow, ...]:
        return _constant_forecast(
            scenario,
            scenario.aggregate_windows,
            self.name,
        )


class ReactiveLastWindowForecaster:
    """Predict the horizon by replaying the most recent aggregate window."""

    name = "reactive_last_window"
    control_kind = "reactive"

    def forecast(
        self, scenario: WorkloadForecastScenario
    ) -> tuple[ForecastedWindow, ...]:
        return _constant_forecast(
            scenario,
            scenario.aggregate_windows[-1:],
            self.name,
        )


@dataclass(frozen=True)
class MovingWindowForecaster:
    """Predict the horizon from the mean of the last N aggregate windows."""

    window_count: int = 3
    name: str = "moving_window"
    control_kind: str = "forecast"

    def forecast(
        self, scenario: WorkloadForecastScenario
    ) -> tuple[ForecastedWindow, ...]:
        if self.window_count <= 0:
            raise ValueError("window_count must be positive")
        return _constant_forecast(
            scenario,
            scenario.aggregate_windows[-self.window_count :],
            self.name,
        )


@dataclass(frozen=True)
class SeasonalNaiveForecaster:
    """Replay the same slot from the previous season."""

    season_length_windows: int = 3
    fallback_window_count: int = 3
    name: str = "seasonal_naive"
    control_kind: str = "forecast"

    def forecast(
        self, scenario: WorkloadForecastScenario
    ) -> tuple[ForecastedWindow, ...]:
        if self.season_length_windows <= 0:
            raise ValueError("season_length_windows must be positive")
        if len(scenario.aggregate_windows) < self.season_length_windows:
            return MovingWindowForecaster(self.fallback_window_count).forecast(scenario)
        templates = _horizon_templates(scenario)
        history = scenario.aggregate_windows
        windows = []
        for idx, template in enumerate(templates):
            source = history[
                -self.season_length_windows + idx % self.season_length_windows
            ]
            windows.append(_project_from_reference(template, (source,), self.name))
        return tuple(windows)


@dataclass(frozen=True)
class LinearTrendForecaster:
    """A small statistical baseline that extrapolates aggregate trends."""

    window_count: int = 4
    name: str = "linear_trend"
    control_kind: str = "forecast"

    def forecast(
        self, scenario: WorkloadForecastScenario
    ) -> tuple[ForecastedWindow, ...]:
        if self.window_count <= 1:
            raise ValueError("window_count must be greater than 1")
        history = scenario.aggregate_windows[-self.window_count :]
        templates = _horizon_templates(scenario)
        archetype_ids = tuple(history[-1].archetype_weights)
        rates = [window.arrival_rate for window in history]
        token_rates = [window.tokens_per_request for window in history]
        weights_by_id = {
            archetype_id: [window.archetype_weights[archetype_id] for window in history]
            for archetype_id in archetype_ids
        }
        windows = []
        for horizon_idx, template in enumerate(templates):
            x = len(history) + horizon_idx
            rate = max(1e-9, _linear_predict(rates, x))
            tokens_per_request = max(1.0, _linear_predict(token_rates, x))
            weights = _normalise_weights(
                {
                    archetype_id: max(0.0, _linear_predict(values, x))
                    for archetype_id, values in weights_by_id.items()
                },
                fallback=history[-1].archetype_weights,
            )
            request_count = max(1, round(rate * template.duration_s))
            windows.append(
                ForecastedWindow(
                    start_s=template.start_s,
                    duration_s=template.duration_s,
                    request_count=request_count,
                    total_tokens=max(1, round(tokens_per_request * request_count)),
                    archetype_weights=weights,
                    source_method=self.name,
                    model_class=template.model_class,
                    slo_class=template.slo_class,
                    region=template.region,
                    p50_total_tokens=_linear_optional_int(
                        [window.p50_total_tokens for window in history], x
                    ),
                    p95_total_tokens=_linear_optional_int(
                        [window.p95_total_tokens for window in history], x
                    ),
                    p99_ttft_ms=_linear_optional(
                        [window.p99_ttft_ms for window in history], x
                    ),
                    uncertainty={
                        "arrival_rate_stddev": _stddev(rates),
                        "weight_stddev": _weight_stddev(history),
                    },
                )
            )
        return tuple(windows)


def default_forecasters(
    moving_window_count: int = 3,
    season_length_windows: int = 3,
    trend_window_count: int = 4,
) -> tuple[
    StaticMeanForecaster
    | ReactiveLastWindowForecaster
    | MovingWindowForecaster
    | SeasonalNaiveForecaster
    | LinearTrendForecaster,
    ...,
]:
    """Return the default controls and forecast baselines for backtests."""

    return (
        StaticMeanForecaster(),
        ReactiveLastWindowForecaster(),
        MovingWindowForecaster(moving_window_count),
        SeasonalNaiveForecaster(season_length_windows),
        LinearTrendForecaster(trend_window_count),
    )


def _horizon_templates(
    scenario: WorkloadForecastScenario,
) -> tuple[AggregateWindow, ...]:
    if scenario.holdout_windows:
        return scenario.holdout_windows[: scenario.forecast_horizon_windows]
    last = scenario.aggregate_windows[-1]
    templates = []
    for idx in range(scenario.forecast_horizon_windows):
        templates.append(
            AggregateWindow(
                start_s=last.end_s + idx * last.duration_s,
                duration_s=last.duration_s,
                request_count=last.request_count,
                total_tokens=last.total_tokens,
                archetype_weights=dict(last.archetype_weights),
                model_class=last.model_class,
                slo_class=last.slo_class,
                region=last.region,
                p50_total_tokens=last.p50_total_tokens,
                p95_total_tokens=last.p95_total_tokens,
                p99_ttft_ms=last.p99_ttft_ms,
                uncertainty=dict(last.uncertainty),
            )
        )
    return tuple(templates)


def _constant_forecast(
    scenario: WorkloadForecastScenario,
    references: Sequence[AggregateWindow],
    method: str,
) -> tuple[ForecastedWindow, ...]:
    templates = _horizon_templates(scenario)
    return tuple(
        _project_from_reference(template, references, method) for template in templates
    )


def _project_from_reference(
    template: AggregateWindow,
    references: Sequence[AggregateWindow],
    method: str,
) -> ForecastedWindow:
    rate = _mean([window.arrival_rate for window in references])
    request_count = max(1, round(rate * template.duration_s))
    tokens_per_request = _mean([window.tokens_per_request for window in references])
    return ForecastedWindow(
        start_s=template.start_s,
        duration_s=template.duration_s,
        request_count=request_count,
        total_tokens=max(1, round(tokens_per_request * request_count)),
        archetype_weights=_mean_weights(references),
        source_method=method,
        model_class=template.model_class,
        slo_class=template.slo_class,
        region=template.region,
        p50_total_tokens=_mean_optional_int(
            [window.p50_total_tokens for window in references]
        ),
        p95_total_tokens=_mean_optional_int(
            [window.p95_total_tokens for window in references]
        ),
        p99_ttft_ms=_mean_optional([window.p99_ttft_ms for window in references]),
        uncertainty={
            "arrival_rate_stddev": _stddev(
                [window.arrival_rate for window in references]
            ),
            "weight_stddev": _weight_stddev(references),
        },
    )


def _mean_weights(windows: Sequence[AggregateWindow]) -> dict[str, float]:
    ids = tuple(windows[-1].archetype_weights)
    return _normalise_weights(
        {
            archetype_id: _mean(
                [window.archetype_weights[archetype_id] for window in windows]
            )
            for archetype_id in ids
        },
        fallback=windows[-1].archetype_weights,
    )


def _weight_stddev(windows: Sequence[AggregateWindow]) -> dict[str, float]:
    ids = tuple(windows[-1].archetype_weights)
    return {
        archetype_id: _stddev(
            [window.archetype_weights[archetype_id] for window in windows]
        )
        for archetype_id in ids
    }


def _normalise_weights(
    weights: dict[str, float],
    fallback: dict[str, float],
) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0 or math.isnan(total):
        return dict(fallback)
    return {key: value / total for key, value in weights.items()}


def _linear_predict(values: Sequence[float], x: int) -> float:
    if len(values) == 1:
        return float(values[0])
    xs = list(range(len(values)))
    x_mean = _mean(xs)
    y_mean = _mean(values)
    denom = sum((item - x_mean) ** 2 for item in xs)
    if denom <= 0:
        return y_mean
    slope = sum((item - x_mean) * (value - y_mean) for item, value in zip(xs, values))
    slope /= denom
    return y_mean + slope * (x - x_mean)


def _linear_optional(values: Sequence[float | None], x: int) -> float | None:
    available = [float(value) for value in values if value is not None]
    if not available:
        return None
    return max(0.0, _linear_predict(available, min(x, len(available))))


def _linear_optional_int(values: Sequence[int | None], x: int) -> int | None:
    value = _linear_optional(values, x)
    return max(1, round(value)) if value is not None else None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: Sequence[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    if not available:
        return None
    return _mean(available)


def _mean_optional_int(values: Sequence[int | None]) -> int | None:
    value = _mean_optional(values)
    return max(1, round(value)) if value is not None else None


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
