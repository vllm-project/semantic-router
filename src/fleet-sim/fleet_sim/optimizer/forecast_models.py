"""Report models for workload archetype forecast backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..workload.forecast import ForecastedWindow
from .mixture_models import MixtureOptimizationReport


class ForecastBacktestError(RuntimeError):
    """Raised when forecast backtesting is explicitly required to pass."""


@dataclass(frozen=True)
class ForecastCapacityRecommendation:
    """Advise-only fleet capacity recommendation derived from forecast windows."""

    method: str
    n_s: int
    n_l: int
    total_gpus: int
    cost_per_hr: float
    annualised_cost_kusd: float
    worst_case_id: str
    slo_met: bool
    gamma: float
    source: str = "forecast_window"


@dataclass(frozen=True)
class ForecastActuationRecord:
    """Downstream acknowledgement/action record, stored separately by design."""

    method: str
    status: str
    downstream_action: str | None = None
    acknowledged_at_s: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ForecastWindowBacktest:
    """One forecast-vs-actual aggregate window comparison."""

    start_s: float
    duration_s: float
    actual_arrival_rate: float
    forecast_arrival_rate: float
    arrival_rate_error_pct: float
    actual_tokens_per_request: float
    forecast_tokens_per_request: float
    tokens_per_request_error_pct: float
    actual_p95_total_tokens: int | None
    forecast_p95_total_tokens: int | None
    p95_total_tokens_error_pct: float | None
    actual_p99_ttft_ms: float | None
    forecast_p99_ttft_ms: float | None
    p99_ttft_error_pct: float | None
    weight_l1_error: float
    actual_weights: dict[str, float]
    forecast_weights: dict[str, float]
    covered_by_arrival_uncertainty: bool


@dataclass
class ForecastMethodBacktest:
    """Backtest result for one control/forecast method."""

    method: str
    control_kind: str
    forecast_windows: tuple[ForecastedWindow, ...]
    window_results: tuple[ForecastWindowBacktest, ...]
    mixture_report: MixtureOptimizationReport
    recommendation: ForecastCapacityRecommendation | None
    actual_required: ForecastCapacityRecommendation | None
    diagnostics: tuple[str, ...] = ()
    beats_static_control: bool | None = None
    beats_reactive_control: bool | None = None

    @property
    def mean_abs_arrival_rate_error_pct(self) -> float:
        if not self.window_results:
            return 0.0
        return sum(
            abs(item.arrival_rate_error_pct) for item in self.window_results
        ) / len(self.window_results)

    @property
    def mean_weight_l1_error(self) -> float:
        if not self.window_results:
            return 0.0
        return sum(item.weight_l1_error for item in self.window_results) / len(
            self.window_results
        )

    @property
    def mean_abs_tokens_per_request_error_pct(self) -> float:
        if not self.window_results:
            return 0.0
        return sum(
            abs(item.tokens_per_request_error_pct) for item in self.window_results
        ) / len(self.window_results)

    @property
    def mean_abs_p95_total_tokens_error_pct(self) -> float:
        values = [
            abs(item.p95_total_tokens_error_pct)
            for item in self.window_results
            if item.p95_total_tokens_error_pct is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_abs_p99_ttft_error_pct(self) -> float:
        values = [
            abs(item.p99_ttft_error_pct)
            for item in self.window_results
            if item.p99_ttft_error_pct is not None
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def score(self) -> float:
        demand_error_pct = (
            self.mean_abs_arrival_rate_error_pct
            + self.mean_abs_tokens_per_request_error_pct
            + self.mean_abs_p95_total_tokens_error_pct
            + self.mean_abs_p99_ttft_error_pct
        ) / 4.0
        return self.mean_weight_l1_error + demand_error_pct / 100.0

    @property
    def headroom_gpus_vs_actual(self) -> int | None:
        if self.recommendation is None or self.actual_required is None:
            return None
        return self.recommendation.total_gpus - self.actual_required.total_gpus


@dataclass
class ForecastBacktestReport:
    """Structured backtest report for proactive archetype forecasts."""

    scenario_id: str
    scenario_version: str
    source_mixture_id: str
    base_lam: float
    methods: tuple[ForecastMethodBacktest, ...]
    actual_report: MixtureOptimizationReport
    actual_required: ForecastCapacityRecommendation | None
    recommended_method: str | None
    diagnostics: tuple[str, ...] = ()
    actuation_records: tuple[ForecastActuationRecord, ...] = ()
    fail_safe_triggered: bool = False
    fallback_control: str | None = None
    rollback_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.actual_required is not None and all(
            method.recommendation is not None for method in self.methods
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_version": self.scenario_version,
            "source_mixture_id": self.source_mixture_id,
            "base_lam": self.base_lam,
            "ok": self.ok,
            "fail_safe_triggered": self.fail_safe_triggered,
            "fallback_control": self.fallback_control,
            "rollback_reason": self.rollback_reason,
            "recommended_method": self.recommended_method,
            "actual_required": (
                asdict(self.actual_required) if self.actual_required else None
            ),
            "diagnostics": list(self.diagnostics),
            "actuation_records": [asdict(item) for item in self.actuation_records],
            "actual_report": self.actual_report.to_dict(),
            "methods": [_method_to_dict(item) for item in self.methods],
        }

    def print_report(self) -> None:
        print("\nForecast Backtest Report")
        print(
            f"  scenario={self.scenario_id} version={self.scenario_version} "
            f"source_mixture={self.source_mixture_id} base_lambda={self.base_lam:.2f}"
        )
        if self.actual_required:
            print(
                f"  actual required: n_s={self.actual_required.n_s} "
                f"n_l={self.actual_required.n_l} "
                f"total={self.actual_required.total_gpus} GPUs"
            )
        if self.fail_safe_triggered:
            print(
                f"  fail-safe: {self.fallback_control} "
                f"rollback_reason={self.rollback_reason}"
            )
        print("\n  Recommendations are advise-only; actuation is recorded separately.")
        if self.actuation_records:
            for record in self.actuation_records:
                print(
                    f"  actuation: method={record.method} status={record.status} "
                    f"action={record.downstream_action}"
                )
        else:
            print("  actuation: none recorded")
        print("\n  Methods:")
        print(
            "  method                 kind       err_rate err_token  err_mix  GPUs  "
            "headroom  status"
        )
        for method in self.methods:
            rec = method.recommendation
            gpus = rec.total_gpus if rec else 0
            headroom = method.headroom_gpus_vs_actual
            headroom_text = f"{headroom:+d}" if headroom is not None else "n/a"
            status = "ok" if rec and rec.slo_met else "no recommendation"
            print(
                f"  {method.method:<22} {method.control_kind:<10} "
                f"{method.mean_abs_arrival_rate_error_pct:>7.1f}% "
                f"{method.mean_abs_tokens_per_request_error_pct:>8.1f}% "
                f"{method.mean_weight_l1_error:>7.3f} "
                f"{gpus:>5} {headroom_text:>9}  {status}"
            )
        if self.recommended_method:
            print(f"  recommended_method: {self.recommended_method}")
        for diagnostic in self.diagnostics:
            print(f"  diagnostic: {diagnostic}")


def _method_to_dict(method: ForecastMethodBacktest) -> dict[str, Any]:
    return {
        "method": method.method,
        "control_kind": method.control_kind,
        "mean_abs_arrival_rate_error_pct": method.mean_abs_arrival_rate_error_pct,
        "mean_abs_tokens_per_request_error_pct": (
            method.mean_abs_tokens_per_request_error_pct
        ),
        "mean_abs_p95_total_tokens_error_pct": (
            method.mean_abs_p95_total_tokens_error_pct
        ),
        "mean_abs_p99_ttft_error_pct": method.mean_abs_p99_ttft_error_pct,
        "mean_weight_l1_error": method.mean_weight_l1_error,
        "score": method.score,
        "beats_static_control": method.beats_static_control,
        "beats_reactive_control": method.beats_reactive_control,
        "headroom_gpus_vs_actual": method.headroom_gpus_vs_actual,
        "recommendation": (
            asdict(method.recommendation) if method.recommendation else None
        ),
        "actual_required": (
            asdict(method.actual_required) if method.actual_required else None
        ),
        "diagnostics": list(method.diagnostics),
        "window_results": [asdict(item) for item in method.window_results],
        "forecast_windows": [asdict(item) for item in method.forecast_windows],
        "mixture_report": method.mixture_report.to_dict(),
    }
