"""Report models for FleetOptimizer workload mixture evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .base import SweepResult


class MixtureOptimizationError(RuntimeError):
    """Raised when a mixture optimization report is explicitly required to pass."""


@dataclass(frozen=True)
class MixtureStressCase:
    """One distribution/arrival-rate case derived from a mixture scenario."""

    id: str
    kind: str
    lam: float
    weights: dict[str, float]
    cdf: list[tuple[int, float]]
    description: str = ""


@dataclass
class MixtureCaseResult:
    """FleetOptimizer result for one mixture stress case."""

    case_id: str
    kind: str
    lam: float
    weights: dict[str, float]
    best: SweepResult | None
    baseline: SweepResult | None
    sweep: tuple[SweepResult, ...]
    infeasible_reason: str | None = None
    annual_cost_delta_vs_nominal_kusd: float | None = None
    cost_sensitivity_pct: float | None = None

    @property
    def slo_met(self) -> bool:
        return (
            self.infeasible_reason is None
            and self.best is not None
            and self.best.slo_met
        )


@dataclass(frozen=True)
class RobustMixtureRecommendation:
    """A single gamma/fleet recommendation sized across all stress cases."""

    gamma: float
    n_s: int
    n_l: int
    total_gpus: int
    cost_per_hr: float
    annualised_cost_kusd: float
    worst_case_id: str
    worst_p99_ttft_short_ms: float
    worst_p99_ttft_long_ms: float
    slo_met: bool
    infeasible_reason: str | None = None


@dataclass
class MixtureOptimizationReport:
    """Comparison report for aggregate, nominal, stress, and robust mixture cases."""

    scenario_id: str
    scenario_version: str
    lam: float
    cases: tuple[MixtureCaseResult, ...]
    robust_recommendation: RobustMixtureRecommendation | None
    diagnostics: tuple[str, ...] = ()
    aggregate_baseline_case_id: str = "aggregate-cdf"
    nominal_case_id: str = "nominal-mixture"

    @property
    def ok(self) -> bool:
        return self.robust_recommendation is not None and all(
            case.slo_met for case in self.cases
        )

    @property
    def worst_case(self) -> MixtureCaseResult | None:
        feasible = [case for case in self.cases if case.best is not None]
        if not feasible:
            return None
        return max(feasible, key=lambda case: case.best.annualised_cost_kusd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_version": self.scenario_version,
            "lam": self.lam,
            "ok": self.ok,
            "aggregate_baseline_case_id": self.aggregate_baseline_case_id,
            "nominal_case_id": self.nominal_case_id,
            "robust_recommendation": (
                asdict(self.robust_recommendation)
                if self.robust_recommendation is not None
                else None
            ),
            "diagnostics": list(self.diagnostics),
            "cases": [_case_to_dict(case) for case in self.cases],
        }

    def print_report(self) -> None:
        print("\nMixture Optimization Report")
        print(
            f"  scenario={self.scenario_id} version={self.scenario_version} "
            f"lambda={self.lam:.2f} req/s"
        )
        if self.robust_recommendation:
            rr = self.robust_recommendation
            print(
                f"  robust: gamma={rr.gamma:.2f} n_s={rr.n_s} n_l={rr.n_l} "
                f"total={rr.total_gpus} ${rr.annualised_cost_kusd:.1f}K/yr "
                f"worst={rr.worst_case_id}"
            )
        else:
            print("  robust: infeasible")
        print("\n  Cases:")
        print(
            "  case                         kind                 lam     GPUs   "
            "$K/yr   sensitivity  status"
        )
        for case in self.cases:
            best = case.best
            total = best.total_gpus if best else 0
            cost = best.annualised_cost_kusd if best else 0.0
            sensitivity = (
                f"{case.cost_sensitivity_pct:+.1f}%"
                if case.cost_sensitivity_pct is not None
                else "n/a"
            )
            status = "ok" if case.slo_met else f"fail: {case.infeasible_reason}"
            print(
                f"  {case.case_id:<28} {case.kind:<20} {case.lam:>6.1f} "
                f"{total:>6} {cost:>7.1f} {sensitivity:>11}  {status}"
            )
        for diagnostic in self.diagnostics:
            print(f"  diagnostic: {diagnostic}")


def _case_to_dict(case: MixtureCaseResult) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "lam": case.lam,
        "weights": case.weights,
        "best": asdict(case.best) if case.best is not None else None,
        "baseline": asdict(case.baseline) if case.baseline is not None else None,
        "sweep": [asdict(item) for item in case.sweep],
        "slo_met": case.slo_met,
        "infeasible_reason": case.infeasible_reason,
        "annual_cost_delta_vs_nominal_kusd": case.annual_cost_delta_vs_nominal_kusd,
        "cost_sensitivity_pct": case.cost_sensitivity_pct,
    }
