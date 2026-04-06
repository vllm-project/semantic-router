"""Reporting helpers for the aggregated fleet optimizer."""

from __future__ import annotations

from .base import SweepResult


class OptimizationReport:
    """Results of a full FleetOptimizer run."""

    def __init__(
        self,
        B_short: int,
        t_slo_ms: float,
        lam: float,
        analytical: list[SweepResult],
        simulated: list[SweepResult],
    ):
        self.B_short = B_short
        self.t_slo_ms = t_slo_ms
        self.lam = lam
        self.analytical = analytical
        self.simulated = simulated

    @property
    def best_analytical(self) -> SweepResult | None:
        valid = [r for r in self.analytical if r.slo_met]
        return (
            min(valid, key=lambda r: r.cost_per_hr)
            if valid
            else (self.analytical[0] if self.analytical else None)
        )

    @property
    def best_simulated(self) -> SweepResult | None:
        valid = [r for r in self.simulated if r.slo_met]
        return (
            min(valid, key=lambda r: r.cost_per_hr)
            if valid
            else (self.simulated[0] if self.simulated else None)
        )

    def print_report(self) -> None:
        ba = self.best_analytical
        bs = self.best_simulated
        print(f"\n{'=' * 60}")
        print("  Fleet Optimization Report")
        print(
            f"  B_short={self.B_short:,}  λ={self.lam:.0f} req/s  SLO={self.t_slo_ms:.0f}ms"
        )
        print(f"{'=' * 60}")
        if ba:
            print(
                f"\n  Best (analytical): γ={ba.gamma}  "
                f"n_s={ba.n_s}  n_l={ba.n_l}  total={ba.total_gpus}  "
                f"${ba.annualised_cost_kusd:.1f}K/yr"
            )
        if bs:
            print(
                f"  Best (simulated):  γ={bs.gamma}  "
                f"n_s={bs.n_s}  n_l={bs.n_l}  total={bs.total_gpus}  "
                f"${bs.annualised_cost_kusd:.1f}K/yr"
            )
            print(
                f"    P99 TTFT: short={bs.p99_ttft_short_ms:.1f}ms  "
                f"long={bs.p99_ttft_long_ms:.1f}ms  "
                f"SLO:{'✓' if bs.slo_met else '✗'}"
            )

        if len(self.analytical) <= 1:
            return

        baseline = self.analytical[0]
        for sr in self.analytical:
            if sr.gamma == 1.0:
                baseline = sr
                break

        print(
            f"\n  γ sweep (analytical, baseline=γ=1.0 with {baseline.total_gpus} GPUs):"
        )
        print(
            f"  {'γ':>5} {'n_s':>5} {'n_l':>5} {'total':>7}"
            f" {'$K/yr':>9} {'saving':>8} {'P99_s':>8} {'P99_l':>8}"
        )
        for sr in sorted(self.analytical, key=lambda r: r.gamma):
            saving = (
                (baseline.cost_per_hr - sr.cost_per_hr) / baseline.cost_per_hr * 100
            )
            ok = "✓" if sr.slo_met else "✗"
            print(
                f"  {sr.gamma:>5.1f} {sr.n_s:>5} {sr.n_l:>5}"
                f" {sr.total_gpus:>7} ${sr.annualised_cost_kusd:>7.1f}K"
                f" {saving:>+7.1f}% {sr.p99_ttft_short_ms:>7.1f}ms"
                f" {sr.p99_ttft_long_ms:>7.1f}ms {ok}"
            )
