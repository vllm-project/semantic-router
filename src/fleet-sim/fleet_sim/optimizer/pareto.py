"""Pareto frontier helpers for threshold sweeps."""

from __future__ import annotations

from dataclasses import dataclass

from ..gpu_profiles.profiles import GpuProfile
from .base import FleetOptimizer, _cdf_eval


@dataclass
class ThresholdResult:
    """One point on the threshold-cost-latency Pareto frontier."""

    b_short: int
    alpha: float
    n_s: int
    n_l: int
    total_gpus: int
    cost_kusd_yr: float
    savings_vs_homo_pct: float
    p99_short_ms: float
    p99_long_ms: float
    worst_p99_ms: float
    slo_met: bool
    pareto: bool = False


def threshold_pareto(
    cdf: list,
    lam: float,
    gpu_short: GpuProfile,
    gpu_long: GpuProfile,
    t_slo_ms: float = 500.0,
    long_max_ctx: int = 8192,
    gamma: float = 1.0,
) -> list[ThresholdResult]:
    """Sweep all CDF breakpoints as candidate B_short thresholds."""
    cdf_toks = [t for t, _ in cdf]
    candidates = [t for t in cdf_toks[:-1] if 0.01 <= _cdf_eval(cdf, t) <= 0.999]

    homo_opt = FleetOptimizer(
        gpu_short=gpu_short,
        gpu_long=gpu_long,
        B_short=long_max_ctx,
        t_slo_ms=t_slo_ms,
        long_max_ctx=long_max_ctx,
    )
    homo_sweep = homo_opt.sweep_analytical(cdf, lam, gammas=[gamma], verbose=False)
    homo_cost = homo_sweep[0].cost_per_hr * 8760 / 1000 if homo_sweep else 1e9

    results: list[ThresholdResult] = []
    for b_short in candidates:
        opt = FleetOptimizer(
            gpu_short=gpu_short,
            gpu_long=gpu_long,
            B_short=b_short,
            t_slo_ms=t_slo_ms,
            long_max_ctx=long_max_ctx,
        )
        sweep = opt.sweep_analytical(cdf, lam, gammas=[gamma], verbose=False)
        if not sweep:
            continue
        sr = sweep[0]
        cost_kusd_yr = sr.cost_per_hr * 8760 / 1000
        savings = (homo_cost - cost_kusd_yr) / homo_cost * 100 if homo_cost > 0 else 0.0
        results.append(
            ThresholdResult(
                b_short=b_short,
                alpha=_cdf_eval(cdf, b_short),
                n_s=sr.n_s,
                n_l=sr.n_l,
                total_gpus=sr.total_gpus,
                cost_kusd_yr=cost_kusd_yr,
                savings_vs_homo_pct=savings,
                p99_short_ms=sr.p99_ttft_short_ms,
                p99_long_ms=sr.p99_ttft_long_ms,
                worst_p99_ms=max(sr.p99_ttft_short_ms, sr.p99_ttft_long_ms),
                slo_met=sr.slo_met,
            )
        )

    _mark_pareto_frontier(results)
    results.sort(key=lambda r: r.b_short)
    return results


def print_threshold_pareto(
    results: list[ThresholdResult], t_slo_ms: float, homo_cost_kusd: float
) -> None:
    """Print a formatted Pareto frontier table."""
    del t_slo_ms, homo_cost_kusd
    print(
        f"\n  {'B_short':>8}  {'α-short':>8}  {'n_s':>4} {'n_l':>4}"
        f"  {'GPUs':>5}  {'$/yr':>8}  {'saving':>7}"
        f"  {'P99-s':>7}  {'P99-l':>7}  {'SLO':>4}  {'Pareto':>6}"
    )
    print(f"  {'-' * 87}")
    for result in results:
        ok = "✓" if result.slo_met else "✗"
        star = "★" if result.pareto else " "
        print(
            f"  {result.b_short:>8,}  {result.alpha:>7.1%}  {result.n_s:>4} {result.n_l:>4}"
            f"  {result.total_gpus:>5}  ${result.cost_kusd_yr:>6.0f}K"
            f"  {result.savings_vs_homo_pct:>+6.1f}%"
            f"  {result.p99_short_ms:>6.0f}ms  {result.p99_long_ms:>6.0f}ms"
            f"  {ok:>4}  {star:>6}"
        )

    pareto_slo = [result for result in results if result.pareto and result.slo_met]
    if not pareto_slo:
        return

    best = min(pareto_slo, key=lambda result: result.cost_kusd_yr)
    print(
        f"\n  Recommended B_short = {best.b_short:,} tokens"
        f"  (α={best.alpha:.1%} short, saving={best.savings_vs_homo_pct:+.1f}%,"
        f"  P99-short={best.p99_short_ms:.0f}ms, P99-long={best.p99_long_ms:.0f}ms)"
    )


def _mark_pareto_frontier(results: list[ThresholdResult]) -> None:
    for result in results:
        result.pareto = not any(
            other.cost_kusd_yr < result.cost_kusd_yr
            and other.worst_p99_ms < result.worst_p99_ms
            for other in results
            if other is not result
        )
