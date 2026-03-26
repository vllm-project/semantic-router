"""Fleet optimizer: find minimum-cost fleet configuration meeting SLO.

FleetOptimizer sweeps over fleet configurations using a combination of:
  1. Analytical M/G/c sizing (fast, approximate) to prune the search space
  2. DES simulation (accurate) to verify the top-N candidates

This two-phase approach is much faster than simulating all configurations.

Usage
-----
::

    from fleet_sim import FleetOptimizer, A100_80GB
    from fleet_sim.workload import CdfWorkload, PoissonWorkload

    workload = CdfWorkload("data/azure_cdf.json")
    optimizer = FleetOptimizer(
        gpu_short=A100_80GB,
        gpu_long=A100_80GB,
        B_short=6144,
        t_slo_ms=500,
    )
    result = optimizer.optimize(
        workload=workload,
        lam=200,
        gammas=[1.0, 1.1, 1.2, 1.5],
        n_gpus_range=range(1, 300),
    )
    result.print_report()
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..gpu_profiles.profiles import A100_80GB, GpuProfile
from ..workload.synthetic import CdfWorkload
from . import analytical

# ── ThresholdResult (Pareto sweep over B_short candidates) ────────────────────


@dataclass
class ThresholdResult:
    """One point on the threshold–cost–latency Pareto frontier."""

    b_short: int  # candidate short/long split (tokens)
    alpha: float  # fraction of requests routed to short pool
    n_s: int  # short-pool GPUs
    n_l: int  # long-pool GPUs
    total_gpus: int
    cost_kusd_yr: float
    savings_vs_homo_pct: (
        float  # positive = cheaper than homo; negative = more expensive
    )
    p99_short_ms: float
    p99_long_ms: float
    worst_p99_ms: float  # max(p99_short, p99_long) — single latency objective
    slo_met: bool
    pareto: bool = False  # True if no other threshold dominates on (cost, worst_p99)


def threshold_pareto(
    cdf: list,
    lam: float,
    gpu_short: GpuProfile,
    gpu_long: GpuProfile,
    t_slo_ms: float = 500.0,
    long_max_ctx: int = 8192,
    gamma: float = 1.0,
) -> list[ThresholdResult]:
    """Sweep all CDF breakpoints as candidate B_short thresholds.

    For each candidate, sizes the fleet analytically at gamma=1 (pure length
    routing) and records cost and P99 TTFT for both pools.  Marks each result
    as Pareto-optimal: a threshold is dominated if another threshold achieves
    strictly lower cost AND strictly lower worst-case P99.

    Parameters
    ----------
    cdf          : workload CDF as list of (token_threshold, cumulative_frac)
    lam          : total arrival rate (req/s)
    gpu_short    : GPU type for short pool
    gpu_long     : GPU type for long pool
    t_slo_ms     : P99 TTFT SLO (ms)
    long_max_ctx : max context length the long pool is configured for
    gamma        : C&R compression factor (1.0 = no compression)
    """
    # Candidate thresholds: all CDF breakpoints except the last (which is the max)
    cdf_toks = [t for t, _ in cdf]
    # Exclude breakpoints where virtually all traffic is already in the short pool
    # (alpha > 0.999) or where the short pool would receive < 1% of traffic.
    candidates = [
        t for t in cdf_toks[:-1] if 0.01 <= analytical.cdf_eval(cdf, t) <= 0.999
    ]

    # Homo baseline: B_short = long_max_ctx (single pool, no split)
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
    for b in candidates:
        opt = FleetOptimizer(
            gpu_short=gpu_short,
            gpu_long=gpu_long,
            B_short=b,
            t_slo_ms=t_slo_ms,
            long_max_ctx=long_max_ctx,
        )
        sweep = opt.sweep_analytical(cdf, lam, gammas=[gamma], verbose=False)
        if not sweep:
            continue
        sr = sweep[0]
        alpha = analytical.cdf_eval(cdf, b)
        cost = sr.cost_per_hr * 8760 / 1000
        saving = (homo_cost - cost) / homo_cost * 100 if homo_cost > 0 else 0.0
        worst = max(sr.p99_ttft_short_ms, sr.p99_ttft_long_ms)
        results.append(
            ThresholdResult(
                b_short=b,
                alpha=alpha,
                n_s=sr.n_s,
                n_l=sr.n_l,
                total_gpus=sr.total_gpus,
                cost_kusd_yr=cost,
                savings_vs_homo_pct=saving,
                p99_short_ms=sr.p99_ttft_short_ms,
                p99_long_ms=sr.p99_ttft_long_ms,
                worst_p99_ms=worst,
                slo_met=sr.slo_met,
            )
        )

    # Mark Pareto-optimal points: lower cost AND lower worst_p99
    for r in results:
        dominated = any(
            other.cost_kusd_yr < r.cost_kusd_yr and other.worst_p99_ms < r.worst_p99_ms
            for other in results
            if other is not r
        )
        r.pareto = not dominated

    results.sort(key=lambda r: r.b_short)
    return results


def print_threshold_pareto(
    results: list[ThresholdResult], t_slo_ms: float, homo_cost_kusd: float
) -> None:
    """Print a formatted Pareto frontier table."""
    print(
        f"\n  {'B_short':>8}  {'α-short':>8}  {'n_s':>4} {'n_l':>4}"
        f"  {'GPUs':>5}  {'$/yr':>8}  {'saving':>7}"
        f"  {'P99-s':>7}  {'P99-l':>7}  {'SLO':>4}  {'Pareto':>6}"
    )
    print(f"  {'-'*87}")
    for r in results:
        ok = "✓" if r.slo_met else "✗"
        star = "★" if r.pareto else " "
        print(
            f"  {r.b_short:>8,}  {r.alpha:>7.1%}  {r.n_s:>4} {r.n_l:>4}"
            f"  {r.total_gpus:>5}  ${r.cost_kusd_yr:>6.0f}K"
            f"  {r.savings_vs_homo_pct:>+6.1f}%"
            f"  {r.p99_short_ms:>6.0f}ms  {r.p99_long_ms:>6.0f}ms"
            f"  {ok:>4}  {star:>6}"
        )
    pareto_slo = [r for r in results if r.pareto and r.slo_met]
    if pareto_slo:
        best = min(pareto_slo, key=lambda r: r.cost_kusd_yr)
        print(
            f"\n  Recommended B_short = {best.b_short:,} tokens"
            f"  (α={best.alpha:.1%} short, saving={best.savings_vs_homo_pct:+.1f}%,"
            f"  P99-short={best.p99_short_ms:.0f}ms, P99-long={best.p99_long_ms:.0f}ms)"
        )


# ── SweepResult ───────────────────────────────────────────────────────────────


@dataclass
class SweepResult:
    """One data point from the optimizer sweep."""

    gamma: float
    n_s: int
    n_l: int
    total_gpus: int
    cost_per_hr: float
    annualised_cost_kusd: float
    p99_ttft_short_ms: float  # analytical or simulated
    p99_ttft_long_ms: float
    slo_met: bool
    source: str = "analytical"  # "analytical" or "simulated"

    @property
    def savings_vs_baseline_pct(self) -> float:
        return float("nan")


# ── FleetOptimizer ────────────────────────────────────────────────────────────


def node_availability(r_f_per_node_day: float, mttr_hours: float) -> float:
    """Steady-state availability of a single GPU node.

    Models each node as an M/M/1 repair queue:
        A = MTTF / (MTTF + MTTR)  =  1 / (1 + r_f * MTTR_days)

    **MTTR is primarily a function of failure TYPE, not GPU model.**
    Both A100 and H100 SXM modules are vendor-swapped on failure; physical
    repair timescales are similar (~24-72h).  What differs between GPU models
    is the *failure rate* r_f, not the repair time:

    ┌─────────────────────┬──────────────────────────────┬───────────────┐
    │ Failure type        │ Repair path                  │ MTTR          │
    ├─────────────────────┼──────────────────────────────┼───────────────┤
    │ Soft (driver/GSP)   │ Driver reset / node reboot   │ ~1–4 h        │
    │ Medium (OS/health)  │ Health-check quarantine +    │ ~4–24 h       │
    │                     │   node reimaging             │               │
    │ Hard (HBM/NVLink/   │ Physical GPU/HGX board swap  │ ~24–72 h      │
    │   PCIe die failure) │   by vendor (both A100/H100) │               │
    └─────────────────────┴──────────────────────────────┴───────────────┘

    GPU model differences (Cui et al. 2025, arXiv:2503.11901):
      A100: lower memory error rate; moderate hardware failure rate.
      H100: 3.2× higher memory MTBE (more soft ECC events, same MTTR);
            but significantly better critical-hardware resilience.
            Net recommendation: ~5% overprovisioning for H100 at scale
            → use node_avail ≈ 0.95 directly.

    Parameters
    ----------
    r_f_per_node_day : failure rate in failures per node per day.
        A100 RSC-1 (Meta research cluster): 6.50 / 1000  = 0.0065
        A100 RSC-2 (Meta vision cluster):   2.34 / 1000  = 0.00234
        H100 Delta system: see Cui et al. 2025 — overall failure rate
          slightly lower than A100 for hardware, but more ECC events.
        Source: Kokolis et al. 2024 (arXiv:2410.21680);
                Cui et al. 2025 (arXiv:2503.11901)
    mttr_hours : effective mean time to repair (hours).
        Use the FAILURE-TYPE-WEIGHTED average for your fleet, e.g.:
          4 h  if ~90% soft failures resolved by automated health checks
         24 h  conservative estimate mixing soft + medium failures
         48 h  worst-case (assumes many hard GPU/NVLink swaps)

    Returns
    -------
    float in (0, 1]: fraction of time the node is available.

    Examples
    --------
    >>> node_availability(0.0065, 4)    # A100 RSC-1, fast soft-failure path
    0.9989...
    >>> node_availability(0.0065, 48)   # A100 RSC-1, conservative hardware-swap
    0.9871...
    >>> # H100 at scale: use the Cui 2025 recommendation directly
    >>> node_avail_h100 = 0.95          # 5% overprovisioning rule of thumb
    """
    mttr_days = mttr_hours / 24.0
    return 1.0 / (1.0 + r_f_per_node_day * mttr_days)


# ── Pre-computed availability constants from published measurements ───────────
# Use these as starting points; tune from your own fleet telemetry.
#
# A100_AVAIL_RSC1_FAST  : Meta RSC-1 A100, r_f=0.0065, MTTR=4h  (soft failures)
# A100_AVAIL_RSC1_SLOW  : Meta RSC-1 A100, r_f=0.0065, MTTR=48h (hardware swap)
# H100_AVAIL_5PCT       : H100 at scale, 5% overprovisioning rule (Cui 2025)
A100_AVAIL_RSC1_FAST = node_availability(0.0065, 4)  # ≈ 0.9989
A100_AVAIL_RSC1_SLOW = node_availability(0.0065, 48)  # ≈ 0.9871
H100_AVAIL_5PCT = 0.95  # Cui et al. 2025


class FleetOptimizer:
    """Find minimum-cost (n_s, n_l, gamma) fleet meeting a P99 TTFT SLO.

    Parameters
    ----------
    gpu_short        : GPU type for short-context pool
    gpu_long         : GPU type for long-context pool (may differ for heterogeneous)
    B_short          : short/long context threshold (tokens)
    t_slo_ms         : P99 TTFT SLO target (ms)
    long_max_ctx     : maximum context length for long pool (tokens)
    p_c              : effective C&R compression success probability
    node_avail       : steady-state fraction of GPU nodes that are healthy
                       (default 1.0 = no reliability margin).
                       Use ``node_availability(r_f, mttr_h)`` to compute from
                       empirical failure-rate data (Kokolis et al. 2024).
                       Example: node_availability(0.0065, 48) ≈ 0.987 for
                       Meta RSC-1 with a 48-hour GPU-swap MTTR.
                       The optimizer inflates the raw SLO-sized GPU count by
                       1/node_avail to keep the SLO met even while (1-node_avail)
                       fraction of nodes are under repair.
    """

    def __init__(
        self,
        gpu_short: GpuProfile = A100_80GB,
        gpu_long: GpuProfile = A100_80GB,
        B_short: int = 4096,
        t_slo_ms: float = 500.0,
        long_max_ctx: int = 65536,
        p_c: float = 0.75,
        node_avail: float = 1.0,
    ):
        self.gpu_short = gpu_short
        self.gpu_long = gpu_long
        self.B_short = B_short
        self.t_slo_ms = t_slo_ms
        self.t_slo = t_slo_ms / 1000.0
        self.long_max_ctx = long_max_ctx
        # Effective compression success probability for the C&R analytical model.
        # The greedy compressor has p_c=1.0 per safe request; multiplying by the
        # safe-category fraction of borderline traffic gives the effective p_c.
        # Default 0.75 matches the agent-heavy mix (75% prose/RAG/mixed, 25% code).
        self.p_c = p_c
        if not (0.0 < node_avail <= 1.0):
            raise ValueError(f"node_avail must be in (0, 1]; got {node_avail}")
        self.node_avail = node_avail

    def sweep_analytical(
        self,
        cdf: list,
        lam: float,
        gammas: list[float] | None = None,
        verbose: bool = False,
    ) -> list[SweepResult]:
        """Sweep gamma values using the analytical M/G/c model.

        Returns a list of SweepResult sorted by total cost.
        """
        if gammas is None:
            gammas = [round(1.0 + 0.1 * k, 1) for k in range(11)]

        results = []
        alpha_base = analytical.cdf_eval(cdf, self.B_short)

        # Calibrate short pool (requests <= B_short, fixed)
        short_cdf = [(t, f) for t, f in cdf if t <= self.B_short]
        if short_cdf:
            last_f = short_cdf[-1][1]
            short_cdf_norm = [(t, f / last_f) for t, f in short_cdf]
        else:
            short_cdf_norm = [(self.B_short, 1.0)]

        mu_s, cv2_s, ns_s, pref_s = analytical.calibrate(
            short_cdf_norm, self.B_short, self.gpu_short
        )

        if verbose:
            print(
                f"\n  Analytical sweep  (λ={lam:.0f} req/s, SLO={self.t_slo_ms:.0f}ms)"
            )
            print(f"  B_short={self.B_short:,}  alpha_base={alpha_base:.4f}")
            print(
                f"  {'gamma':>5} {'alpha_p':>8} {'n_s':>5} {'n_l':>5} {'total':>7}"
                f" {'$/yr':>10} {'P99_s':>8} {'P99_l':>8} {'OK':>4}"
            )
            print(f"  {'-'*68}")

        for gamma in gammas:
            gamma_bs = int(gamma * self.B_short)
            f_at_gamma_bs = analytical.cdf_eval(cdf, gamma_bs)
            borderline_frac = max(0.0, f_at_gamma_bs - alpha_base)
            # p_c = 1.0: the greedy vLLM Semantic Router compressor always
            # hits the token budget for safe-category (prose/RAG/mixed)
            # requests.  The optimizer does not know the per-request category
            # split, so it uses the fraction of safe traffic in the borderline
            # band.  Without a workload-level safe_frac override we assume the
            # conservative default from the agent-heavy mix (75% safe, i.e.
            # p_c_eff = 0.75 * 1.0).  Pass p_c explicitly via OptimizeParams
            # to override for a specific workload.
            p_c = getattr(self, "p_c", 0.75)
            alpha_prime = min(1.0, alpha_base + borderline_frac * p_c)

            # Approximate sub-stream rates.  Poisson thinning is exact only
            # for independent per-arrival coin-flips; length-based routing
            # produces correlated sub-streams.  The DES step verifies the
            # resulting P99 TTFT empirically.
            lam_s = alpha_prime * lam
            lam_l = (1.0 - alpha_prime) * lam

            n_s_raw = analytical.min_gpus_analytical(
                lam_s, mu_s, self.t_slo, cv2_s, n_slots=ns_s
            )
            # Reliability margin: inflate so SLO still holds when (1-A) nodes are
            # under repair.  node_avail=1.0 (default) → no change.
            n_s = math.ceil(n_s_raw / self.node_avail)

            # Long pool: calibrate from requests > gamma*B_short.
            # Use interpolated CDF at gamma_bs for correct normalization.
            long_frac = 1.0 - f_at_gamma_bs
            long_cdf_raw_l = [(t, f) for t, f in cdf if t > gamma_bs]
            if long_cdf_raw_l and long_frac > 1e-6:
                long_cdf_norm = [
                    (t, max(0.0, min(1.0, (f - f_at_gamma_bs) / long_frac)))
                    for t, f in long_cdf_raw_l
                ]
                if long_cdf_norm:
                    long_cdf_norm[-1] = (long_cdf_norm[-1][0], 1.0)
                mu_l, cv2_l, ns_l, pref_l = analytical.calibrate(
                    long_cdf_norm,
                    self.long_max_ctx,
                    self.gpu_long,
                    lo_clamp=max(1, gamma_bs),
                )
            else:
                mu_l, cv2_l, ns_l, pref_l = 0.01, 1.0, 1, 0.0

            n_l_raw = (
                analytical.min_gpus_analytical(
                    lam_l, mu_l, self.t_slo, cv2_l, n_slots=ns_l
                )
                if lam_l > 0.01
                else 0
            )
            n_l = math.ceil(n_l_raw / self.node_avail)

            # P99 TTFT = P99 slot wait + mean prefill time
            p99_s = (
                analytical.p99_wait(n_s * ns_s, lam_s, mu_s / ns_s, cv2_s) + pref_s
            ) * 1000
            p99_l = (
                (analytical.p99_wait(n_l * ns_l, lam_l, mu_l / ns_l, cv2_l) + pref_l)
                * 1000
                if lam_l > 0.01
                else 0.0
            )
            total = n_s + n_l
            cost_hr = n_s * self.gpu_short.cost_per_hr + n_l * self.gpu_long.cost_per_hr
            ann_k = cost_hr * 8760 / 1000

            sr = SweepResult(
                gamma=gamma,
                n_s=n_s,
                n_l=n_l,
                total_gpus=total,
                cost_per_hr=cost_hr,
                annualised_cost_kusd=ann_k,
                p99_ttft_short_ms=p99_s,
                p99_ttft_long_ms=p99_l,
                slo_met=(p99_s <= self.t_slo_ms and p99_l <= self.t_slo_ms),
                source="analytical",
            )
            results.append(sr)

            if verbose:
                ok = "✓" if sr.slo_met else "✗"
                print(
                    f"  {gamma:>5.1f} {alpha_prime:>8.4f} {n_s:>5} {n_l:>5}"
                    f" {total:>7} ${ann_k:>8.1f}K {p99_s:>7.1f}ms"
                    f" {p99_l:>7.1f}ms {ok:>4}"
                )

        results.sort(key=lambda r: r.cost_per_hr)
        return results

    def optimize(
        self,
        cdf: list,
        lam: float,
        gammas: list[float] | None = None,
        n_sim_requests: int = 40_000,
        verify_top_n: int = 3,
        verbose: bool = True,
    ) -> OptimizationReport:
        """Full two-phase optimization: analytical sweep + DES verification.

        Parameters
        ----------
        cdf            : empirical CDF list of (token_threshold, cumulative_frac)
        lam            : total arrival rate (req/s)
        gammas         : gamma values to sweep (default [1.0, 1.1, ..., 2.0])
        n_sim_requests : requests to simulate in DES verification phase
        verify_top_n   : how many top-N analytical solutions to verify in DES
        verbose        : print progress
        """
        # Phase 1: analytical sweep
        if verbose:
            print("[1/2] Analytical sweep...")
        candidates = self.sweep_analytical(cdf, lam, gammas, verbose=verbose)

        # Phase 2: DES verification of top-N (optional)
        verified = []
        if verify_top_n > 0 and n_sim_requests > 0:
            if verbose:
                print(f"\n[2/2] DES verification of top-{verify_top_n} candidates...")
            wl_gen = CdfWorkload(cdf)
            for sr in candidates[:verify_top_n]:
                if verbose:
                    print(f"  Verifying γ={sr.gamma} (n_s={sr.n_s}, n_l={sr.n_l})...")
                sim_result = self._run_des(
                    cdf=cdf,
                    lam=lam,
                    gamma=sr.gamma,
                    n_s=sr.n_s,
                    n_l=sr.n_l,
                    n_req=n_sim_requests,
                )
                sr_v = SweepResult(
                    gamma=sr.gamma,
                    n_s=sr.n_s,
                    n_l=sr.n_l,
                    total_gpus=sr.total_gpus,
                    cost_per_hr=sr.cost_per_hr,
                    annualised_cost_kusd=sr.annualised_cost_kusd,
                    p99_ttft_short_ms=sim_result.get("p99_short_ms", 0.0),
                    p99_ttft_long_ms=sim_result.get("p99_long_ms", 0.0),
                    slo_met=(
                        sim_result.get("p99_short_ms", 999) <= self.t_slo_ms
                        and sim_result.get("p99_long_ms", 999) <= self.t_slo_ms
                    ),
                    source="simulated",
                )
                verified.append(sr_v)
                if verbose:
                    ok = "✓" if sr_v.slo_met else "✗"
                    print(
                        f"    P99 short={sr_v.p99_ttft_short_ms:.1f}ms  "
                        f"long={sr_v.p99_ttft_long_ms:.1f}ms  SLO:{ok}"
                    )

        return OptimizationReport(
            B_short=self.B_short,
            t_slo_ms=self.t_slo_ms,
            lam=lam,
            analytical=candidates,
            simulated=verified,
        )

    def _run_des(
        self, cdf: list, lam: float, gamma: float, n_s: int, n_l: int, n_req: int
    ) -> dict:
        """Run DES for one (gamma, n_s, n_l) configuration."""

        alpha_base = analytical.cdf_eval(cdf, self.B_short)
        gamma_bs = int(gamma * self.B_short)
        f_at_gamma_bs = analytical.cdf_eval(cdf, gamma_bs)
        borderline_frac = max(0.0, f_at_gamma_bs - alpha_base)
        # Same p_c logic as the analytical sweep: greedy compressor p_c=1.0
        # for safe cats; effective p_c = safe_fraction * 1.0 (default 0.75).
        p_c = getattr(self, "p_c", 0.75)
        alpha_prime = min(1.0, alpha_base + borderline_frac * p_c)
        # Same sub-stream approximation as the analytical sweep (see comment above).
        lam_s = alpha_prime * lam
        lam_l = (1.0 - alpha_prime) * lam

        # Simulate short pool
        short_cdf_raw = [(t, f) for t, f in cdf if t <= self.B_short]
        if short_cdf_raw:
            lf = short_cdf_raw[-1][1]
            short_cdf = [(t, f / lf) for t, f in short_cdf_raw]
        else:
            short_cdf = [(self.B_short, 1.0)]

        # Build long pool CDF: requests with L > gamma_bs.
        # Insert a virtual breakpoint at gamma_bs using interpolation so the
        # normalized CDF properly represents the [gamma_bs+1, long_max] mass.
        f_at_gbs = analytical.cdf_eval(cdf, gamma_bs)  # interpolated
        long_frac = 1.0 - f_at_gbs
        if long_frac > 1e-6:
            # Entries strictly above gamma_bs, plus virtual start point
            long_cdf_raw = [(t, f) for t, f in cdf if t > gamma_bs]
            # Normalize: subtract f_at_gbs and divide by long_frac
            long_cdf = [
                (t, max(0.0, min(1.0, (f - f_at_gbs) / long_frac)))
                for t, f in long_cdf_raw
            ]
            if long_cdf:
                long_cdf[-1] = (long_cdf[-1][0], 1.0)  # clip to 1.0
            else:
                long_cdf = [(self.long_max_ctx, 1.0)]
        else:
            long_cdf = [(self.long_max_ctx, 1.0)]

        # Fast M/G/c DES (heap-based, slot-level model).
        # Each KV-cache slot is modelled as a separate server holding one request
        # for the full physical service_time.  This gives the correct slot-wait
        # distribution regardless of per-GPU concurrency (n_slots).
        import heapq
        import math as _math
        import random

        rng = random.Random(42)
        n_slots_s = self.gpu_short.n_slots(self.B_short)
        gpu_s = self.gpu_short

        def st_short_with_prefill(total: int):
            """Return (service_time, prefill_time_s) for a short-pool request."""
            l_in = max(1, int(total * 0.8))
            l_out = max(1, total - l_in)
            st = gpu_s.service_time(l_in, l_out, self.B_short)
            pref = _math.ceil(l_in / gpu_s.chunk) * gpu_s.iter_latency(1)
            return st, pref

        def sample_short_st() -> tuple:
            u = rng.random()
            prev = 0
            for t, f in short_cdf:
                if u <= f:
                    length = rng.randint(max(1, prev), t)
                    return st_short_with_prefill(length)
                prev = t
            return st_short_with_prefill(short_cdf[-1][0])

        def sim_pool_slots(st_fn, n_gpu, n_sl, lam_pool, n_r):
            """Simulate M/G/c queue with c = n_gpu * n_sl KV slots.

            Returns P99 TTFT = P99(slot_wait + prefill_time) in milliseconds.
            """
            total_slots = max(1, n_gpu * n_sl)
            servers = [0.0] * total_slots
            heapq.heapify(servers)
            warm = n_r // 5
            ttfts = []
            t = 0.0
            for i in range(n_r + warm):
                t += rng.expovariate(max(lam_pool, 1e-9))
                st, pref = st_fn()
                free = heapq.heappop(servers)
                slot_wait = max(0.0, free - t)
                heapq.heappush(servers, max(free, t) + st)
                if i >= warm:
                    ttfts.append(slot_wait + pref)
            ttfts.sort()
            idx = max(0, int(len(ttfts) * 0.99) - 1)
            return ttfts[idx] * 1000 if ttfts else 0.0

        p99_s = sim_pool_slots(sample_short_st, n_s, n_slots_s, lam_s, n_req)

        if lam_l > 0.1 and n_l > 0:
            n_slots_l = self.gpu_long.n_slots(self.long_max_ctx)
            gpu_l = self.gpu_long

            def st_long_with_prefill(total: int):
                l_in = max(1, int(total * 0.8))
                l_out = max(1, total - l_in)
                st = gpu_l.service_time(l_in, l_out, self.long_max_ctx)
                pref = _math.ceil(l_in / gpu_l.chunk) * gpu_l.iter_latency(1)
                return st, pref

            def sample_long_st() -> tuple:
                u = rng.random()
                prev = gamma_bs
                for t, f in long_cdf:
                    if u <= f:
                        lo = prev + 1
                        return st_long_with_prefill(rng.randint(lo, max(lo, t)))
                    prev = t
                return st_long_with_prefill(long_cdf[-1][0])

            p99_l = sim_pool_slots(sample_long_st, n_l, n_slots_l, lam_l, n_req)
        else:
            p99_l = 0.0

        return {"p99_short_ms": p99_s, "p99_long_ms": p99_l}


# ── OptimizationReport ────────────────────────────────────────────────────────


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
        print(f"\n{'='*60}")
        print("  Fleet Optimization Report")
        print(
            f"  B_short={self.B_short:,}  λ={self.lam:.0f} req/s"
            f"  SLO={self.t_slo_ms:.0f}ms"
        )
        print(f"{'='*60}")
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

        # Comparison table
        if len(self.analytical) > 1:
            baseline = self.analytical[0]  # already sorted by cost
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
                sav = (
                    (baseline.cost_per_hr - sr.cost_per_hr) / baseline.cost_per_hr * 100
                )
                ok = "✓" if sr.slo_met else "✗"
                print(
                    f"  {sr.gamma:>5.1f} {sr.n_s:>5} {sr.n_l:>5}"
                    f" {sr.total_gpus:>7} ${sr.annualised_cost_kusd:>7.1f}K"
                    f" {sav:>+7.1f}% {sr.p99_ttft_short_ms:>7.1f}ms"
                    f" {sr.p99_ttft_long_ms:>7.1f}ms {ok}"
                )
