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

from ..core.fleet import Fleet, FleetConfig, PoolConfig
from ..gpu_profiles.profiles import A100_80GB, GpuProfile
from ..workload.synthetic import CdfWorkload, PoissonWorkload

# ── CDF utilities ─────────────────────────────────────────────────────────────


def _cdf_eval(cdf: list, t: int) -> float:
    """Evaluate CDF at token length t using linear interpolation."""
    if t <= 0:
        return 0.0
    prev_t, prev_f = 0, 0.0
    for thresh, frac in cdf:
        if t <= thresh:
            # Interpolate between (prev_t, prev_f) and (thresh, frac)
            if thresh == prev_t:
                return frac
            return prev_f + (frac - prev_f) * (t - prev_t) / (thresh - prev_t)
        prev_t, prev_f = thresh, frac
    return 1.0


# ── Analytical sizing (Erlang-C / Kimura) ────────────────────────────────────


def _erlang_c(c: int, a: float) -> float:
    """Numerically stable Erlang-C P(W_q > 0)."""
    if c <= 0 or a <= 0:
        return 0.0
    rho = a / c
    if rho >= 1.0:
        return 1.0
    import math

    log_sum = 0.0
    for k in range(c):
        log_sum_k = k * math.log(a) - math.lgamma(k + 1)
        if k == 0:
            log_sum = log_sum_k
        else:
            mx = max(log_sum, log_sum_k)
            log_sum = mx + math.log(math.exp(log_sum - mx) + math.exp(log_sum_k - mx))
    log_last = c * math.log(a) - math.lgamma(c + 1) - math.log(1 - rho)
    mx = max(log_sum, log_last)
    log_denom = mx + math.log(math.exp(log_sum - mx) + math.exp(log_last - mx))
    return math.exp(log_last - log_denom)


def _p99_wait(c: int, lam: float, mu: float, cv2: float = 1.0) -> float:
    """Kimura (1994) M/G/c P99 waiting time (s)."""
    if c <= 0 or lam <= 0 or mu <= 0:
        return float("inf")
    a = lam / mu
    rho = a / c
    if rho >= 1.0:
        return float("inf")
    C = _erlang_c(c, a)
    if C <= 0.01:
        return 0.0  # P(wait > 0) so small P99 wait ≈ 0
    decay = 2 * (c * mu - lam) / max(1e-9, 1 + cv2)
    if decay <= 0:
        return math.inf
    return math.log(C / 0.01) / decay


def _calibrate(
    cdf: list, pool_max: int, gpu: GpuProfile, lo_clamp: int = 1
) -> tuple[float, float, int, float]:
    """Estimate (mu_gpu, cv2, n_slots) from CDF for a pool handled by gpu.

    Samples requests from the CDF slice, computes the raw service time
    (seq-len-aware) for each, and returns:
      mu_gpu  : GPU-level throughput = n_slots / E[service_time]  (req/s per GPU)
      cv2     : coefficient of variation squared of service_time
      n_slots : KV-cache concurrency of this GPU at pool_max context

    The caller should use n_slots to compute the correct Erlang-C server count:
    c_slots = n_gpus * n_slots, with per-slot rate mu_slot = mu_gpu / n_slots.

    lo_clamp : minimum token length to sample (pass gamma*B_short+1 for the
               long pool so short-side lengths are excluded from calibration).
    """
    import random

    rng = random.Random(42)
    n_slots = gpu.n_slots(pool_max)
    raw_samples = []
    prefill_samples = []
    for _ in range(3000):
        u = rng.random()
        prev = 0
        for thresh, frac in cdf:
            if u <= frac:
                lo = max(lo_clamp, prev + 1)
                hi = min(pool_max, thresh)
                length = rng.randint(lo, hi) if lo <= hi else pool_max
                l_in = max(1, int(length * 0.80))
                l_out = max(1, length - l_in)
                s_raw = gpu.service_time(l_in, l_out, pool_max)
                raw_samples.append(s_raw)
                # Prefill: ceil(l_in / chunk) iterations at single-sequence iter_t
                pref = math.ceil(l_in / gpu.chunk) * gpu.iter_latency(1)
                prefill_samples.append(pref)
                break
            prev = thresh
    if not raw_samples:
        return float(n_slots), 1.0, n_slots, 0.0
    n = len(raw_samples)
    e1 = sum(raw_samples) / n
    e2 = sum(s * s for s in raw_samples) / n
    cv2 = max(0.01, e2 / (e1 * e1) - 1.0)
    mu_gpu = n_slots / e1  # GPU-level throughput (n_slots parallel requests)
    mean_prefill_s = (
        sum(prefill_samples) / len(prefill_samples) if prefill_samples else 0.0
    )
    return mu_gpu, cv2, n_slots, mean_prefill_s


def _min_gpus_analytical(
    lam: float,
    mu: float,
    t_slo: float,
    cv2: float = 1.0,
    rho_max: float = 0.85,
    n_slots: int = 1,
) -> int:
    """Minimum GPU count such that P99 wait ≤ t_slo AND utilisation ≤ rho_max.

    Each GPU provides n_slots concurrent KV-cache slots. The Erlang-C model
    uses c_slots = n_gpus * n_slots servers each at rate mu_slot = mu / n_slots.
    This correctly captures KV-slot-level queuing: a request waits only until
    any slot is free, not until an entire GPU is free.

    The utilisation cap (default ρ_max=0.85) guards against the Kimura
    approximation becoming inaccurate near ρ→1.
    """
    if lam <= 0:
        return 1
    mu_slot = mu / n_slots  # per-slot service rate
    a = lam * (1.0 / mu_slot)  # Erlang load in slot-units (= n_slots * lam/mu)
    # Minimum from P99 SLO constraint — iterate over GPU counts
    c_slo = max(1, math.ceil(lam / mu) + 1)
    while _p99_wait(c_slo * n_slots, lam, mu_slot, cv2) > t_slo:
        c_slo += 1
        if c_slo > 5000:
            return c_slo
    # Minimum from utilisation cap constraint (GPU-level utilisation)
    c_rho = math.ceil(lam / (mu * rho_max)) if rho_max < 1.0 else c_slo
    return max(c_slo, c_rho)


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
    candidates = [t for t in cdf_toks[:-1] if 0.01 <= _cdf_eval(cdf, t) <= 0.999]

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
        alpha = _cdf_eval(cdf, b)
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
        alpha_base = _cdf_eval(cdf, self.B_short)

        # Calibrate short pool (requests <= B_short, fixed)
        short_cdf = [(t, f) for t, f in cdf if t <= self.B_short]
        if short_cdf:
            last_f = short_cdf[-1][1]
            short_cdf_norm = [(t, f / last_f) for t, f in short_cdf]
        else:
            short_cdf_norm = [(self.B_short, 1.0)]

        mu_s, cv2_s, ns_s, pref_s = _calibrate(
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
            f_at_gamma_bs = _cdf_eval(cdf, gamma_bs)
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

            n_s_raw = _min_gpus_analytical(lam_s, mu_s, self.t_slo, cv2_s, n_slots=ns_s)
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
                mu_l, cv2_l, ns_l, pref_l = _calibrate(
                    long_cdf_norm,
                    self.long_max_ctx,
                    self.gpu_long,
                    lo_clamp=max(1, gamma_bs),
                )
            else:
                mu_l, cv2_l, ns_l, pref_l = 0.01, 1.0, 1, 0.0

            n_l_raw = (
                _min_gpus_analytical(lam_l, mu_l, self.t_slo, cv2_l, n_slots=ns_l)
                if lam_l > 0.01
                else 0
            )
            n_l = math.ceil(n_l_raw / self.node_avail)

            # P99 TTFT = P99 slot wait + mean prefill time
            p99_s = (_p99_wait(n_s * ns_s, lam_s, mu_s / ns_s, cv2_s) + pref_s) * 1000
            p99_l = (
                (_p99_wait(n_l * ns_l, lam_l, mu_l / ns_l, cv2_l) + pref_l) * 1000
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

        alpha_base = _cdf_eval(cdf, self.B_short)
        gamma_bs = int(gamma * self.B_short)
        f_at_gamma_bs = _cdf_eval(cdf, gamma_bs)
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
        f_at_gbs = _cdf_eval(cdf, gamma_bs)  # interpolated
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

# ── Grid-flexibility analysis ─────────────────────────────────────────────────


@dataclass
class GridFlexPoint:
    """One operating point on the power-vs-latency trade-off curve.

    Generated by :func:`grid_flex_analysis`.  Each row answers:
    "If I cap per-GPU batch size to *n_max_cap* concurrent requests
    (committing *flex_pct*% power reduction to the grid), what P99 TTFT
    does the fleet deliver?"

    Attributes
    ----------
    flex_pct          : power reduction committed to the grid operator (%)
    n_max_cap         : resulting concurrent-request cap per GPU
    power_per_gpu_w   : estimated per-GPU power draw at this cap (W)
    power_fleet_kw    : total fleet power (n_gpus × power_per_gpu_w / 1000)
    p99_ttft_ms       : analytical P99 TTFT estimate at this cap (ms)
    p99_ttft_des_ms   : DES-verified P99 TTFT (ms); None if DES not run
    slo_met           : True if p99 ≤ t_slo_ms (uses DES P99 when available)
    power_model       : "logistic" or "linear" — which GPU power model was used
    """

    flex_pct: float
    n_max_cap: int
    power_per_gpu_w: float
    power_fleet_kw: float
    p99_ttft_ms: float
    slo_met: bool
    p99_ttft_des_ms: float | None = None
    power_model: str = "linear"


def _throttled_profile(gpu, n_max_cap: int, max_ctx: int):
    """Return a copy of *gpu* with KV-slot count forced to *n_max_cap*.

    Used by grid_flex_analysis() DES verification to simulate a fleet with
    a batch-size cap applied (the G2G max_num_seqs curtailment mechanism).
    """
    import dataclasses
    import math

    blks_per_seq = math.ceil(max_ctx / gpu.blk_size)
    # Force both KV-memory limit and compute limit to equal n_max_cap
    new_kv = n_max_cap * blks_per_seq
    # compute_cap = max_slots * calibration_ctx // max_ctx; we want this = n_max_cap
    new_max_slots = math.ceil(n_max_cap * max_ctx / max(1, gpu.calibration_ctx))
    return dataclasses.replace(gpu, max_slots=new_max_slots, total_kv_blks=new_kv)


def grid_flex_analysis(
    cdf: list,
    lam: float,
    n_gpus: int,
    gpu: ManualProfile,
    t_slo_ms: float,
    max_ctx: int = 8192,
    flex_pcts: list[float] | None = None,
    n_sim_requests: int = 0,
    verbose: bool = False,
) -> list[GridFlexPoint]:
    """Compute the power–latency trade-off curve for a fleet under demand response.

    Models the GPU-to-Grid (G2G) batch-size control mechanism proposed by
    Hassan et al. (arXiv:2602.05116v1, 2025): when the grid signals high load,
    the LLM serving engine caps the maximum in-flight batch size (vLLM
    ``max_num_seqs``).  A lower cap reduces GPU power draw at the cost of
    higher queuing latency.

    For each *flex_pct* this function:

    1. Derives ``n_max_cap`` — the batch-size cap that achieves the target
       power reduction — by inverting the GPU power model.  If the profile
       has logistic parameters (``power_logistic_k > 0``) the logistic curve
       is used; otherwise linear interpolation is used.
    2. Estimates P99 TTFT analytically (M/G/c) with ``n_max_cap`` slots.
    3. Optionally runs a DES simulation at ``n_max_cap`` slots to verify the
       analytical estimate (``n_sim_requests > 0``).

    Verification note
    -----------------
    The simulator verifies the *latency* side of the G2G trade-off: capping
    n_max_cap on a fleet of *n_gpus* GPUs at arrival rate *lam* req/s, both
    the analytical and DES paths predict P99 TTFT independently.  Agreement
    between them (typically within 10–15%) confirms the M/G/c model is
    adequate for production sizing.

    The *power* side is verified by comparing the logistic power curve against
    the ML.ENERGY benchmark data points embedded in the profile constants.
    The G2G paper uses ``H100-SXM5`` hardware; the ``H100_80GB`` profile
    logistic parameters (k=1.0, x0=4.2) are fitted to those measurements.

    Sources
    -------
    - Hassan et al. (2025) "GPU-to-Grid: Coupling LLM Inference with Power
      System Control", arXiv:2602.05116v1.
    - ML.ENERGY Benchmark v3.0 (Chung et al., NeurIPS 2025 D&B track),
      https://ml.energy/leaderboard — measured H100-SXM5 power vs. batch size.
    - NVIDIA GPU Spec Sheets: H100-SXM5 TDP 700W; A100-SXM4 TDP 400W.

    Parameters
    ----------
    cdf             : workload CDF as list of (token_threshold, cumulative_frac)
    lam             : total arrival rate (req/s) hitting this fleet
    n_gpus          : fixed fleet size to evaluate (e.g. from a prior optimize() run)
    gpu             : ManualProfile with power_idle_w and power_nominal_w set
    t_slo_ms        : P99 TTFT SLO (ms) — used to flag slo_met
    max_ctx         : maximum context window for this pool (tokens)
    flex_pcts       : list of power-reduction percentages to sweep.
                      Defaults to [0, 5, 10, 15, 20, 25, 30, 40, 50].
    n_sim_requests  : number of requests for DES verification at each flex level.
                      0 (default) disables DES.  10 000–30 000 is sufficient
                      for a stable P99 estimate.  DES runs are independent per
                      flex level, so runtime scales linearly.
    verbose         : print progress when running DES verification.

    Returns
    -------
    List of :class:`GridFlexPoint`, one per flex_pct, sorted ascending.
    Each point includes ``p99_ttft_des_ms`` when DES is enabled.

    Raises
    ------
    ValueError
        If *gpu* does not have power model constants configured.

    Examples
    --------
    >>> from fleet_sim import grid_flex_analysis, H100_80GB
    >>> import json
    >>> cdf = json.load(open("data/azure_cdf.json"))
    >>> # Analytical only (fast)
    >>> curve = grid_flex_analysis(cdf, lam=200, n_gpus=40, gpu=H100_80GB,
    ...                            t_slo_ms=500)
    >>> # With DES verification (slower but verified)
    >>> curve = grid_flex_analysis(cdf, lam=200, n_gpus=40, gpu=H100_80GB,
    ...                            t_slo_ms=500, n_sim_requests=20000)
    >>> for pt in curve:
    ...     des = f"{pt.p99_ttft_des_ms:.1f}ms" if pt.p99_ttft_des_ms else "—"
    ...     print(f"flex={pt.flex_pct:4.0f}%  n_max={pt.n_max_cap:3d}"
    ...           f"  {pt.power_per_gpu_w:.0f}W/GPU  P99={pt.p99_ttft_ms:.1f}ms"
    ...           f"  DES={des}  {'OK' if pt.slo_met else 'BREACH'}")
    """
    if flex_pcts is None:
        flex_pcts = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    # Validate power model is available
    _ = gpu.power_at_concurrency(1)  # raises ValueError if not configured

    power_model_type = "logistic" if gpu.power_logistic_k > 0.0 else "linear"
    n_slots_full = gpu.n_slots(max_ctx)

    results: list[GridFlexPoint] = []
    for flex_pct in flex_pcts:
        # Target power per GPU after flex commitment (clamped to idle floor)
        target_power = max(
            gpu.power_idle_w, gpu.power_nominal_w * (1.0 - flex_pct / 100.0)
        )

        # Invert power model to find n_max_cap, then cap at the actual
        # KV-cache-limited slot count for this max_ctx window.
        n_max_cap = min(n_slots_full, gpu.invert_power(target_power))

        actual_power = gpu.power_at_concurrency(n_max_cap)
        fleet_kw = n_gpus * actual_power / 1000.0

        # Recalibrate service rate at the *throttled* concurrency level.
        # At n_max_cap < n_slots_full, each GPU iteration covers fewer
        # concurrent sequences: iter_t(n_max_cap) = W + H*n_max_cap < iter_t(n_slots_full).
        # Using the full-load mu_slot underestimates throughput and makes the
        # analytical model falsely diverge (inf P99) when n_max_cap is below
        # the full-load Erlang saturation threshold but actually still stable.
        # Using a throttled profile here gives the correct M/G/c calibration
        # for the reduced concurrency level.
        capped_gpu = _throttled_profile(gpu, n_max_cap, max_ctx)
        mu_cap, cv2_cap, _, mean_prefill_cap = _calibrate(cdf, max_ctx, capped_gpu)
        mu_slot_cap = mu_cap / n_max_cap if n_max_cap > 0 else 1e-9

        c_slots = n_gpus * n_max_cap
        p99_s = _p99_wait(c_slots, lam, mu_slot_cap, cv2_cap)
        p99_ttft_ms = (p99_s + mean_prefill_cap) * 1000.0

        # ── Optional DES verification ─────────────────────────────────────
        p99_ttft_des_ms = None
        if n_sim_requests > 0:
            try:
                des_p99 = _run_grid_flex_des(
                    cdf=cdf,
                    lam=lam,
                    n_gpus=n_gpus,
                    gpu=gpu,
                    n_max_cap=n_max_cap,
                    max_ctx=max_ctx,
                    n_req=n_sim_requests,
                )
                p99_ttft_des_ms = des_p99
                if verbose:
                    print(
                        f"  flex={flex_pct:.0f}%  n_max={n_max_cap}"
                        f"  analytical={p99_ttft_ms:.1f}ms"
                        f"  DES={p99_ttft_des_ms:.1f}ms"
                    )
            except Exception as e:
                if verbose:
                    print(f"  [DES error at flex={flex_pct}%]: {e}")

        # slo_met uses DES P99 when available (more accurate)
        effective_p99 = p99_ttft_des_ms if p99_ttft_des_ms is not None else p99_ttft_ms
        results.append(
            GridFlexPoint(
                flex_pct=flex_pct,
                n_max_cap=n_max_cap,
                power_per_gpu_w=actual_power,
                power_fleet_kw=fleet_kw,
                p99_ttft_ms=p99_ttft_ms,
                p99_ttft_des_ms=p99_ttft_des_ms,
                slo_met=effective_p99 <= t_slo_ms,
                power_model=power_model_type,
            )
        )

    return results


def _run_grid_flex_des(
    cdf: list,
    lam: float,
    n_gpus: int,
    gpu,
    n_max_cap: int,
    max_ctx: int,
    n_req: int,
    seed: int = 42,
) -> float:
    """Run a single-pool DES with ``n_max_cap`` KV slots per GPU.

    Returns P99 TTFT in milliseconds.  Raises if the queue is severely
    overloaded (>95% of requests never complete).
    """
    from ..workload.synthetic import CdfWorkload

    # Build a profile with n_max_cap slots (simulates max_num_seqs cap)
    capped = _throttled_profile(gpu, n_max_cap, max_ctx)

    pool = PoolConfig(pool_id="flex", gpu=capped, n_gpus=n_gpus, max_ctx=max_ctx)
    fc = FleetConfig(pools=[pool], router_type="LengthRouter")

    wl = CdfWorkload(cdf, seed=seed)
    arrivals = PoissonWorkload(lam, wl, n_requests=n_req, seed=seed).generate()

    fleet = Fleet(fc)
    result = fleet.run(arrivals, verbose=False)

    # Fleet has a single pool; use the top-level P99 helper
    p99 = result.p99_ttft_ms()
    return p99 if p99 != float("inf") else float("inf")


def print_grid_flex_table(
    results: list[GridFlexPoint],
    t_slo_ms: float,
    n_gpus: int,
    lam: float,
) -> None:
    """Pretty-print the grid flex analysis results."""
    baseline = results[0] if results else None
    has_des = any(pt.p99_ttft_des_ms is not None for pt in results)
    power_model = results[0].power_model if results else "linear"

    print(f"\n{'='*70}")
    print(f"  Grid Flexibility Analysis  [{power_model} power model]")
    print(f"  Fleet: {n_gpus} GPUs  λ={lam:.0f} req/s  SLO={t_slo_ms:.0f} ms")
    if baseline:
        print(
            f"  Baseline: {baseline.power_fleet_kw:.1f} kW fleet power"
            f"  ({baseline.power_per_gpu_w:.0f} W/GPU)"
        )
    print(f"{'='*70}")

    if has_des:
        print(
            f"  {'Flex':>6} {'n_max':>6} {'W/GPU':>7} {'Fleet kW':>9}"
            f" {'P99 analyt':>11} {'P99 DES':>9} {'SLO':>5}"
        )
        print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*11} {'-'*9} {'-'*5}")
        for pt in results:
            ok = "  OK" if pt.slo_met else "BREACH"
            des_str = (
                f"{pt.p99_ttft_des_ms:>8.1f}ms"
                if pt.p99_ttft_des_ms is not None
                else "        —"
            )
            print(
                f"  {pt.flex_pct:>5.0f}% {pt.n_max_cap:>6d}"
                f" {pt.power_per_gpu_w:>6.0f}W {pt.power_fleet_kw:>8.1f}kW"
                f" {pt.p99_ttft_ms:>10.1f}ms {des_str} {ok}"
            )
    else:
        print(
            f"  {'Flex':>6} {'n_max':>6} {'W/GPU':>7} {'Fleet kW':>9}"
            f" {'P99 TTFT':>9} {'SLO':>5}"
        )
        print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*9} {'-'*5}")
        for pt in results:
            ok = "  OK" if pt.slo_met else "BREACH"
            print(
                f"  {pt.flex_pct:>5.0f}% {pt.n_max_cap:>6d}"
                f" {pt.power_per_gpu_w:>6.0f}W {pt.power_fleet_kw:>8.1f}kW"
                f" {pt.p99_ttft_ms:>8.1f}ms {ok}"
            )

    # Find max safe flex depth
    safe = [pt for pt in results if pt.slo_met]
    if safe:
        max_safe = max(pt.flex_pct for pt in safe)
        max_safe_pt = max(safe, key=lambda p: p.flex_pct)
        baseline_kw = baseline.power_fleet_kw if baseline else 0.0
        saved_kw = baseline_kw - max_safe_pt.power_fleet_kw
        p99_disp = (
            f"{max_safe_pt.p99_ttft_des_ms:.1f}ms (DES)"
            if max_safe_pt.p99_ttft_des_ms is not None
            else f"{max_safe_pt.p99_ttft_ms:.1f}ms (analytical)"
        )
        print(
            f"\n  Max safe flex depth: {max_safe:.0f}%"
            f"  (saves {saved_kw:.1f} kW fleet-wide,"
            f" P99={p99_disp})"
        )
    else:
        print("\n  No flex depth meets SLO — consider adding GPUs first.")


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


# ── Tokens-per-Watt analysis ──────────────────────────────────────────────────


@dataclass
class TpwPoint:
    """Energy-efficiency snapshot for one pool at one utilisation level.

    Generated by :func:`tpw_analysis` (single-pool) or per-pool entries
    inside :class:`FleetTpwResult` (multi-pool).

    **Model dependency warning**: W, H, and the power logistic curve all
    depend on the served model, not just the GPU.  Pre-built profiles
    (H100_80GB, A100_80GB, A10G) are calibrated for Llama-3-70B on 8-GPU TP
    (H100/A100) or 7B-class on single-GPU (A10G).  Do not compare ``tok/W``
    across profiles that represent different models — the ratio will reflect
    model size, not GPU efficiency.

    **Single-pool N_gpus cancellation**: For a *single homogeneous pool*,
    ``tokens_per_watt = mu_gpu × ρ × mean_L_out / P(n_active)`` is
    independent of fleet size (N numerator and denominator cancel).

    **Multi-pool fleets**: N does NOT cancel across pools.  Use
    :class:`FleetTpwResult` which computes the correct aggregate:
    ``fleet_tpw = Σ(λ_i × L_out_i) / Σ(N_i × P_i(n_i))``.

    Attributes
    ----------
    gpu_name          : profile label (e.g. "H100-80GB")
    pool_label        : optional human label for the pool (e.g. "short-7B")
    n_gpus            : fleet size at this operating point
    rho               : GPU-level utilisation  λ / (N × mu_gpu)
    n_active          : mean concurrent in-flight requests per GPU
    power_per_gpu_w   : per-GPU power (W) at n_active; from logistic model
    tokens_per_watt   : output tok/J for THIS pool in isolation
                        = (λ_pool × mean_L_out) / (N_gpus × power_per_gpu_w)
    cost_per_mil      : $/1M output tokens for this pool's λ_pool at N_gpus
    p99_ttft_ms       : analytical P99 TTFT (ms) for this pool
    power_model_qual  : "HIGH" / "FAIR" / "LOW" — confidence in P(n_active)
    slo_optimal       : True when N_gpus is the SLO-optimal (minimum) fleet
    """

    gpu_name: str
    n_gpus: int
    rho: float
    n_active: float
    power_per_gpu_w: float
    tokens_per_watt: float
    cost_per_mil: float
    p99_ttft_ms: float
    power_model_qual: str
    pool_label: str = ""
    slo_optimal: bool = False


@dataclass
class FleetTpwResult:
    """Fleet-level tokens-per-watt result for a multi-pool routing topology.

    Generated by :func:`fleet_tpw_analysis`.  Holds per-pool breakdowns and
    the correct fleet-level aggregate.

    Fleet-level formula (N does NOT cancel across pools):

        fleet_tpw = Σ_i(λ_i × mean_L_out_i) / Σ_i(N_i × P_i(n_i))

    This is the only correct formula for comparing homo-pool vs two-pool
    routing, or for semantic-router configurations where different pools
    serve different model sizes.

    Attributes
    ----------
    topology          : human-readable description (e.g. "homo H100-70B",
                        "two-pool A10G-7B + H100-70B")
    pools             : per-pool TpwPoint list (one entry per pool)
    fleet_tpw         : fleet-level tok/W using the formula above
    fleet_cost_per_mil: fleet-level $/1M output tokens
    fleet_power_kw    : total fleet power draw (kW) at this operating point
    total_gpus        : total GPU count across all pools
    worst_p99_ms      : max(p99_ttft per pool)
    power_model_notes : list of caveat strings for LOW-quality power models
    """

    topology: str
    pools: list[TpwPoint]
    fleet_tpw: float
    fleet_cost_per_mil: float
    fleet_power_kw: float
    total_gpus: int
    worst_p99_ms: float
    power_model_notes: list[str]


def _power_model_quality(gpu) -> str:
    """Return 'HIGH' / 'FAIR' / 'LOW' from the profile's own docstring labels."""
    if gpu.power_logistic_k <= 0.0 or gpu.power_idle_w <= 0.0:
        return "NONE"
    name = getattr(gpu, "name", "").upper()
    if "H100" in name or "H200" in name:
        return "HIGH"
    if "A100" in name:
        return "FAIR"
    return "LOW"


def _cdf_mean_l_out(cdf: list) -> int:
    """Return mean output-token count from a (possibly sub-)CDF.

    Assumes the 80/20 input-to-output split used throughout the simulator.
    Works on a normalised sub-CDF (last cumulative fraction == 1.0) or on
    the raw full CDF (last fraction may equal anything ≤ 1.0).
    """
    total_prob = 0.0
    mean_total = 0.0
    prev_t, prev_f = 0, 0.0
    for thresh, frac in cdf:
        width = thresh - prev_t
        prob = frac - prev_f
        mean_total += (prev_t + width / 2.0) * prob
        total_prob += prob
        prev_t, prev_f = thresh, frac
    if total_prob > 0:
        mean_total /= total_prob
    return max(1, int(0.20 * mean_total))


def _split_cdf(full_cdf: list, b_short: int) -> tuple:
    """Split a full CDF into (short_sub_cdf, long_sub_cdf, alpha).

    Parameters
    ----------
    full_cdf : list of (token_threshold, cumulative_frac).
    b_short  : token-count threshold.  Requests with total_tokens ≤ b_short
               go to the short pool; the rest go to the long pool.

    Returns
    -------
    short_cdf : normalised sub-CDF for the short pool (fractions rescaled so
                the last entry is 1.0).
    long_cdf  : normalised sub-CDF for the long pool.
    alpha     : fraction of traffic assigned to the short pool.

    Notes
    -----
    If b_short falls between two CDF knots, linear interpolation is used to
    find alpha accurately.
    """
    # Linear interpolation to find alpha = CDF(b_short)
    alpha = 0.0
    for i, (t, f) in enumerate(full_cdf):
        if t >= b_short:
            if i == 0:
                alpha = f * (b_short / max(1, t))
            else:
                t0, f0 = full_cdf[i - 1]
                if t == t0:
                    alpha = f
                else:
                    alpha = f0 + (f - f0) * (b_short - t0) / (t - t0)
            break
    else:
        alpha = full_cdf[-1][1]  # b_short > max threshold

    alpha = max(1e-6, min(1.0 - 1e-6, alpha))

    short_cdf: list = []
    long_cdf: list = []
    for thresh, frac in full_cdf:
        if thresh <= b_short:
            short_cdf.append((thresh, frac / alpha))
        else:
            norm_f = max(0.0, (frac - alpha) / (1.0 - alpha))
            long_cdf.append((thresh, norm_f))

    # Ensure the short CDF endpoint is exactly b_short at frac=1.0
    if not short_cdf or short_cdf[-1][0] < b_short:
        short_cdf.append((b_short, 1.0))
    else:
        short_cdf[-1] = (short_cdf[-1][0], 1.0)

    if long_cdf:
        long_cdf[-1] = (long_cdf[-1][0], 1.0)
    else:
        # Degenerate: all traffic is short
        long_cdf = [(b_short + 1, 1.0)]

    return short_cdf, long_cdf, alpha


def _tpw_one_pool(
    cdf: list,
    lam: float,
    gpu,
    t_slo_ms: float,
    max_ctx: int,
    pool_label: str = "",
    rho_override: float | None = None,
) -> TpwPoint:
    """Compute a single TpwPoint for one pool.

    Parameters
    ----------
    cdf           : (possibly sub-)CDF for this pool's traffic.
    lam           : arrival rate into THIS pool (req/s).
    gpu           : GPU profile for this pool.
    t_slo_ms      : P99 TTFT SLO (ms).
    max_ctx       : context window (tokens).
    pool_label    : human label (e.g. "short-7B").
    rho_override  : if set, force this utilisation instead of SLO-optimal.
    """
    mu_gpu, cv2, n_slots, mean_pref = _calibrate(cdf, max_ctx, gpu)
    t_slo_s = t_slo_ms / 1000.0
    qual = _power_model_quality(gpu)
    mean_l_out = _cdf_mean_l_out(cdf)

    if rho_override is None:
        n_gpus = _min_gpus_analytical(lam, mu_gpu, t_slo_s, cv2, n_slots=n_slots)
        slo_optimal = True
    else:
        rho_c = max(0.05, min(0.95, rho_override))
        n_gpus = max(1, int(math.ceil(lam / (mu_gpu * rho_c))))
        slo_optimal = False

    rho = lam / (n_gpus * mu_gpu)
    n_active = rho * n_slots
    p99_ms = (
        _p99_wait(n_gpus * n_slots, lam, mu_gpu / n_slots, cv2) + mean_pref
    ) * 1000.0

    try:
        pw = gpu.power_at_concurrency(max(1, int(n_active)))
    except ValueError:
        pw = float("nan")

    # per-pool tok/W (N cancels within the pool)
    tpw = mu_gpu * rho * mean_l_out / pw if (pw and pw > 0) else float("nan")
    cost = (n_gpus * gpu.cost_per_hr / 3600.0) / (lam * mean_l_out) * 1e6

    return TpwPoint(
        gpu_name=gpu.name,
        pool_label=pool_label or gpu.name,
        n_gpus=n_gpus,
        rho=rho,
        n_active=n_active,
        power_per_gpu_w=pw,
        tokens_per_watt=tpw,
        cost_per_mil=cost,
        p99_ttft_ms=p99_ms,
        power_model_qual=qual,
        slo_optimal=slo_optimal,
    )


def tpw_analysis(
    cdf: list,
    lam: float,
    gpus: list,
    t_slo_ms: float = 500.0,
    max_ctx: int = 8192,
    rho_sweep: list[float] | None = None,
) -> list[TpwPoint]:
    """Compute tokens-per-watt for a list of GPU profiles (single-pool each).

    All GPUs receive the **same** full CDF and arrival rate, so this function
    is only correct when comparing GPUs that serve the **same model** on the
    **same workload** — i.e. you are only changing the hardware, not the
    model.

    For multi-pool routing (homo vs hetero, or semantic-router small+large
    model), use :func:`fleet_tpw_analysis` instead, which computes the
    correct fleet-level aggregate across pools.

    .. warning::

        Pre-built profiles are calibrated for Llama-3-70B on 8-GPU TP (H100,
        A100) or 7B-class on single-GPU (A10G).  Comparing H100 (70B) vs
        A10G (7B) with this function mixes two different served models; the
        result reflects model-size difference, not GPU efficiency.  Set the
        same model by creating :class:`ManualProfile` instances manually
        before calling this function.

    Parameters
    ----------
    cdf       : workload CDF as list of (token_threshold, cumulative_frac).
    lam       : arrival rate (req/s).
    gpus      : GPU profiles to compare (list of ManualProfile).
    t_slo_ms  : P99 TTFT SLO (ms).
    max_ctx   : context window (tokens).
    rho_sweep : optional list of utilisation fractions at which to also
                compute tokens/watt (in addition to the SLO-optimal point).

    Returns
    -------
    List of TpwPoint.  SLO-optimal points have ``slo_optimal=True``.

    Notes
    -----
    **Do not over-interpret absolute values.** The tokens/watt numbers depend
    on power model quality (see ``power_model_qual`` field):
      - H100: HIGH (fitted to ML.ENERGY v3.0 measured data).
      - A100: FAIR (one measured anchor + FLOPS-scaling projection).
      - A10G: LOW  (projection only; no published batch-vs-power data).
    """
    points: list[TpwPoint] = []

    for gpu in gpus:
        pt = _tpw_one_pool(cdf, lam, gpu, t_slo_ms, max_ctx)
        points.append(pt)

        if rho_sweep:
            for rho in rho_sweep:
                pt_s = _tpw_one_pool(cdf, lam, gpu, t_slo_ms, max_ctx, rho_override=rho)
                points.append(pt_s)

    return points


def fleet_tpw_analysis(
    pools: list,
    lam_total: float,
    t_slo_ms: float = 500.0,
    topology: str = "",
) -> FleetTpwResult:
    """Compute fleet-level tokens-per-watt across a multi-pool routing topology.

    This is the correct function for comparing:

    * **Homo pool**   — all traffic to one GPU type / one model.
    * **Hetero pool** — short requests routed to small-GPU/small-model pool,
                        long requests to large-GPU/large-model pool.
    * **Semantic router** — fraction α of requests to a fast/cheap pool
                            (e.g. A10G + 7B model), rest to a capable pool
                            (e.g. H100 + 70B).

    Unlike :func:`tpw_analysis`, this function correctly accounts for the
    fact that N does **not** cancel across pools:

        fleet_tpw = Σ_i(λ_i × mean_L_out_i) / Σ_i(N_i × P_i(n_i))

    Parameters
    ----------
    pools : list of dicts, one per pool.  Each dict must contain:

        ``gpu``        : ManualProfile for this pool.
        ``cdf``        : (sub-)CDF for this pool's traffic.  For a b_short
                         split use :func:`_split_cdf` to obtain per-pool
                         sub-CDFs with correctly normalised fractions.
        ``lam``        : arrival rate into this pool (req/s).
        ``max_ctx``    : context window for this pool (tokens).
        ``label``      : human-readable label (e.g. "short-7B", "long-70B").

    lam_total : total arrival rate (req/s); used only for output annotation.
    t_slo_ms  : P99 TTFT SLO (ms) — applied to every pool independently.
    topology  : optional label for the routing topology.

    Returns
    -------
    :class:`FleetTpwResult` with per-pool :class:`TpwPoint` breakdowns and
    the fleet-level aggregate.

    Notes
    -----
    The CDF for each pool must be normalised so its last cumulative fraction
    is 1.0.  Use :func:`_split_cdf` to produce correctly normalised sub-CDFs.

    Power model accuracy caveats propagate: if any pool uses a LOW-quality
    power model, the fleet-level tok/W estimate inherits that uncertainty.
    """
    pool_points: list[TpwPoint] = []
    for spec in pools:
        pt = _tpw_one_pool(
            cdf=spec["cdf"],
            lam=spec["lam"],
            gpu=spec["gpu"],
            t_slo_ms=t_slo_ms,
            max_ctx=spec.get("max_ctx", 8192),
            pool_label=spec.get("label", spec["gpu"].name),
        )
        pool_points.append(pt)

    # Fleet-level aggregation: N does NOT cancel across pools
    total_output_tps = 0.0
    total_fleet_power_w = 0.0
    total_cost_per_s = 0.0
    total_gpus = 0
    worst_p99 = 0.0
    notes: list[str] = []

    for spec, pt in zip(pools, pool_points):
        mean_l_out = _cdf_mean_l_out(spec["cdf"])
        output_tps_pool = spec["lam"] * mean_l_out  # output tok/s from this pool
        pool_power_w = pt.n_gpus * pt.power_per_gpu_w  # W for whole pool

        total_output_tps += output_tps_pool
        total_fleet_power_w += pool_power_w
        total_cost_per_s += pt.n_gpus * spec["gpu"].cost_per_hr / 3600.0
        total_gpus += pt.n_gpus
        worst_p99 = max(worst_p99, pt.p99_ttft_ms)

        if pt.power_model_qual == "LOW":
            notes.append(
                f"  Pool '{pt.pool_label}' ({pt.gpu_name}): LOW power-model quality "
                f"— projection only, no published batch-vs-power measurements."
            )

    fleet_tpw = (
        total_output_tps / total_fleet_power_w
        if total_fleet_power_w > 0
        else float("nan")
    )
    fleet_cost = (total_cost_per_s / (lam_total * 1.0)) * 1e6
    # cost per 1M output tokens: total $/s / total output tok/s × 1e6
    fleet_cost_per_mil = (
        total_cost_per_s / total_output_tps * 1e6
        if total_output_tps > 0
        else float("nan")
    )

    if not topology:
        topology = " + ".join(p.pool_label for p in pool_points)

    return FleetTpwResult(
        topology=topology,
        pools=pool_points,
        fleet_tpw=fleet_tpw,
        fleet_cost_per_mil=fleet_cost_per_mil,
        fleet_power_kw=total_fleet_power_w / 1000.0,
        total_gpus=total_gpus,
        worst_p99_ms=worst_p99,
        power_model_notes=notes,
    )


def print_tpw_table(points: list[TpwPoint], title: str = "") -> None:
    """Print a formatted tokens-per-watt comparison table (single-pool entries)."""
    if title:
        print(f"\n{title}")
    hdr = (
        f"  {'Pool / GPU':22s}  {'GPUs':>5}  {'ρ':>5}  {'n_act':>6}  "
        f"{'Tok/W':>6}  {'P/GPU (W)':>9}  {'$/1M tok':>9}  "
        f"{'P99 (ms)':>9}  {'PwrQ':>4}  {'SLO':>3}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)
    for p in points:
        if not p.slo_optimal:
            continue
        label = p.pool_label or p.gpu_name
        flag = "★" if p.slo_optimal else " "
        qual_tag = p.power_model_qual
        if p.power_model_qual == "LOW":
            qual_tag += "*"
        print(
            f"  {label:22s}  {p.n_gpus:>5}  {p.rho:>5.2f}  "
            f"{p.n_active:>6.1f}  {p.tokens_per_watt:>6.2f}  "
            f"{p.power_per_gpu_w:>9.1f}  ${p.cost_per_mil:>8.2f}  "
            f"{p.p99_ttft_ms:>9.1f}  {qual_tag:>4}  {flag:>3}"
        )
    print(sep)
    if any(p.power_model_qual == "LOW" for p in points if p.slo_optimal):
        print(
            "  * LOW quality power model: projection only, no published measurements."
        )
        print("    Tok/W estimate for this GPU has high uncertainty.")
    print()


def print_fleet_tpw(result: FleetTpwResult, title: str = "") -> None:
    """Print a formatted fleet-level tokens-per-watt comparison."""
    if title:
        print(f"\n{title}")
    print(f"\n  Topology : {result.topology}")
    print(f"  Total GPUs     : {result.total_gpus}")
    print(f"  Fleet power    : {result.fleet_power_kw:.1f} kW")
    print(f"  Worst P99 TTFT : {result.worst_p99_ms:.1f} ms")
    print(
        f"  Fleet tok/W    : {result.fleet_tpw:.3f}  tok/J  ← correct multi-pool aggregate"
    )
    print(f"  Fleet $/1M tok : ${result.fleet_cost_per_mil:.2f}")

    hdr = (
        f"\n  {'Pool':22s}  {'GPU':18s}  {'GPUs':>5}  {'ρ':>5}  "
        f"{'Tok/W(pool)':>12}  {'P99(ms)':>9}  {'PwrQ':>4}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for p in result.pools:
        label = p.pool_label or p.gpu_name
        qual_tag = p.power_model_qual + ("*" if p.power_model_qual == "LOW" else "")
        print(
            f"  {label:22s}  {p.gpu_name:18s}  {p.n_gpus:>5}  {p.rho:>5.2f}  "
            f"{p.tokens_per_watt:>12.3f}  {p.p99_ttft_ms:>9.1f}  {qual_tag:>4}"
        )
    print()
    if result.power_model_notes:
        print("  Power model caveats:")
        for note in result.power_model_notes:
            print(note)
        print()
