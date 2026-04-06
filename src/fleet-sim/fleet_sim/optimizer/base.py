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
import random
from dataclasses import dataclass

from ..gpu_profiles.profiles import A100_80GB, GpuProfile

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


from .reporting import OptimizationReport


def _normalized_short_cdf(cdf: list, b_short: int) -> list:
    short_cdf = [(t, f) for t, f in cdf if t <= b_short]
    if not short_cdf:
        return [(b_short, 1.0)]
    return [(t, f / short_cdf[-1][1]) for t, f in short_cdf]


def _normalized_long_cdf(cdf: list, gamma_bs: int, long_max_ctx: int) -> list:
    f_at_gamma_bs = _cdf_eval(cdf, gamma_bs)
    long_frac = 1.0 - f_at_gamma_bs
    if long_frac <= 1e-6:
        return [(long_max_ctx, 1.0)]
    long_cdf = [
        (t, max(0.0, min(1.0, (f - f_at_gamma_bs) / long_frac)))
        for t, f in cdf
        if t > gamma_bs
    ]
    if not long_cdf:
        return [(long_max_ctx, 1.0)]
    long_cdf[-1] = (long_cdf[-1][0], 1.0)
    return long_cdf


def _service_time_with_prefill(
    gpu: GpuProfile, pool_max: int, total: int
) -> tuple[float, float]:
    l_in = max(1, int(total * 0.8))
    l_out = max(1, total - l_in)
    service_time = gpu.service_time(l_in, l_out, pool_max)
    prefill = math.ceil(l_in / gpu.chunk) * gpu.iter_latency(1)
    return service_time, prefill


def _sample_cdf_service_time(
    rng: random.Random,
    cdf: list,
    gpu: GpuProfile,
    pool_max: int,
    *,
    start_prev: int = 0,
    inclusive_low: bool = True,
):
    def sample() -> tuple[float, float]:
        u = rng.random()
        prev = start_prev
        for threshold, frac in cdf:
            if u <= frac:
                lo = max(1, prev if inclusive_low else prev + 1)
                return _service_time_with_prefill(
                    gpu, pool_max, rng.randint(lo, max(lo, threshold))
                )
            prev = threshold
        return _service_time_with_prefill(gpu, pool_max, cdf[-1][0])

    return sample


def _simulate_slot_queue(
    rng: random.Random,
    sample_service_time,
    n_gpu: int,
    n_slots: int,
    lam_pool: float,
    n_req: int,
) -> float:
    import heapq

    total_slots = max(1, n_gpu * n_slots)
    servers = [0.0] * total_slots
    heapq.heapify(servers)
    warm = n_req // 5
    ttfts = []
    current = 0.0
    for index in range(n_req + warm):
        current += rng.expovariate(max(lam_pool, 1e-9))
        service_time, prefill = sample_service_time()
        free_at = heapq.heappop(servers)
        slot_wait = max(0.0, free_at - current)
        heapq.heappush(servers, max(free_at, current) + service_time)
        if index >= warm:
            ttfts.append(slot_wait + prefill)
    ttfts.sort()
    idx = max(0, int(len(ttfts) * 0.99) - 1)
    return ttfts[idx] * 1000 if ttfts else 0.0


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
        gamma_values = gammas or [round(1.0 + 0.1 * k, 1) for k in range(11)]
        alpha_base = _cdf_eval(cdf, self.B_short)
        mu_s, cv2_s, ns_s, pref_s = self._short_pool_calibration(cdf)
        if verbose:
            self._print_sweep_header(lam, alpha_base)

        results = []
        for gamma in gamma_values:
            result, alpha_prime = self._analytical_candidate(
                cdf, lam, gamma, alpha_base, mu_s, cv2_s, ns_s, pref_s
            )
            results.append(result)
            if verbose:
                self._print_sweep_row(result, alpha_prime)
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
        if verbose:
            print("[1/2] Analytical sweep...")
        candidates = self.sweep_analytical(cdf, lam, gammas, verbose=verbose)

        verified: list[SweepResult] = []
        if verify_top_n > 0 and n_sim_requests > 0:
            if verbose:
                print(f"\n[2/2] DES verification of top-{verify_top_n} candidates...")
            for sr in candidates[:verify_top_n]:
                if verbose:
                    print(f"  Verifying γ={sr.gamma} (n_s={sr.n_s}, n_l={sr.n_l})...")
                sim_result = self._run_des(
                    cdf, lam, sr.gamma, sr.n_s, sr.n_l, n_sim_requests
                )
                verified.append(
                    SweepResult(
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
                    ),
                )
                if verbose:
                    sr_v = verified[-1]
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
        _alpha_base, gamma_bs, _alpha_prime, lam_s, lam_l = self._gamma_split(
            cdf, lam, gamma
        )
        rng = random.Random(42)
        p99_s = _simulate_slot_queue(
            rng,
            _sample_cdf_service_time(
                rng,
                _normalized_short_cdf(cdf, self.B_short),
                self.gpu_short,
                self.B_short,
            ),
            n_s,
            self.gpu_short.n_slots(self.B_short),
            lam_s,
            n_req,
        )
        p99_l = 0.0
        if lam_l > 0.1 and n_l > 0:
            p99_l = _simulate_slot_queue(
                rng,
                _sample_cdf_service_time(
                    rng,
                    _normalized_long_cdf(cdf, gamma_bs, self.long_max_ctx),
                    self.gpu_long,
                    self.long_max_ctx,
                    start_prev=gamma_bs,
                    inclusive_low=False,
                ),
                n_l,
                self.gpu_long.n_slots(self.long_max_ctx),
                lam_l,
                n_req,
            )
        return {"p99_short_ms": p99_s, "p99_long_ms": p99_l}

    def _short_pool_calibration(self, cdf: list) -> tuple[float, float, int, float]:
        return _calibrate(
            _normalized_short_cdf(cdf, self.B_short),
            self.B_short,
            self.gpu_short,
        )

    def _analytical_candidate(
        self,
        cdf: list,
        lam: float,
        gamma: float,
        alpha_base: float,
        mu_s: float,
        cv2_s: float,
        ns_s: int,
        pref_s: float,
    ) -> tuple[SweepResult, float]:
        _, gamma_bs, alpha_prime, lam_s, lam_l = self._gamma_split(
            cdf, lam, gamma, alpha_base
        )
        n_s = math.ceil(
            _min_gpus_analytical(lam_s, mu_s, self.t_slo, cv2_s, n_slots=ns_s)
            / self.node_avail
        )
        mu_l, cv2_l, ns_l, pref_l = self._long_pool_calibration(cdf, gamma_bs)
        n_l = math.ceil(
            (
                _min_gpus_analytical(lam_l, mu_l, self.t_slo, cv2_l, n_slots=ns_l)
                if lam_l > 0.01
                else 0
            )
            / self.node_avail
        )
        p99_s = (_p99_wait(n_s * ns_s, lam_s, mu_s / ns_s, cv2_s) + pref_s) * 1000
        p99_l = (
            (_p99_wait(n_l * ns_l, lam_l, mu_l / ns_l, cv2_l) + pref_l) * 1000
            if lam_l > 0.01
            else 0.0
        )
        cost_per_hr = n_s * self.gpu_short.cost_per_hr + n_l * self.gpu_long.cost_per_hr
        result = SweepResult(
            gamma=gamma,
            n_s=n_s,
            n_l=n_l,
            total_gpus=n_s + n_l,
            cost_per_hr=cost_per_hr,
            annualised_cost_kusd=cost_per_hr * 8760 / 1000,
            p99_ttft_short_ms=p99_s,
            p99_ttft_long_ms=p99_l,
            slo_met=(p99_s <= self.t_slo_ms and p99_l <= self.t_slo_ms),
            source="analytical",
        )
        return result, alpha_prime

    def _long_pool_calibration(
        self, cdf: list, gamma_bs: int
    ) -> tuple[float, float, int, float]:
        long_cdf = _normalized_long_cdf(cdf, gamma_bs, self.long_max_ctx)
        if (
            long_cdf == [(self.long_max_ctx, 1.0)]
            and _cdf_eval(cdf, gamma_bs) >= 1.0 - 1e-6
        ):
            return 0.01, 1.0, 1, 0.0
        return _calibrate(
            long_cdf,
            self.long_max_ctx,
            self.gpu_long,
            lo_clamp=max(1, gamma_bs),
        )

    def _gamma_split(
        self, cdf: list, lam: float, gamma: float, alpha_base: float | None = None
    ) -> tuple[float, int, float, float, float]:
        alpha_base = (
            alpha_base if alpha_base is not None else _cdf_eval(cdf, self.B_short)
        )
        gamma_bs = int(gamma * self.B_short)
        f_at_gamma_bs = _cdf_eval(cdf, gamma_bs)
        borderline_frac = max(0.0, f_at_gamma_bs - alpha_base)
        alpha_prime = min(
            1.0, alpha_base + borderline_frac * getattr(self, "p_c", 0.75)
        )
        lam_s = alpha_prime * lam
        lam_l = (1.0 - alpha_prime) * lam
        return alpha_base, gamma_bs, alpha_prime, lam_s, lam_l

    def _print_sweep_header(self, lam: float, alpha_base: float) -> None:
        print(f"\n  Analytical sweep  (λ={lam:.0f} req/s, SLO={self.t_slo_ms:.0f}ms)")
        print(f"  B_short={self.B_short:,}  alpha_base={alpha_base:.4f}")
        print(
            f"  {'gamma':>5} {'alpha_p':>8} {'n_s':>5} {'n_l':>5} {'total':>7}"
            f" {'$/yr':>10} {'P99_s':>8} {'P99_l':>8} {'OK':>4}"
        )
        print(f"  {'-' * 68}")

    def _print_sweep_row(self, result: SweepResult, alpha_prime: float) -> None:
        ok = "✓" if result.slo_met else "✗"
        print(
            f"  {result.gamma:>5.1f} {alpha_prime:>8.4f} {result.n_s:>5} {result.n_l:>5}"
            f" {result.total_gpus:>7} ${result.annualised_cost_kusd:>8.1f}K"
            f" {result.p99_ttft_short_ms:>7.1f}ms"
            f" {result.p99_ttft_long_ms:>7.1f}ms {ok:>4}"
        )
