"""ManualProfile — GPU profile with directly specified constants.

Use this when you have measured or estimated W, H, and KV-cache budget
for a specific GPU + model combination and want to supply them directly
rather than computing them from hardware/model specs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ManualProfile:
    """GPU performance profile with directly specified constants.

    Parameters
    ----------
    name             : human-readable label
    W                : base iteration latency (s) — memory/compute constant
    H                : per-active-sequence overhead (s/seq), calibrated at
                       ``calibration_ctx`` tokens.  Attention cost scales as
                       O(seq_len), so H_eff = H * (mean_seq_len / calibration_ctx).
    calibration_ctx  : context length (tokens) at which H was measured.
                       Defaults to 8192.
    chunk            : prefill tokens processed per iteration per sequence
    blk_size         : PagedAttention KV-cache block size (tokens)
    total_kv_blks    : total KV-cache blocks available per GPU
    max_slots        : concurrent-sequence saturation limit, calibrated at
                       ``calibration_ctx`` tokens.  The GPU can run
                       ``max_slots × calibration_ctx / max_ctx`` sequences at
                       shorter context lengths (same memory-bandwidth pressure).
    cost_per_hr      : on-demand $/GPU-hr
    """
    name: str
    W: float
    H: float
    chunk: int
    blk_size: int
    total_kv_blks: int
    max_slots: int
    cost_per_hr: float
    calibration_ctx: int = 8192
    # ── Power model (for grid flexibility analysis) ────────────────────────
    # Source: ML.ENERGY Benchmark v3.0 (Chung et al., NeurIPS 2025,
    # https://ml.energy/leaderboard) measured on H100-SXM5 + A100-SXM + A10G
    # running vLLM.  At inference-active states power follows a logistic
    # curve vs. log2(batch_size); these two scalars capture the operating
    # range relevant for demand-response analysis.
    #
    # power_idle_w    : GPU power at minimum batch (≈ 1 request in-flight),
    #                   from ML.ENERGY "energy / token" × 1-req throughput.
    # power_nominal_w : GPU power at the default max_slots concurrency
    #                   (full production load); approximately 80–88% of TDP
    #                   for SXM-class GPUs.  Use 0.0 to disable power
    #                   modelling (grid_flex_analysis will raise ValueError).
    power_idle_w: float = 0.0
    power_nominal_w: float = 0.0
    # ── Logistic power curve (optional, more accurate than linear) ──────────
    # From the GPU-to-Grid paper (Hassan et al. arXiv:2602.05116v1), Eq. 2:
    #   P(b) = P_range / (1 + exp(-k_p * (log2(b) - x0))) + P_0
    # where b = batch_size (≈ n_active concurrent requests).
    #
    # When power_logistic_k > 0, power_at_concurrency() uses this logistic
    # model instead of linear interpolation.  The logistic is more accurate
    # in the low-batch regime (sub-linear power rise) and captures saturation
    # at high batch.  For 40–100% load the two models agree within ±5%.
    #
    # Fit parameters for H100-SXM5 running vLLM (Llama-3.1-class models):
    #   k=1.0, x0=4.2  — derived from ML.ENERGY Benchmark v3.0 data points
    #   at b ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256} on H100-SXM5 SXM5.
    #   Source: https://ml.energy/leaderboard (NeurIPS 2025 D&B track).
    #
    # Set power_logistic_k = 0 (default) to use the linear model.
    power_logistic_k: float = 0.0    # steepness parameter k_p  (0 = linear)
    power_logistic_x0: float = 0.0   # log2-batch midpoint x_0

    def power_at_concurrency(self, n_active: int) -> float:
        """Estimated GPU power draw (W) at *n_active* concurrent requests.

        Two model variants, selected by whether ``power_logistic_k > 0``:

        **Logistic model** (recommended, when ``power_logistic_k`` is set):
        Implements the GPU-to-Grid power curve (Hassan et al. 2025, Eq. 2)::

            P(b) = P_range / (1 + exp(-k * (log2(b) - x0))) + P_idle

        where *b* ≈ ``n_active``, ``P_range = power_nominal_w - power_idle_w``,
        ``k = power_logistic_k``, ``x0 = power_logistic_x0``.

        **Linear model** (fallback, when ``power_logistic_k == 0``):
        Linearly interpolates between ``power_idle_w`` and ``power_nominal_w``.
        Accurate to within ±5% in the 40–100% load range; under-estimates
        power reduction at deep curtailment (low batch) levels.

        Raises ``ValueError`` if power model is not configured (both
        ``power_idle_w`` and ``power_nominal_w`` must be non-zero).
        """
        if self.power_idle_w <= 0.0 or self.power_nominal_w <= 0.0:
            raise ValueError(
                f"Power model not configured for profile '{self.name}'. "
                "Set power_idle_w and power_nominal_w."
            )
        if self.max_slots <= 0 or n_active <= 0:
            return self.power_idle_w
        p_range = self.power_nominal_w - self.power_idle_w
        if self.power_logistic_k > 0.0:
            import math
            # Logistic: P(b) = P_range / (1 + exp(-k*(log2(b) - x0))) + P_idle
            x = math.log2(max(1, n_active))
            p = p_range / (1.0 + math.exp(-self.power_logistic_k * (x - self.power_logistic_x0)))
            return self.power_idle_w + p
        else:
            frac = max(0.0, min(1.0, n_active / self.max_slots))
            return self.power_idle_w + p_range * frac

    def invert_power(self, target_w: float) -> int:
        """Return the maximum n_active such that power ≤ target_w.

        Inverts ``power_at_concurrency`` to find the largest batch size that
        stays within a target power budget.  Uses bisection (20 iterations,
        accurate to ±0.5 requests).

        Returns 1 at minimum (at least one request must be in-flight).
        """
        if self.power_idle_w <= 0.0 or self.power_nominal_w <= 0.0:
            raise ValueError(
                f"Power model not configured for profile '{self.name}'."
            )
        target_w = max(self.power_idle_w, min(target_w, self.power_nominal_w))
        lo, hi = 1, self.max_slots
        for _ in range(20):
            mid = (lo + hi) // 2
            if self.power_at_concurrency(mid) <= target_w:
                lo = mid
            else:
                hi = mid - 1
            if lo >= hi:
                break
        return max(1, lo)

    def iter_latency(self, n_active: int,
                     mean_seq_len: Optional[float] = None) -> float:
        """Iteration wall-clock time: W + H_eff × n_active.

        Parameters
        ----------
        n_active     : number of sequences currently in-flight
        mean_seq_len : mean total token length (l_in + l_out) of the active
                       sequences.  When provided, H is scaled proportionally
                       so that attention cost reflects the actual sequence
                       lengths rather than always assuming calibration_ctx.
                       Defaults to calibration_ctx (no scaling) when None.
        """
        if mean_seq_len is not None and self.calibration_ctx > 0:
            H_eff = self.H * (mean_seq_len / self.calibration_ctx)
        else:
            H_eff = self.H
        return self.W + H_eff * n_active

    def n_slots(self, max_ctx: int) -> int:
        """Max concurrent sequences for a pool with this context window.

        Two limits apply:
        - KV-cache memory: total_kv_blks / blks_per_seq(max_ctx)
        - Compute throughput: max_slots × calibration_ctx / max_ctx
          (max_slots was measured at calibration_ctx; at shorter contexts the
          GPU can hold proportionally more sequences before saturating the
          same memory-bandwidth budget)

        Both limits decrease together as max_ctx grows, so they are equal at
        max_ctx == calibration_ctx (by construction of the profile constants).
        """
        blks_per_seq = math.ceil(max_ctx / self.blk_size)
        kv_limit = self.total_kv_blks // blks_per_seq
        compute_cap = self.max_slots * self.calibration_ctx // max(max_ctx, 1)
        return min(compute_cap, kv_limit)

    def prefill_iter_latency(self, chunk_tokens: int,
                              kv_history_tokens: float,
                              n_active: int,
                              mean_seq_len: Optional[float] = None) -> float:
        """Prefill-chunk iteration time (seconds).

        ManualProfile has no compute-throughput spec, so it cannot determine
        whether a given prefill chunk is compute-bound.  Falls back to the
        same memory-bandwidth model as decode — a conservative estimate that
        may slightly overstate prefill latency for compute-bound cases.

        Use ComputedProfile (built from HardwareSpec + ModelSpec) to get the
        roofline-correct prefill time.
        """
        return self.iter_latency(n_active, mean_seq_len)

    def service_time(self, l_in: int, l_out: int, max_ctx: int) -> float:
        """Total service time (s) for one request at steady-state concurrency.

        Uses mean_seq_len = l_in + l_out as a proxy for the typical sequence
        length in the pool, so that attention cost scales correctly with the
        actual request length rather than always assuming calibration_ctx.
        """
        ns = self.n_slots(max_ctx)
        mean_seq_len = float(l_in + l_out)
        iter_t = self.iter_latency(ns, mean_seq_len)
        return (math.ceil(l_in / self.chunk) + l_out) * iter_t

    def throughput(self, max_ctx: int, mean_l_in: float,
                   mean_l_out: float) -> float:
        """Steady-state throughput (req/s) at full load: n_slots / E[S]."""
        ns = self.n_slots(max_ctx)
        es = self.service_time(int(mean_l_in), int(mean_l_out), max_ctx)
        return ns / es if es > 0 else 1e-6
