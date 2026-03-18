"""DisaggFleetOptimizer — disaggregated prefill/decode fleet sizing.

Models a system where prefill (context) and decode (generation) phases
run on separate, independently-scaled GPU pools. Finds the optimal
(n_prefill, n_decode) worker ratio that maximises per-GPU throughput
while satisfying TTFT and TPOT SLA constraints.

Modeling constants (validated against production deployments):
  α_pre  = 0.90  — prefill throughput degradation from pipeline interference
  α_dec  = 0.92  — decode throughput degradation from pipeline interference
  β_TTFT = 1.80  — TTFT correction factor for KV-cache transfer overhead

Source: AIConfigurator Algorithm 3 (arXiv 2601.06288), NVIDIA 2025.
Constants validated on DeepSeek-V3 disaggregated serving across 2 nodes
and Qwen3-32B-FP8 on 8× H200 (production SLA case study).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ── Published empirical constants ─────────────────────────────────────────────

ALPHA_PRE: float = 0.90    # prefill throughput degradation
ALPHA_DEC: float = 0.92    # decode throughput degradation
BETA_TTFT: float = 1.80    # TTFT multiplier for KV-transfer overhead


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DisaggResult:
    """Optimal disaggregated fleet configuration.

    Attributes
    ----------
    n_prefill       : number of prefill worker instances
    n_decode        : number of decode worker instances
    prefill_gpus    : GPUs per prefill instance (= prefill profile TP)
    decode_gpus     : GPUs per decode instance (= decode profile TP)
    total_gpus      : total GPU count
    thru_per_gpu    : system throughput (req/s) per GPU at optimal config
    system_rate     : total system request throughput (req/s)
    ttft_ms         : effective TTFT including KV-transfer (ms)
    tpot_ms         : effective TPOT for decode pool (ms)
    cost_per_hr     : total fleet cost ($/hr)
    prefill_name    : profile name for prefill instances
    decode_name     : profile name for decode instances
    """
    n_prefill: int
    n_decode: int
    prefill_gpus: int
    decode_gpus: int
    total_gpus: int
    thru_per_gpu: float
    system_rate: float
    ttft_ms: float
    tpot_ms: float
    cost_per_hr: float
    prefill_name: str
    decode_name: str

    def print_report(self) -> None:
        print(
            f"\n{'='*60}\n"
            f"  Disaggregated Fleet Result\n"
            f"{'='*60}\n"
            f"  Config           : {self.n_prefill}P × {self.n_decode}D\n"
            f"  Prefill profile  : {self.prefill_name}\n"
            f"  Decode profile   : {self.decode_name}\n"
            f"  Total GPUs       : {self.total_gpus}\n"
            f"  System rate      : {self.system_rate:.2f} req/s\n"
            f"  Throughput/GPU   : {self.thru_per_gpu:.4f} req/s/GPU\n"
            f"  Effective TTFT   : {self.ttft_ms:.1f} ms\n"
            f"  Effective TPOT   : {self.tpot_ms:.1f} ms\n"
            f"  Fleet cost       : ${self.cost_per_hr:.2f}/hr\n"
            f"{'='*60}\n"
        )


@dataclass
class DisaggSweepPoint:
    """One point in a disaggregated fleet sweep."""
    n_prefill: int
    n_decode: int
    total_gpus: int
    system_rate: float
    thru_per_gpu: float
    ttft_ms: float
    tpot_ms: float
    cost_per_hr: float
    slo_met: bool


# ── Optimizer ─────────────────────────────────────────────────────────────────

class DisaggFleetOptimizer:
    """Find the optimal xP + yD disaggregated fleet configuration.

    Parameters
    ----------
    prefill_profile : GpuProfile for prefill (context) workers.
                      Use ServingConfig(phase="prefill") with ProfileBuilder
                      to get a compute-optimised prefill profile, or pass any
                      existing ManualProfile / ComputedProfile.
    decode_profile  : GpuProfile for decode (generation) workers.
                      Use ServingConfig(phase="decode") for memory-optimised.
    mean_isl        : mean input sequence length (tokens) — used to compute
                      prefill TTFT from the prefill profile's iter_latency.
    mean_osl        : mean output sequence length (tokens) — used to compute
                      TPOT from the decode profile.
    slo_ttft_ms     : maximum acceptable TTFT (ms), *including* KV-transfer.
    slo_tpot_ms     : maximum acceptable TPOT (ms).
    max_ctx         : pool context window (tokens).
    alpha_pre       : prefill throughput degradation factor (default 0.90).
    alpha_dec       : decode throughput degradation factor (default 0.92).
    beta_ttft       : TTFT correction for KV-transfer overhead (default 1.80).
    """

    def __init__(
        self,
        prefill_profile,
        decode_profile,
        mean_isl: float = 1024,
        mean_osl: float = 256,
        slo_ttft_ms: float = 2000.0,
        slo_tpot_ms: float = 50.0,
        max_ctx: int = 4096,
        alpha_pre: float = ALPHA_PRE,
        alpha_dec: float = ALPHA_DEC,
        beta_ttft: float = BETA_TTFT,
    ):
        self.prefill_profile = prefill_profile
        self.decode_profile = decode_profile
        self.mean_isl = mean_isl
        self.mean_osl = mean_osl
        self.slo_ttft_ms = slo_ttft_ms
        self.slo_tpot_ms = slo_tpot_ms
        self.max_ctx = max_ctx
        self.alpha_pre = alpha_pre
        self.alpha_dec = alpha_dec
        self.beta_ttft = beta_ttft

        # GPUs consumed by one instance of each pool
        self._prefill_gpus = self._instance_gpus(prefill_profile)
        self._decode_gpus = self._instance_gpus(decode_profile)

    # ── Per-instance performance ──────────────────────────────────────────────

    def _instance_gpus(self, profile) -> int:
        """Return the TP degree (GPUs per instance) for a profile."""
        # ComputedProfile exposes cfg.tp; ManualProfile has no TP concept → 1
        if hasattr(profile, "cfg"):
            return profile.cfg.tp
        return 1

    def _prefill_ttft_ms(self) -> float:
        """Base TTFT (ms) for one prefill instance processing mean_isl tokens."""
        import math
        chunk = getattr(self.prefill_profile, "chunk",
                        getattr(getattr(self.prefill_profile, "cfg", None), "chunk", 512))
        n_slots = self.prefill_profile.n_slots(self.max_ctx)
        iter_t = self.prefill_profile.iter_latency(n_slots)
        prefill_iters = math.ceil(self.mean_isl / chunk)
        return prefill_iters * iter_t * 1000.0

    def _decode_tpot_ms(self) -> float:
        """TPOT (ms) for one decode instance at steady-state concurrency."""
        n_slots = self.decode_profile.n_slots(self.max_ctx)
        return self.decode_profile.iter_latency(n_slots) * 1000.0

    def _prefill_thru(self) -> float:
        """Single prefill instance throughput (req/s)."""
        return self.prefill_profile.throughput(self.max_ctx, self.mean_isl, 1.0)

    def _decode_thru(self) -> float:
        """Single decode instance throughput (req/s)."""
        return self.decode_profile.throughput(self.max_ctx, 1.0, self.mean_osl)

    # ── Main sweep ────────────────────────────────────────────────────────────

    def optimize(
        self,
        max_prefill: int = 32,
        max_decode: int = 64,
        valid_total_gpus: Optional[List[int]] = None,
    ) -> Optional[DisaggResult]:
        """Find (n_prefill, n_decode) that maximises per-GPU throughput.

        Parameters
        ----------
        max_prefill         : maximum prefill worker count to sweep
        max_decode          : maximum decode worker count to sweep
        valid_total_gpus    : if set, only consider total GPU counts in this
                              list (e.g. [8, 16, 32, 64] for a fixed cluster)
        """
        thru_pre_single = self._prefill_thru()
        thru_dec_single = self._decode_thru()
        ttft_base_ms    = self._prefill_ttft_ms()
        tpot_ms         = self._decode_tpot_ms()

        effective_ttft  = ttft_base_ms * self.beta_ttft
        if effective_ttft > self.slo_ttft_ms or tpot_ms > self.slo_tpot_ms:
            # Single-instance already violates SLO — cannot build valid fleet
            return None

        best: Optional[DisaggResult] = None
        best_thru_gpu = 0.0

        for n_pre in range(1, max_prefill + 1):
            for n_dec in range(1, max_decode + 1):
                g_total = n_pre * self._prefill_gpus + n_dec * self._decode_gpus

                if valid_total_gpus and g_total not in valid_total_gpus:
                    continue

                r_pre = thru_pre_single * n_pre * self.alpha_pre
                r_dec = thru_dec_single * n_dec * self.alpha_dec
                r_sys = min(r_pre, r_dec)

                thru_gpu = r_sys / g_total
                cost = (n_pre * self.prefill_profile.cost_per_hr
                        + n_dec * self.decode_profile.cost_per_hr)

                if thru_gpu > best_thru_gpu:
                    best_thru_gpu = thru_gpu
                    best = DisaggResult(
                        n_prefill=n_pre,
                        n_decode=n_dec,
                        prefill_gpus=self._prefill_gpus,
                        decode_gpus=self._decode_gpus,
                        total_gpus=g_total,
                        thru_per_gpu=thru_gpu,
                        system_rate=r_sys,
                        ttft_ms=effective_ttft,
                        tpot_ms=tpot_ms,
                        cost_per_hr=cost,
                        prefill_name=self.prefill_profile.name,
                        decode_name=self.decode_profile.name,
                    )

        return best

    def sweep(
        self,
        max_prefill: int = 16,
        max_decode: int = 32,
        valid_total_gpus: Optional[List[int]] = None,
    ) -> List[DisaggSweepPoint]:
        """Return all SLO-feasible (n_prefill, n_decode) configurations.

        Useful for plotting Pareto frontiers of throughput vs. cost.
        """
        thru_pre_single = self._prefill_thru()
        thru_dec_single = self._decode_thru()
        effective_ttft  = self._prefill_ttft_ms() * self.beta_ttft
        tpot_ms         = self._decode_tpot_ms()

        points: List[DisaggSweepPoint] = []

        for n_pre in range(1, max_prefill + 1):
            for n_dec in range(1, max_decode + 1):
                g_total = n_pre * self._prefill_gpus + n_dec * self._decode_gpus

                if valid_total_gpus and g_total not in valid_total_gpus:
                    continue

                r_sys = min(
                    thru_pre_single * n_pre * self.alpha_pre,
                    thru_dec_single * n_dec * self.alpha_dec,
                )
                thru_gpu = r_sys / g_total
                cost = (n_pre * self.prefill_profile.cost_per_hr
                        + n_dec * self.decode_profile.cost_per_hr)
                slo_met = (
                    effective_ttft <= self.slo_ttft_ms
                    and tpot_ms <= self.slo_tpot_ms
                )

                points.append(DisaggSweepPoint(
                    n_prefill=n_pre, n_decode=n_dec,
                    total_gpus=g_total,
                    system_rate=r_sys,
                    thru_per_gpu=thru_gpu,
                    ttft_ms=effective_ttft,
                    tpot_ms=tpot_ms,
                    cost_per_hr=cost,
                    slo_met=slo_met,
                ))

        return points
