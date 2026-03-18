"""ComputedProfile — GpuProfile derived from HardwareSpec + ModelSpec.

Implements the same interface as ManualProfile so it can be used
interchangeably throughout the fleet simulator without any changes to
the simulation engine or optimizer.

Token-per-Watt derivation chain
--------------------------------
Every quantity in the tok/W computation traces back to hardware datasheets
and model architecture parameters — no free-fitted scalars::

    model.param_bytes_per_gpu(tp)  →  W  (weight-streaming time per decode iter)
    model.kv_bytes_per_token()     →  H  (marginal latency per in-flight sequence)

    n_active, W, H
        iter_latency(n) = W + H × n_active           [seconds]
        decode_tps(n)   = n_active / iter_latency(n) [output tokens/s]

    kv_frac(n) = n × H_eff / (W + n × H_eff)
        Fraction of HBM traffic coming from KV reads vs. weight streaming.
        Derived entirely from W and H — no additional parameters.

    power(n) = P_idle + (P_active − P_idle) × kv_frac(n)
        P_idle   = hw.power × _POWER_IDLE_FRAC   (43 % TDP — measured, H100 SXM)
        P_active = hw.power × _POWER_ACTIVE_FRAC  (86 % TDP — measured, H100 SXM)
        kv_frac transitions 0 → 1 as batch grows.  Physical interpretation:
        at small batch the GPU streams only model weights (minimum power); at
        large batch it also scans a growing KV cache per iteration, driving
        HBM utilisation and SM activity higher.

    tok/W = decode_tps(n) / power(n)              [tokens / Joule]

The two empirical fractions (_POWER_IDLE_FRAC, _POWER_ACTIVE_FRAC) come from
a single pair of batch=1 and batch=128 measurements published in ML.ENERGY
Benchmark v3.0 (Chung et al., NeurIPS 2025 D&B track).  They are applied to
hw.power (TDP from the manufacturer datasheet) for any GPU, so the power
estimate inherits any error in hw.power.  For GPUs without published
batch-vs-power curves (anything other than H100), treat the result as an
order-of-magnitude estimate only.

Validation (H100-SXM5 + Llama-3.1-70B, TP=8, fp16):
  n=1  : model predicts ≈ P_idle   (300 W); ML.ENERGY reports 300 W  ✓
  n=128: model predicts ≈ P_active (600 W); ML.ENERGY reports 600 W  ✓
  n=32 : model predicts  ~450 W;  ML.ENERGY ~480 W  (6 % error)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..hardware.spec import HardwareSpec
from ..models.spec import ModelSpec
from .builder import ServingConfig

# ── Power-model empirical fractions ──────────────────────────────────────────
#
# Source: ML.ENERGY Benchmark v3.0 (Chung et al., arXiv:2601.22076 /
#         NeurIPS 2025 D&B track), H100-SXM5 + vLLM serving Llama-3-70B.
#   batch=1  →  300 W  →  300 W / 700 W TDP ≈ 0.429
#   batch=128 → 600 W  →  600 W / 700 W TDP ≈ 0.857
#
# Applied as:  P(n) = hw.power × (IDLE + (ACTIVE − IDLE) × kv_frac(n))
# where kv_frac(n) is derived from the roofline W / H decomposition.
#
# These fractions transfer to other GPUs *in the same HBM-bandwidth-bound
# regime* (Ampere, Hopper, Blackwell dense-decode workloads).  Confidence
# degrades for GPUs or workloads outside this regime.
_POWER_IDLE_FRAC: float = 0.43  # P at batch=1  / TDP
_POWER_ACTIVE_FRAC: float = 0.86  # P at batch≫1 / TDP


@dataclass
class DecodeEfficiencyPoint:
    """Full derivation chain from hardware + model → tok/W at one concurrency level.

    All intermediate quantities are exposed so callers can verify the
    derivation or report sub-results independently.

    Derivation order
    ----------------
    1. ``iter_latency_s``  — roofline model: W + H_eff × n_active
    2. ``tokens_per_s``    — n_active / iter_latency_s
    3. ``mem_bytes_per_iter`` — weight bytes + KV bytes (per GPU, per iteration)
    4. ``kv_frac``         — KV share of HBM traffic  =  KV_bytes / total_bytes
    5. ``flops_per_iter``  — 2 × active_params/tp × n  +  attention FLOPs
    6. ``arithmetic_intensity`` — flops / mem_bytes  [flops/byte]
    7. ``roofline_bound``  — "memory" when AI < hw.fp16_tc/mem_bw, else "compute"
    8. ``power_w``         — hw.power × (IDLE + (ACTIVE − IDLE) × kv_frac)
    9. ``tokens_per_watt`` — tokens_per_s / power_w  [tok/J]

    Attributes
    ----------
    n_active            : in-flight sequences (batch size for decode)
    iter_latency_s      : one decode iteration time (s)
    tokens_per_s        : output tokens produced per second by this GPU
    mem_bytes_per_iter  : total HBM bytes read per GPU per decode iteration
    kv_frac             : fraction of mem_bytes_per_iter from KV-cache reads
    flops_per_iter      : FLOPs executed per GPU per decode iteration
    arithmetic_intensity: flops_per_iter / mem_bytes_per_iter  [fl/byte]
    roofline_bound      : "memory" or "compute"
    power_w             : estimated GPU power (W) at this concurrency
    tokens_per_watt     : output tokens per Joule  [tok/J]
    """

    n_active: int
    iter_latency_s: float
    tokens_per_s: float
    mem_bytes_per_iter: float
    kv_frac: float
    flops_per_iter: float
    arithmetic_intensity: float
    roofline_bound: str
    power_w: float
    tokens_per_watt: float

    def show(self) -> str:
        """Return a multi-line human-readable derivation trace."""
        lines = [
            f"  n_active             = {self.n_active}",
            f"  iter_latency         = {self.iter_latency_s*1000:.3f} ms",
            f"  tokens/s (this GPU)  = {self.tokens_per_s:.1f}",
            f"  mem_bytes/iter       = {self.mem_bytes_per_iter/1e9:.2f} GB",
            f"    of which KV frac   = {self.kv_frac:.3f}  "
            f"(KV:{self.kv_frac*self.mem_bytes_per_iter/1e9:.2f} GB  "
            f"weights:{(1-self.kv_frac)*self.mem_bytes_per_iter/1e9:.2f} GB)",
            f"  FLOPs/iter           = {self.flops_per_iter/1e12:.3f} TFLOPs",
            f"  arith. intensity     = {self.arithmetic_intensity:.1f} fl/byte  "
            f"({self.roofline_bound}-bound)",
            f"  power                = {self.power_w:.0f} W",
            "  ──────────────────────────────",
            f"  tok/W                = {self.tokens_per_watt:.4f}  tok/J",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class ComputedProfile:
    """First-principles GPU performance profile.

    Attributes
    ----------
    hw               : physical GPU hardware specification
    model            : model architecture parameters
    cfg              : serving configuration (TP, dtype, chunk size, etc.)
    W                : computed base iteration latency (seconds)
    H                : computed per-sequence overhead (seconds/sequence),
                       calibrated at ``calibration_ctx`` tokens.
    calibration_ctx  : context length at which H was derived (tokens).
    total_kv_blks    : computed KV-cache block budget
    """

    hw: HardwareSpec
    model: ModelSpec
    cfg: ServingConfig
    W: float
    H: float
    total_kv_blks: int
    calibration_ctx: int = 8192

    # ── GpuProfile protocol surface ───────────────────────────────────────────

    @property
    def chunk(self) -> int:
        """Prefill chunk size — delegate to ServingConfig."""
        return self.cfg.chunk

    @property
    def blk_size(self) -> int:
        """KV-cache block size — delegate to ServingConfig."""
        return self.cfg.blk_size

    @property
    def max_slots(self) -> int:
        """Hard slot cap (0 = no cap, defer to KV budget)."""
        return self.cfg.max_slots

    @property
    def name(self) -> str:
        dtype_str = {2.0: "fp16", 1.0: "fp8", 0.5: "int4"}.get(
            self.cfg.dtype_bytes, f"{self.cfg.dtype_bytes}b"
        )
        return f"{self.hw.name}/{self.model.display_name}/TP{self.cfg.tp}/{dtype_str}"

    @property
    def cost_per_hr(self) -> float:
        """Cost of one full instance (tp GPUs) per hour."""
        return self.hw.cost_per_hr * self.cfg.tp

    def iter_latency(self, n_active: int, mean_seq_len: float | None = None) -> float:
        """Iteration wall-clock time: W + H_eff × n_active (seconds).

        When ``mean_seq_len`` is provided, H is scaled by
        ``mean_seq_len / calibration_ctx`` so that attention cost reflects
        the actual sequence lengths in the batch.
        """
        if mean_seq_len is not None and self.calibration_ctx > 0:
            H_eff = self.H * (mean_seq_len / self.calibration_ctx)
        else:
            H_eff = self.H
        return self.W + H_eff * n_active

    def n_slots(self, max_ctx: int) -> int:
        """Max concurrent sequences for a pool with given context window.

        Bounded by (1) the KV-cache memory budget and (2) an optional hard cap
        from ServingConfig.max_slots. No artificial compute cap — the KV budget
        is the dominant constraint for all practical context windows.
        """
        blks_per_seq = math.ceil(max_ctx / self.cfg.blk_size)
        if blks_per_seq == 0:
            return 0
        kv_limit = max(1, self.total_kv_blks // blks_per_seq)
        if self.cfg.max_slots > 0:
            return min(self.cfg.max_slots, kv_limit)
        return kv_limit

    def prefill_iter_latency(
        self,
        chunk_tokens: int,
        kv_history_tokens: float,
        n_active: int,
        mean_seq_len: float | None = None,
    ) -> float:
        """Prefill-chunk iteration time using a full roofline compute/memory check.

        Total prefill FLOPs for one chunk (``c`` tokens, ``q`` KV history)::

            proj_flops  = 2 c h (2 n_h + 2 n_kv) d L / tp   # QKV + O projections
            attn_flops  = 4 n_h d c (c/2 + q) L / tp         # Flash-attention QK+AV
            ffn_flops   = 6 h I_eff c L / tp                  # FFN gate+up+down

        where ``h``=hidden, ``n_h``=n_heads, ``n_kv``=n_kv_heads, ``d``=head_dim,
        ``L``=n_layers, ``tp``=tensor-parallel, ``I_eff``=effective intermediate
        (topk×moe_inter for MoE, intermediate_size for dense).

        For large models the FFN term dominates: e.g., Llama-3.1-70B at chunk=512
        has ~9 ms compute vs ~7 ms weight-streaming on H100, and ~29 ms vs ~12 ms
        on A100, making A100 prefill strongly compute-bound.  The method returns
        ``max(compute_time, mem_time)`` to cover both regimes correctly.

        Parameters
        ----------
        chunk_tokens      : tokens in this prefill chunk (≤ cfg.chunk)
        kv_history_tokens : mean KV tokens already in cache across active seqs
        n_active          : total in-flight sequences when this chunk executes
        mean_seq_len      : mean total seq len for attention-cost H scaling
        """
        mem_time = self.iter_latency(n_active, mean_seq_len)

        # Choose FLOP rate: fp8 when requested and supported, else fp16
        if self.cfg.dtype_bytes <= 1.0 and self.hw.fp8_tc_flops > 0:
            tc_flops = self.hw.fp8_tc_flops
        else:
            tc_flops = self.hw.fp16_tc_flops
        if tc_flops <= 0:
            return mem_time

        # ── Compute-bound check: sum all FLOPs for one prefill chunk ─────────
        #
        # 1. QKV + output projection matmuls
        #    Q:  chunk × hidden × (n_heads × head_dim)
        #    K:  chunk × hidden × (n_kv_heads × head_dim)
        #    V:  chunk × hidden × (n_kv_heads × head_dim)
        #    O:  chunk × (n_heads × head_dim) × hidden
        #    All are weight-stationary matmuls; factor 2 for multiply-accumulate.
        proj_flops = (
            2
            * chunk_tokens
            * self.model.hidden_size
            * (2 * self.model.n_heads + 2 * self.model.n_kv_heads)
            * self.model.head_dim
            * self.model.n_layers
            / self.cfg.tp
        )

        # 2. Flash-attention scores: QK^T (within chunk) + AV (full KV history)
        #    Factor 4 = 2 (QK) + 2 (AV), averaged over the growing KV context.
        attn_flops = (
            4
            * self.model.n_heads
            * self.model.head_dim
            * chunk_tokens
            * (chunk_tokens / 2 + kv_history_tokens)
            * self.model.n_layers
            / self.cfg.tp
        )

        # 3. FFN / MoE FLOPs (gate + up + down, SwiGLU structure)
        if self.model.is_moe:
            ffn_flops = (
                6
                * (
                    self.model.n_experts_topk * self.model.moe_intermediate_size
                    + self.model.n_shared_experts * self.model.intermediate_size
                )
                * self.model.hidden_size
                * chunk_tokens
                * self.model.n_layers
                / self.cfg.tp
            )
        else:
            ffn_flops = (
                6
                * self.model.hidden_size
                * self.model.intermediate_size
                * chunk_tokens
                * self.model.n_layers
                / self.cfg.tp
            )

        compute_time = (proj_flops + attn_flops + ffn_flops) / tc_flops
        return max(mem_time, compute_time)

    def service_time(self, l_in: int, l_out: int, max_ctx: int) -> float:
        """Total service time (s) for one request at steady-state concurrency.

        Separates prefill and decode costs:
        - Prefill iterations: ``max(compute_bound, memory_bound)`` — prefill
          attention is O(chunk²) FLOPs and may be compute-bound for large
          chunks or long-context histories.
        - Decode iterations: memory-bandwidth-bound ``W + H_eff × n_active``.
        """
        ns = self.n_slots(max_ctx)
        mean_seq_len = float(l_in + l_out)

        # Prefill: average KV history across the prefill phase is l_in/2
        # (starts near 0 for first chunk, reaches l_in for the last chunk).
        n_prefill_iters = math.ceil(l_in / self.cfg.chunk)
        kv_history_avg = l_in / 2.0
        prefill_iter_t = self.prefill_iter_latency(
            self.cfg.chunk, kv_history_avg, ns, mean_seq_len
        )
        prefill_time = n_prefill_iters * prefill_iter_t

        # Decode: memory-bandwidth-bound
        decode_iter_t = self.iter_latency(ns, mean_seq_len)
        decode_time = l_out * decode_iter_t

        return prefill_time + decode_time

    def throughput(self, max_ctx: int, mean_l_in: float, mean_l_out: float) -> float:
        """Steady-state request throughput (req/s) at full concurrency."""
        ns = self.n_slots(max_ctx)
        es = self.service_time(int(mean_l_in), int(mean_l_out), max_ctx)
        return ns / es if es > 0 else 1e-6

    # ── Power and efficiency ──────────────────────────────────────────────────

    def power_at_concurrency(self, n_active: int, mean_ctx: int | None = None) -> float:
        """Estimated GPU power (W) at *n_active* in-flight sequences.

        Derived entirely from W, H, model architecture, and hw.power (TDP).
        Two empirical fractions (module-level constants) are applied to TDP.

        Formula::

            H_eff        = H × (mean_ctx / calibration_ctx)
            iter_t       = W + H_eff × n_active

            # GPU activity has two components that each contribute to power:

            kv_frac      = n × H_eff / iter_t
                           HBM traffic fraction from KV-cache reads.
                           Grows 0 → 1 as KV traffic overtakes weight streaming.

            compute_frac = (2 × active_params/tp × n) / (iter_t × fp16_tc_flops)
                           Fraction of peak tensor-core FLOP throughput utilised.
                           Low (<0.2) in memory-bound decode; grows with n.

            activity     = min(1, kv_frac + compute_frac)   # union bound
            P(n)         = hw.power × (_POWER_IDLE_FRAC
                                       + (_POWER_ACTIVE_FRAC - _POWER_IDLE_FRAC)
                                       × activity)

        Accuracy (H100-SXM5 + Llama-3.1-70B, TP=8, fp16, validated vs ML.ENERGY):
          n=1  → ~300 W (measured 300 W,  ~0 % error)
          n=32 → ~420 W (measured ~480 W, ~12 % error)  [ctx-dependent]

        The remaining error at high n reflects: (a) NVLink all-reduce power not
        in the FLOP count, (b) layer-norm / softmax kernel overhead, and
        (c) GPU dynamic voltage scaling at high compute load.

        Parameters
        ----------
        n_active : in-flight (concurrent) sequences
        mean_ctx : mean KV tokens per sequence.  Defaults to
                   ``cfg.mean_ctx_tokens`` used to calibrate H.
        """
        ctx = mean_ctx if mean_ctx is not None else self.cfg.mean_ctx_tokens
        H_eff = (
            self.H * (ctx / self.calibration_ctx)
            if self.calibration_ctx > 0
            else self.H
        )
        iter_t = self.W + H_eff * n_active

        # Component 1: KV-cache fraction of HBM traffic
        kv_frac = (H_eff * n_active) / iter_t if iter_t > 0 else 0.0

        # Component 2: tensor-core compute fraction (no fitting parameters —
        # uses active_params from ModelSpec and fp16_tc_flops from HardwareSpec)
        compute_frac = 0.0
        if iter_t > 0 and self.hw.fp16_tc_flops > 0:
            flops_n = 2.0 * self.model.active_param_count() / self.cfg.tp * n_active
            compute_frac = flops_n / (iter_t * self.hw.fp16_tc_flops)

        activity = min(1.0, kv_frac + compute_frac)
        p_idle = self.hw.power * _POWER_IDLE_FRAC
        p_range = self.hw.power * (_POWER_ACTIVE_FRAC - _POWER_IDLE_FRAC)
        return p_idle + p_range * activity

    def decode_efficiency(
        self, n_active: int, mean_ctx: int | None = None
    ) -> DecodeEfficiencyPoint:
        """Full derivation chain from hardware + model → tok/W.

        Exposes every intermediate quantity so the computation is fully
        transparent and auditable.  See :class:`DecodeEfficiencyPoint` for
        the full list of attributes and their derivation order.

        Parameters
        ----------
        n_active : in-flight (concurrent) sequences
        mean_ctx : representative KV tokens per sequence.  Defaults to
                   ``cfg.mean_ctx_tokens``.
        """
        ctx = mean_ctx if mean_ctx is not None else self.cfg.mean_ctx_tokens

        # ── Step 1: latency from roofline ─────────────────────────────────
        iter_t = self.iter_latency(n_active, mean_seq_len=float(ctx))

        # ── Step 2: output tokens per second ─────────────────────────────
        toks_s = n_active / iter_t if iter_t > 0 else 0.0

        # ── Step 3: HBM bytes read per GPU per decode iteration ───────────
        # Weight bytes: entire model shard streamed once per iteration
        model_bytes = self.model.param_bytes_per_gpu(self.cfg.tp, self.cfg.dtype_bytes)
        # KV bytes: each in-flight sequence has mean_ctx tokens of KV history
        # KV is tensor-parallel-sharded (K and V heads split across TP GPUs)
        kv_per_tok_per_gpu = (
            self.model.kv_bytes_per_token_dtype(self.cfg.dtype_bytes) / self.cfg.tp
        )
        kv_bytes = n_active * kv_per_tok_per_gpu * ctx
        mem_bytes = model_bytes + kv_bytes

        # ── Step 4: KV fraction of HBM traffic ───────────────────────────
        kv_frac = kv_bytes / mem_bytes if mem_bytes > 0 else 0.0

        # ── Step 5: FLOPs per GPU per decode iteration ────────────────────
        # Weight matmuls: 2 × active_params_per_gpu × n_active
        #   Factor-2 = one multiply + one accumulate per MAC.
        #   TP sharding: each GPU holds active_params / tp weights.
        active_params_per_gpu = self.model.active_param_count() / self.cfg.tp
        flops_weights = 2.0 * active_params_per_gpu * n_active

        # Attention (KK^T + softmax + AV) per sequence per layer:
        #   4 × n_kv_heads × head_dim × mean_ctx  (QK: 2×n_kv×d×ctx + AV: 2×n_kv×ctx×d)
        # For n_active sequences across n_layers, TP-sharded:
        flops_attn = (
            4.0
            * self.model.n_kv_heads
            * self.model.head_dim
            * ctx
            * n_active
            * self.model.n_layers
            / self.cfg.tp
        )
        flops = flops_weights + flops_attn

        # ── Step 6: arithmetic intensity ─────────────────────────────────
        ai = flops / mem_bytes if mem_bytes > 0 else 0.0
        ridge = (
            self.hw.fp16_tc_flops / self.hw.mem_bw
            if self.hw.mem_bw > 0
            else float("inf")
        )
        bound = "compute" if ai > ridge else "memory"

        # ── Steps 7–9: power and tok/W ────────────────────────────────────
        # power_at_concurrency() recomputes kv_frac from W/H (same physics,
        # slightly different ctx scaling — kept separate for API consistency)
        power = self.power_at_concurrency(n_active, mean_ctx=ctx)
        tpw = toks_s / power if power > 0 else 0.0

        return DecodeEfficiencyPoint(
            n_active=n_active,
            iter_latency_s=iter_t,
            tokens_per_s=toks_s,
            mem_bytes_per_iter=mem_bytes,
            kv_frac=kv_frac,
            flops_per_iter=flops,
            arithmetic_intensity=ai,
            roofline_bound=bound,
            power_w=power,
            tokens_per_watt=tpw,
        )

    # ── Informational helpers ─────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable profile summary."""
        return (
            f"{self.name}\n"
            f"  W (base iter latency) : {self.W*1000:.3f} ms\n"
            f"  H (per-seq overhead)  : {self.H*1000:.4f} ms/seq\n"
            f"  KV-cache blocks       : {self.total_kv_blks:,}\n"
            f"  Cost                  : ${self.cost_per_hr:.2f}/hr "
            f"({self.cfg.tp} × ${self.hw.cost_per_hr:.2f})"
        )
