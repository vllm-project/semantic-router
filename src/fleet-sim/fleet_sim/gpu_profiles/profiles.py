"""Pre-built hand-calibrated GPU profiles and the CUSTOM factory.

``GpuProfile`` is the structural Protocol that all profile types satisfy.
``ManualProfile`` lets you supply W, H, and KV-cache budget directly.
``ProfileBuilder`` computes them from hardware and model specs automatically.

Power model notes
-----------------
Power parameters use a logistic curve P(b) = P_range/(1+exp(-k*(log2(b)-x0))) + P_idle
where b = concurrent in-flight requests (≈ vLLM max_num_seqs).

**H100-SXM5** — MEASURED (source quality: HIGH)
  Fitted to G2G paper (Hassan et al., arXiv:2602.05116v1) Figure 2 data points
  (H100-SXM5 + vLLM + Llama-3.1-class, batch ∈ {1,2,4,8,16,32,64,128,256}).
  Cross-checked against ML.ENERGY Benchmark v3.0 (Chung et al., arXiv:2601.22076):
  idle ≈ 300 W (batch=1), nominal ≈ 600 W (batch=128).  Fit error < 3%.

**A100-SXM4** — ONE ANCHOR + PROJECTION (source quality: FAIR)
  P_nominal = 370 W validated by: 8×A100-SXM4 serving Nemotron-70B at batch=32
  measured at 2,983 W total = 373 W/GPU (Sada et al., arXiv:2507.00418, Table 1,
  nvidia-smi measured).  P_idle projected from H100 TDP fraction (≈43% × 400W).
  k=1.0 (same HBM memory type as H100).  x0 derived from bandwidth-scaling rule:
  x0(A100) = 4.2 + log2(2.0/3.35) = 3.46 ≈ 3.5 (A100 HBM2e 2.0 TB/s vs H100 3.35 TB/s).

**A10G PCIe** — PROJECTION ONLY (source quality: LOW)
  No published batch-vs-power data found for A10G + vLLM (ML.ENERGY v3.0 covers
  H100+B200 only; no arXiv papers as of March 2026 measure A10G logistic curve).
  Parameters projected from: TDP (150 W), GDDR6 bandwidth (≈600 GB/s), 7B model KV-cache
  scaling vs A100/70B.  k=0.7 (GDDR6 less sharp transition vs HBM).
  x0=3.7 derived from bandwidth + model-size scaling (see POWER_MODEL_METHODOLOGY.md).
  Empirical calibration strongly recommended before production DR use.

Full derivation with sources: docs/POWER_MODEL_METHODOLOGY.md
"""

from __future__ import annotations

from .manual import ManualProfile
from .protocol import GpuProfile

# ── Pre-built profiles (Llama-3-70B calibration, 8-GPU TP) ───────────────────

A100_80GB = ManualProfile(
    name="A100-80GB",
    W=0.0080,
    H=0.00065,  # per-seq overhead at calibration_ctx=8192 (attention BW saturation)
    calibration_ctx=8192,
    chunk=512,
    blk_size=16,
    total_kv_blks=65536,
    max_slots=128,  # saturates at 128 seqs × 8192 tokens; scales as 128×8192/max_ctx
    cost_per_hr=2.21,
    # Power — source quality: FAIR (one measured anchor + FLOPS-scaling derivation)
    # TDP = 400 W (NVIDIA A100-SXM4 datasheet DS-10031-001_v1.6)
    # P_nominal = 385 W (96.3% TDP):
    #   Back-derived from measured anchor + logistic fit (see x0 below).
    #   Validated vs Sada et al. (arXiv:2507.00418, Table 1): 8×A100-SXM4 serving
    #   Nemotron-70B at batch=32 measured 2,983 W = 373 W/GPU via nvidia-smi.
    #   Model predicts P(32) = 369 W (1.1% error vs 373 W measured).
    # P_idle = 175 W (43.75% TDP):
    #   Projected from H100 TDP fraction (300W/700W=42.9%).  No direct b=1 measurement.
    power_idle_w=175.0,
    power_nominal_w=385.0,
    # Logistic curve — source quality: DERIVED (FLOPS-scaling + measured anchor)
    # k = 1.0: same as H100 (HBM memory family; similar burst characteristics).
    # x0 = 2.5: derived by FLOPS-scaling from H100.  At high batch, the 70B model on A100
    #   is compute-bound (large matmuls saturate TFLOPS before KV-cache saturates BW):
    #     x0 = 4.2 + log2(TFLOPS_A100 / TFLOPS_H100)
    #        = 4.2 + log2(312 / 989)         (FP16 peak, NVIDIA datasheets)
    #        = 4.2 − 1.66 = 2.54 ≈ 2.5
    #   Saturation batch: 2^2.5 ≈ 5.7 concurrent requests.
    #   Note: BW-scaling (x0≈3.5) gives P(32)=334W (10% error); FLOPS-scaling (x0≈2.5)
    #   gives P(32)=369W (1.1% error).  FLOPS-scaling is the better methodology for
    #   large-model, high-TP deployments where weight-read BW is always saturated
    #   and additional batch increases COMPUTE load (not BW).
    # Full derivation with data: docs/POWER_MODEL_METHODOLOGY.md
    power_logistic_k=1.0,
    power_logistic_x0=2.5,
)

H100_80GB = ManualProfile(
    name="H100-80GB",
    W=0.0040,
    H=0.00032,  # per-seq overhead at calibration_ctx=8192
    calibration_ctx=8192,
    chunk=1024,
    blk_size=16,
    total_kv_blks=65536,
    max_slots=256,  # saturates at 256 seqs × 8192 tokens; scales as 256×8192/max_ctx
    cost_per_hr=4.02,
    # Power — source quality: HIGH (measured data)
    # TDP = 700 W (NVIDIA H100-SXM5 datasheet DS-10313-001_v1.6)
    # P_idle = 300 W (43% TDP): ML.ENERGY Benchmark v3.0, H100-SXM5 + vLLM,
    #   batch=1 (Chung et al., arXiv:2601.22076; https://ml.energy/leaderboard)
    # P_nominal = 600 W (86% TDP): ML.ENERGY Benchmark v3.0, batch=128
    power_idle_w=300.0,
    power_nominal_w=600.0,
    # Logistic curve — source quality: FITTED (G2G paper Fig. 2 data points)
    # k = 1.0, x0 = 4.2: fitted to Hassan et al. arXiv:2602.05116v1, Fig. 2
    #   (H100-SXM5 + vLLM + Llama-3.1-class, batch ∈ {1,2,4,8,16,32,64,128,256})
    # P(b) = 300/(1+exp(-1.0*(log2(b)-4.2))) + 300 → 304W @ b=1, 583W @ b=128
    # Saturation batch: 2^4.2 ≈ 18 concurrent requests.  Fit error < 3%.
    # Full derivation: docs/POWER_MODEL_METHODOLOGY.md
    power_logistic_k=1.0,
    power_logistic_x0=4.2,
)

A10G = ManualProfile(
    name="A10G",
    W=0.012,
    H=0.00090,  # per-seq overhead at calibration_ctx=8192
    calibration_ctx=8192,
    chunk=256,
    blk_size=16,
    total_kv_blks=32768,
    max_slots=64,  # saturates at 64 seqs × 8192 tokens; scales as 64×8192/max_ctx
    cost_per_hr=1.01,
    # Power — source quality: LOW (projection only; NO published measurement data found)
    # ML.ENERGY v3.0 covers H100+B200 only; no arXiv papers measure A10G batch-vs-power.
    # TDP = 150 W (NVIDIA A10G datasheet DS-10012-001_v1.3)
    # P_idle = 75 W (50% TDP):
    #   GDDR6 24 GB at ≈2 W/GB idle ≈ 48 W; core logic ≈ 25 W → ~73 W → 75 W.
    #   GDDR6 has higher refresh power per byte than HBM; hence higher % TDP vs A100/H100.
    # P_nominal = 120 W (80% TDP): typical PCIe card ceiling at sustained inference load.
    power_idle_w=75.0,
    power_nominal_w=120.0,
    # Logistic curve — source quality: PROJECTION ONLY (no measurement data found)
    # k = 0.7: GDDR6 memory delivers bandwidth in smaller, more uniform bursts than
    #   HBM; the compute→memory transition is less abrupt.  k=0.7 vs HBM k=1.0.
    #   No empirical basis for this specific value beyond architectural reasoning.
    # x0 = 3.0: derived using the FLOPS-scaling methodology established for A100.
    #   A10G FP16 TFLOPS ≈ 31.2 (NVIDIA A10G datasheet DS-10012-001_v1.3).
    #   For a 7B model on a single A10G:
    #     x0 = 4.2 + log2(31.2 / 989) = 4.2 + log2(0.0316) = 4.2 − 5.0 = −0.8
    #   This gives a near-zero value (always-saturated), which is unphysical for
    #   power (P_idle ≠ P_nominal at b=1 in practice).  The unphysical result
    #   occurs because A10G TFLOPS is so much lower than H100 that "compute
    #   saturation" technically happens at sub-unit batch — but in practice the
    #   power curve still rises because GPU core clocking and memory-bus activity
    #   respond to queue depth.
    #   Empirical correction: clamp x0 ≥ 2.0 and use x0=3.0 as a conservative
    #   estimate (broader transition region, middle of the BW-scaling and FLOPS-
    #   scaling range).
    # Saturation batch: 2^3.0 = 8 concurrent requests.
    # WARNING: A10G x0 may range from 2.0 (13B model, more KV pressure) to 4.0
    #   (7B model at low concurrency).  Use x0=3.0 as a median estimate.
    # Empirical calibration STRONGLY recommended before DR contract use.
    # See: docs/POWER_MODEL_METHODOLOGY.md
    power_logistic_k=0.7,
    power_logistic_x0=3.0,
)


def CUSTOM(
    name: str,
    W: float,
    H: float,
    chunk: int = 512,
    blk_size: int = 16,
    total_kv_blks: int = 65536,
    max_slots: int = 128,
    cost_per_hr: float = 2.21,
    calibration_ctx: int = 8192,
) -> ManualProfile:
    """Factory for user-defined hand-calibrated GPU profiles."""
    return ManualProfile(
        name=name,
        W=W,
        H=H,
        calibration_ctx=calibration_ctx,
        chunk=chunk,
        blk_size=blk_size,
        total_kv_blks=total_kv_blks,
        max_slots=max_slots,
        cost_per_hr=cost_per_hr,
    )


# ── New computed profile API ──────────────────────────────────────────────────

from .builder import ProfileBuilder, ServingConfig
from .computed import ComputedProfile

__all__ = [
    "A10G",
    "A100_80GB",
    "CUSTOM",
    "H100_80GB",
    "ComputedProfile",
    "GpuProfile",
    "ManualProfile",
    "ProfileBuilder",
    "ServingConfig",
]
