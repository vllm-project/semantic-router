"""
profiles.py — canonical GPU profiles used throughout paper-f.

All tables in the paper are computed from these two profiles:
  - H100_PROFILE : empirically calibrated ManualProfile (HIGH quality)
                   from ML.ENERGY v3.0 / Hassan et al. G2G data.
  - B200_PROFILE : projected ManualProfile (FAIR quality, ±20%)
                   scaled proportionally from H100 using the
                   ComputedProfile KV-budget ratio (2.62×).

First-principles (ComputedProfile) instances for H200 and GB200 are
also provided for the GPU generation comparison table (Table 5).

Run this file standalone to print a quick sanity-check of all profiles:
    python scripts/profiles.py
"""

import math

from _sim_path import add_sim_to_syspath

# ── Resolve path to inference-fleet-simulator ────────────────────────────────
SIM_ROOT = add_sim_to_syspath()

from fleet_sim.hardware.catalog   import H100_SXM, H200_SXM, B200_SXM, GB200
from fleet_sim.models.catalog     import LLAMA_3_1_70B
from fleet_sim.gpu_profiles.builder import ProfileBuilder, ServingConfig
from fleet_sim.gpu_profiles.profiles import H100_80GB          # empirical
from fleet_sim.gpu_profiles.manual   import ManualProfile
from fleet_sim.gpu_profiles.computed import ComputedProfile

# ── H100 empirical (ManualProfile, HIGH quality) ─────────────────────────────
# Source: Hassan et al. 2026 (G2G), ML.ENERGY v3.0.
# W=4.0 ms, H=0.32 ms/seq @ cal_ctx=8192, TP-sharded GQA KV (1 head/GPU),
# n_max=128 @ 8K context, n_max=16 @ 64K context.
H100_PROFILE = H100_80GB   # imported directly from the simulator

# ── B200 projected (ManualProfile, FAIR quality) ──────────────────────────────
# Methodology:
#   1. Build ComputedProfile for H100 and B200 (first principles).
#   2. Take KV-budget ratio R = cp_b200.total_kv_blks / cp_h100.total_kv_blks.
#   3. Scale H100 empirical total_kv_blks and max_slots by R.
#   4. Scale H (per-seq KV-scan latency) by H100 BW / B200 BW (bandwidth ratio).
#   5. Use cp_b200.W for the per-iteration weight-streaming latency.
#   6. Apply TDP-fraction heuristics for power: idle=0.43×TDP, nom=0.86×TDP.
# This ensures the KV-concurrency model is anchored to H100's empirical
# calibration while correctly accounting for B200's larger VRAM budget.

_builder = ProfileBuilder()
_cfg70   = ServingConfig(tp=8, dtype_bytes=2.0, mean_ctx_tokens=4096, blk_size=16)
_cp_h100 = _builder.build(hw=H100_SXM, model=LLAMA_3_1_70B, cfg=_cfg70)
_cp_b200 = _builder.build(hw=B200_SXM, model=LLAMA_3_1_70B, cfg=_cfg70)

KV_RATIO  = _cp_b200.total_kv_blks / _cp_h100.total_kv_blks   # 2.62×
BW_RATIO  = H100_SXM.mem_bw / B200_SXM.mem_bw                  # 0.419×
W_B200    = _cp_b200.W                                          # 2.955 ms
H_B200    = H100_PROFILE.H * BW_RATIO                          # 0.134 ms/seq @ cal_ctx=8192

B200_PROFILE = ManualProfile(
    name             = "B200-SXM-projected",
    W                = W_B200,
    H                = H_B200,
    calibration_ctx  = H100_PROFILE.calibration_ctx,            # 8192
    chunk            = 2048,
    blk_size         = 16,
    total_kv_blks    = int(H100_PROFILE.total_kv_blks * KV_RATIO),   # 171 936
    max_slots        = int(H100_PROFILE.max_slots * KV_RATIO),        # 671
    cost_per_hr      = 64.0,                                    # 8× GPU @ $8/hr
    power_idle_w     = B200_SXM.power * 0.43,                  # 430 W
    power_nominal_w  = B200_SXM.power * 0.86,                  # 860 W
    power_logistic_k = 1.0,
    power_logistic_x0= math.log2(W_B200 / H_B200),             # 4.46
)

# ── First-principles profiles for GPU generation comparison (Table 5) ─────────
# These use ComputedProfile (consistent methodology across all GPUs).
# They are NOT used for fleet analysis (no empirical anchor for H200/GB200).
def make_computed_profile(hw, label):
    cfg = ServingConfig(tp=8, dtype_bytes=2.0, mean_ctx_tokens=4096, blk_size=16)
    return _builder.build(hw=hw, model=LLAMA_3_1_70B, cfg=cfg)

CP_H100  = _cp_h100
CP_H200  = make_computed_profile(H200_SXM,  "H200-SXM")
CP_B200  = _cp_b200
CP_GB200 = make_computed_profile(GB200,     "GB200-NVL")

GPU_SPECS = {
    "H100-SXM5": (H100_SXM,  CP_H100,  700,  300,  32.2),
    "H200-SXM" : (H200_SXM,  CP_H200,  700,  300,  48.0),
    "B200-SXM" : (B200_SXM,  CP_B200,  1000, 430,  64.0),
    "GB200-NVL": (GB200,      CP_GB200, 1200, 516,  80.0),
}


# ── Standalone sanity check ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("H100 profile (empirical):")
    for ctx in [4096, 8192, 65536]:
        ns = H100_PROFILE.n_slots(ctx)
        print(f"  n_slots({ctx//1024}K) = {ns:4d}  "
              f"P = {H100_PROFILE.power_at_concurrency(ns):.0f} W")

    print("\nB200 profile (projected, KV ratio = {:.2f}×):".format(KV_RATIO))
    for ctx in [4096, 8192, 65536]:
        ns = B200_PROFILE.n_slots(ctx)
        print(f"  n_slots({ctx//1024}K) = {ns:4d}  "
              f"P = {B200_PROFILE.power_at_concurrency(ns):.0f} W")

    print("\nB200/H100 tok/W ratio at key context windows:")
    for ctx in [4096, 8192, 65536]:
        m = ctx // 2
        ns_h = H100_PROFILE.n_slots(ctx)
        ph   = H100_PROFILE.power_at_concurrency(ns_h)
        tpwh = ns_h / H100_PROFILE.iter_latency(ns_h, float(m)) / ph

        ns_b = B200_PROFILE.n_slots(ctx)
        pb   = B200_PROFILE.power_at_concurrency(ns_b)
        tpwb = ns_b / B200_PROFILE.iter_latency(ns_b, float(m)) / pb

        print(f"  {ctx//1024}K:  H100={tpwh:.2f}  B200={tpwb:.2f}  "
              f"ratio={tpwb/tpwh:.2f}×")
