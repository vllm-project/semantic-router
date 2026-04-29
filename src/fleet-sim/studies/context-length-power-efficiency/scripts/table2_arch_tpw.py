"""
table2_arch_tpw.py — reproduce Table 2 of paper-f.

Table 2: Single-GPU tok/W at n_max (8K context window) for various
         model architectures on H100-SXM5 and B200-SXM.

Methodology
-----------
All values use ComputedProfile (first-principles) for:
  - W   : per-iteration weight-streaming latency
  - H   : per-sequence KV-scan overhead
  - n_max : KV-cache concurrency limit
  - P   : power via the logistic model (calibrated to H100 ML.ENERGY,
          extrapolated to B200 via TDP fractions)

MoE correction for W
--------------------
ComputedProfile uses *total* parameter bytes for W, which is wrong for
MoE models where only a fraction of experts is active per token.
For Qwen3-235B-A22B (22B active) and DeepSeek-V3 (~37B active), we
override W with:
    W_MoE = active_param_bytes_per_gpu / hw.mem_bw
All other profile quantities (H, n_max, P) remain from ComputedProfile.

Power quality
-------------
H100 : HIGH  (ComputedProfile calibrated to ML.ENERGY v3.0)
B200 : FAIR  (ComputedProfile with TDP-fraction projection)

Usage:
    python scripts/table2_arch_tpw.py
"""

from _sim_path import add_sim_to_syspath

SIM_ROOT = add_sim_to_syspath()

from fleet_sim.hardware.catalog     import H100_SXM, B200_SXM
from fleet_sim.models.catalog       import (LLAMA_3_1_8B, LLAMA_3_1_70B,
                                             LLAMA_3_1_405B, QWEN3_235B_A22B,
                                             DEEPSEEK_V3)
from fleet_sim.gpu_profiles.builder import ProfileBuilder, ServingConfig
from profiles import B200_POWER_MODE, B200_PROFILE_QUALITY, power_for_profile

builder = ProfileBuilder()
CTX = 8192

# (display_name, model_spec, tp, dtype_bytes, active_params_B_for_MoE_or_None)
MODELS = [
    ("Llama-3.1-8B",           LLAMA_3_1_8B,     1, 2.0, None),
    ("Llama-3.1-70B",          LLAMA_3_1_70B,    8, 2.0, None),
    ("Llama-3.1-405B",         LLAMA_3_1_405B,   8, 2.0, None),
    ("Qwen3-235B-A22B (MoE)",  QWEN3_235B_A22B,  8, 2.0, 22e9),   # 22B active
    ("DeepSeek-V3 (MoE, fp8)", DEEPSEEK_V3,      8, 1.0, 37e9),   # ~37B active (est.)
]

def compute_tpw(hw, model, tp, dtype_bytes, active_params_B=None):
    """Compute (n_max, tok/s, tok/W) for a given hardware/model config at CTX."""
    cfg  = ServingConfig(tp=tp, dtype_bytes=dtype_bytes,
                         mean_ctx_tokens=CTX // 2, blk_size=16)
    prof = builder.build(hw=hw, model=model, cfg=cfg)

    n_max = int(prof.total_kv_blks * 16 // CTX)
    if n_max < 1:
        n_max = 1

    # MoE W correction: use active-parameter bytes, not total
    if active_params_B is not None:
        active_bytes_gpu = active_params_B * dtype_bytes / tp
        W_override = active_bytes_gpu / hw.mem_bw
    else:
        W_override = None

    mean = float(CTX // 2)
    il   = prof.iter_latency_w_override(n_max, mean, W_override) \
           if hasattr(prof, "iter_latency_w_override") \
           else (W_override or prof.W) + prof.H * (mean / prof.calibration_ctx) * n_max
    tps  = n_max / il
    p    = power_for_profile(prof, n_max, mean_ctx=CTX // 2)
    return n_max, tps, tps / p

print("Table 2: Single-GPU tok/W at n_max (8K context window)")
print("  Dense: ComputedProfile throughout.  "
      "MoE (*): W corrected to active-param bytes.")
print(f"\n{'Model':28}  {'TP':>3}  "
      f"{'H100 n_max':>10}  {'H100 tok/s':>10}  {'H100 tok/W':>10}  "
      f"{'B200 n_max':>10}  {'B200 tok/s':>10}  {'B200 tok/W':>10}")
print("-" * 110)

for name, model, tp, dtype, active in MODELS:
    ns_h, tps_h, tpw_h = compute_tpw(H100_SXM, model, tp, dtype, active)
    ns_b, tps_b, tpw_b = compute_tpw(B200_SXM, model, tp, dtype, active)
    tag = " *" if active is not None else ""
    print(f"{name:28}  {tp:>3}  "
          f"{ns_h:>10}  {tps_h:>10,.0f}  {tpw_h:>10.2f}  "
          f"{ns_b:>10}  {tps_b:>10,.0f}  {tpw_b:>10.2f}{tag}")

print()
print("* MoE W = active_param_bytes_per_gpu / hw.mem_bw  "
      "(excludes MoE dispatch overhead).")
print("  DeepSeek-V3 active params: ~37B estimated (671B total, 256 experts, top-8).")
print("H100 power quality: HIGH (ComputedProfile calibrated to ML.ENERGY v3.0).")
print(f"B200 power mode: {B200_POWER_MODE} ({B200_PROFILE_QUALITY}).")
