"""
table5_gen_compare.py — reproduce Table 5 of paper-f.

Table 5: GPU generation comparison — H100, H200, B200, GB200 —
         for Llama-3.1-70B (TP=8, fp16) at 8K context window.

All values use first-principles ComputedProfile (consistent methodology).
H100 cross-validated against ML.ENERGY v3.0 (HIGH quality).
H200 / B200 / GB200 are projections (FAIR quality, ±15%).

Usage:
    python scripts/table5_gen_compare.py
"""

from _sim_path import add_sim_to_syspath

SIM_ROOT = add_sim_to_syspath()

from profiles import B200_POWER_MODE, B200_PROFILE_QUALITY, GPU_SPECS, power_for_profile

CTX      = 8192
MEAN_CTX = CTX // 2

print("Table 5: GPU generation comparison, Llama-3.1-70B TP=8 fp16, n_max @ 8K ctx")
print(f"\n{'GPU':12}  {'TDP':>7}  {'P_idle':>7}  {'W (ms)':>7}  "
      f"{'n_max@8K':>9}  {'P_sat(W)':>9}  {'tok/W':>7}  "
      f"{'$/hr':>6}  {'tok/$M':>8}")
print("-" * 95)

for gpu_name, (hw, cp, tdp, p_idle, cost_hr) in GPU_SPECS.items():
    n_max = int(cp.total_kv_blks * 16 // CTX)
    if n_max < 1:
        n_max = 1
    p_sat = power_for_profile(cp, n_max, mean_ctx=MEAN_CTX)
    il    = cp.iter_latency(n_max, float(MEAN_CTX))
    tpw   = n_max / il / p_sat
    tps   = n_max / il
    # tok per $1M: tps / (cost_per_hr / 3600) / 1e6
    tok_per_dollar_M = tps / (cost_hr / 3600) / 1e6

    quality = "HIGH" if gpu_name == "H100-SXM5" else "FAIR"
    print(f"{gpu_name:12}  {tdp:>6}W  {p_idle:>6}W  {cp.W*1000:>7.2f}  "
          f"{n_max:>9}  {p_sat:>9.0f}  {tpw:>7.2f}  "
          f"{cost_hr:>6.1f}  {tok_per_dollar_M:>7.2f}M  ({quality})")

print()
print("Note: All profiles use ComputedProfile (first-principles, non-TP-sharded KV).")
print("      n_max here is lower than in fleet analysis because fleet analysis uses")
print("      the empirically calibrated H100 profile (TP-sharded KV) as the anchor.")
print("      The generation *ratios* are consistent between the two methodologies.")
print(f"      B200 power mode: {B200_POWER_MODE} ({B200_PROFILE_QUALITY}).")
