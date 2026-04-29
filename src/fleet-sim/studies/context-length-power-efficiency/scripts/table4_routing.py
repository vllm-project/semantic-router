"""
table4_routing.py — reproduce Table 4 of paper-f.

Table 4: Single-GPU tok/W for context-window routing vs. semantic routing,
         H100-SXM5, at ρ=0.85 utilization.

Context routing  : 70B model for both short (8K) and long (64K) pools.
Semantic routing : 8B model for short pool (8K), 70B for long pool (64K).

Usage:
    python scripts/table4_routing.py
"""

from _sim_path import add_sim_to_syspath

SIM_ROOT = add_sim_to_syspath()

from fleet_sim.hardware.catalog     import H100_SXM
from fleet_sim.models.catalog       import LLAMA_3_1_8B, LLAMA_3_1_70B
from fleet_sim.gpu_profiles.builder import ProfileBuilder, ServingConfig
from fleet_sim.gpu_profiles.profiles import H100_80GB

builder = ProfileBuilder()

# 8B profile (TP=1, full KV — no TP sharding since single GPU)
_cfg8 = ServingConfig(tp=1, dtype_bytes=2.0, mean_ctx_tokens=4096, blk_size=16)
CP_8B = builder.build(hw=H100_SXM, model=LLAMA_3_1_8B, cfg=_cfg8)

RHO = 0.85  # operating utilization

def pool_stats(prof, ctx, mean_ctx=None):
    """Return computed and display concurrency plus power/tok/W at ρ=RHO."""
    if mean_ctx is None:
        mean_ctx = ctx // 2
    n_max = prof.n_slots(ctx)
    n_act = max(1, int(n_max * RHO))
    n_act_display = max(1, int(round(n_max * RHO)))
    p     = prof.power_at_concurrency(n_act)
    il    = prof.iter_latency(n_act, float(mean_ctx))
    tpw   = n_act / il / p
    return n_max, n_act_display, p, tpw

rows = [
    ("Context short (70B@8K)",  "Llama-3.1-70B", "8K",  H100_80GB, 8192),
    ("Context long  (70B@64K)", "Llama-3.1-70B", "64K", H100_80GB, 65536),
    ("Semantic small (8B@8K)",  "Llama-3.1-8B",  "8K",  CP_8B,     8192),
    ("Semantic large (70B@64K)","Llama-3.1-70B", "64K", H100_80GB, 65536),
]

print("Table 4: Per-pool tok/W for context-window vs. semantic routing (H100, ρ=0.85)")
print(f"\n{'Pool type':28}  {'Model':15}  {'ctx':>4}  "
      f"{'n_max':>6}  {'n_act':>6}  {'P (W)':>7}  {'tok/W':>7}")
print("-" * 85)

for label, mname, ctx_lbl, prof, ctx in rows:
    nm, na, p, tpw = pool_stats(prof, ctx)
    print(f"{label:28}  {mname:15}  {ctx_lbl:>4}  "
          f"{nm:>6}  {na:>6}  {p:>7.0f}  {tpw:>7.2f}")

print()
# Compute and print key ratio
_, _, _, tpw_ctx_s  = pool_stats(H100_80GB, 8192)
_, _, _, tpw_ctx_l  = pool_stats(H100_80GB, 65536)
_, _, _, tpw_sem_s  = pool_stats(CP_8B, 8192)

print(f"Short-pool (70B@8K) vs long-pool (70B@64K) tok/W ratio: {tpw_ctx_s/tpw_ctx_l:.1f}×")
print(f"Semantic small (8B@8K) vs long-pool (70B@64K) tok/W ratio: {tpw_sem_s/tpw_ctx_l:.1f}×")
print()
print("Note: tok/W is per serving instance (TP=8 group for 70B, TP=1 group for 8B).")
print("      On a per-physical-GPU basis 8B is more efficient (fewer GPUs needed).")
print("      The long pool (64K) is the binding constraint in both routing schemes.")
