"""
table3_fleet_tpw.py — reproduce Table 3 of paper-f.

Table 3: Fleet token efficiency at λ=1000 req/s, P99 TTFT ≤ 500 ms,
         Llama-3.1-70B TP=8 fp16, across topologies and GPU generations.

Workloads
---------
  Azure        : azure_cdf.json        (B_short=4096, gamma*=2.0)
  LMSYS multi  : lmsys_multiturn_cdf.json (B_short=1536, gamma*=2.0)

Topologies
----------
  Homo     : single homogeneous pool, max_ctx=64K
  Pool     : two-pool context routing at B_short
  FleetOpt : two-pool with overflow gamma*

Profile quality
---------------
  H100 : HIGH  (empirically calibrated ManualProfile)
  B200 : FAIR  (projected, ±20% on absolute values)

Usage:
    python scripts/table3_fleet_tpw.py
"""

import json
import os

from _sim_path import add_sim_to_syspath

SIM_ROOT = add_sim_to_syspath()

from profiles import B200_POWER_MODE, B200_PROFILE_QUALITY, B200_PROFILE, H100_PROFILE
from fleet_sim.optimizer import fleet_tpw_analysis, _split_cdf

# ── Workload CDFs ─────────────────────────────────────────────────────────────
SIM_DATA = os.path.join(SIM_ROOT, "data")


def _load_cdf(fname):
    d = json.load(open(os.path.join(SIM_DATA, fname)))
    return d["cdf"] if isinstance(d, dict) else d


def _cdf_eval(cdf, threshold):
    """Linearly interpolate the cumulative fraction at `threshold`."""
    if not cdf:
        return 0.0

    prev_t, prev_f = 0, 0.0
    for t, f in cdf:
        if threshold <= t:
            if t == prev_t:
                return f
            return prev_f + (f - prev_f) * (threshold - prev_t) / (t - prev_t)
        prev_t, prev_f = t, f

    return cdf[-1][1]

WORKLOADS = [
    ("Azure",  _load_cdf("azure_cdf.json"),          4096, 2.0),
    ("LMSYS",  _load_cdf("lmsys_multiturn_cdf.json"), 1536, 2.0),
]

LAM = 1000.0   # req/s
SLO = 500.0    # ms  P99 TTFT

GPUS = [("H100", H100_PROFILE), ("B200", B200_PROFILE)]

print("Table 3: Fleet tok/W at λ=1000 req/s, P99 TTFT ≤ 500 ms")
print(f"\n{'Workload':8}  {'Topology':30}  {'GPU':6}  "
      f"{'GPUs':>5}  {'kW':>6}  {'tok/W':>8}  {'vs H100 Homo':>13}")
print("-" * 85)

for wname, cdf, B, gamma in WORKLOADS:
    alpha  = _cdf_eval(cdf, B)
    alpha2 = min(1.0, _cdf_eval(cdf, int(gamma * B)))
    sc,  lc,  _ = _split_cdf(cdf, B)
    sc2, lc2, _ = _split_cdf(cdf, int(gamma * B))

    results = {}
    for gname, gpu in GPUS:
        homo = fleet_tpw_analysis(
            pools=[dict(gpu=gpu, cdf=cdf, lam=LAM, max_ctx=65536, label="homo")],
            lam_total=LAM, t_slo_ms=SLO)
        pool = fleet_tpw_analysis(
            pools=[dict(gpu=gpu, cdf=sc,  lam=alpha*LAM,  max_ctx=B,     label="s"),
                   dict(gpu=gpu, cdf=lc,  lam=(1-alpha)*LAM, max_ctx=65536, label="l")],
            lam_total=LAM, t_slo_ms=SLO)
        fo   = fleet_tpw_analysis(
            pools=[dict(gpu=gpu, cdf=sc2, lam=alpha2*LAM, max_ctx=B,     label="s"),
                   dict(gpu=gpu, cdf=lc2, lam=(1-alpha2)*LAM, max_ctx=65536, label="l")],
            lam_total=LAM, t_slo_ms=SLO)
        results[(gname, "Homo")]     = homo
        results[(gname, "Pool")]     = pool
        results[(gname, f"FleetOpt(γ={gamma})")] = fo

    base = results[("H100", "Homo")].fleet_tpw

    for topo in ["Homo", "Pool", f"FleetOpt(γ={gamma})"]:
        for gname, _ in GPUS:
            r = results[(gname, topo)]
            d = (r.fleet_tpw - base) / base * 100
            vs = "---" if abs(d) < 0.5 else (f"+{d:.0f}%" if d > 0 else f"{d:.0f}%")
            print(f"{wname:8}  {topo:30}  {gname:6}  "
                  f"{r.total_gpus:>5}  {r.fleet_power_kw:>6.1f}  "
                  f"{r.fleet_tpw:>8.4f}  {vs:>13}")
    print()

print(f"Short-context fraction: Azure ≤4K = {_cdf_eval(_load_cdf('azure_cdf.json'), 4096):.1%}, "
      f"LMSYS ≤1.5K = {_cdf_eval(_load_cdf('lmsys_multiturn_cdf.json'), 1536):.1%}")
print("'GPUs' = number of TP=8 serving instances (×8 physical GPUs each).")
print(f"B200 power mode: {B200_POWER_MODE} ({B200_PROFILE_QUALITY}).")
print("H100 results are HIGH quality.")
