#!/usr/bin/env python3
"""What-if analysis: heterogeneous GPU cost scenarios.

Demonstrates how inference-fleet-sim enables fleet operators to explore
cost-saving scenarios by combining:
  1. Heterogeneous GPU pools (A100 short + cheaper A10G long context)
  2. C&R compression to shift traffic
  3. Arrival rate scaling to model fleet growth

Usage
-----
    cd inference-fleet-sim
    python3 examples/what_if.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_sim.gpu_profiles.profiles import A100_80GB, A10G
from fleet_sim.optimizer import FleetOptimizer

DATA_DIR = Path(__file__).parent.parent / "data"
CDF_PATH = DATA_DIR / "azure_cdf.json"


def load_cdf(path: Path) -> list:
    raw = json.load(open(path))
    cdf = raw["cdf"] if isinstance(raw, dict) else raw
    return [(int(t), float(f)) for t, f in cdf]


def main():
    if not CDF_PATH.exists():
        print(f"CDF not found: {CDF_PATH}")
        return

    cdf = load_cdf(CDF_PATH)

    print(f"\n{'='*60}")
    print(f"  What-If: Heterogeneous GPU cost (A100 short + A10G long)")
    print(f"{'='*60}")

    # ── Scenario 1: Homogeneous A100s ────────────────────────────────────────
    print(f"\n  Scenario 1: All A100-80GB (homogeneous baseline)")
    opt_hom = FleetOptimizer(gpu_short=A100_80GB, gpu_long=A100_80GB,
                             B_short=6144, t_slo_ms=500, long_max_ctx=65536)
    hom = opt_hom.sweep_analytical(cdf, 200, gammas=[1.0], verbose=False)[0]
    print(f"  γ=1.0  n_s={hom.n_s}  n_l={hom.n_l}  "
          f"total={hom.total_gpus}  ${hom.annualised_cost_kusd:.1f}K/yr")

    # ── Scenario 2: A100 short + A10G long ───────────────────────────────────
    print(f"\n  Scenario 2: A100 short + A10G long (A10G @ ${A10G.cost_per_hr:.2f}/hr)")
    opt_het = FleetOptimizer(gpu_short=A100_80GB, gpu_long=A10G,
                             B_short=6144, t_slo_ms=500, long_max_ctx=65536)
    gammas = [round(1.0 + 0.1 * k, 1) for k in range(11)]
    het = opt_het.sweep_analytical(cdf, 200, gammas=gammas, verbose=True)
    best_het = min((r for r in het if r.slo_met), key=lambda r: r.cost_per_hr,
                   default=het[0])
    print(f"\n  Best: γ={best_het.gamma}  n_s={best_het.n_s}  n_l={best_het.n_l}  "
          f"total={best_het.total_gpus}  ${best_het.annualised_cost_kusd:.1f}K/yr")
    sav = (hom.cost_per_hr - best_het.cost_per_hr) / hom.cost_per_hr * 100
    print(f"  Savings vs homogeneous: {sav:+.1f}%")

    # ── Scenario 3: Arrival rate sweep ───────────────────────────────────────
    print(f"\n\n  Scenario 3: Arrival rate scaling (λ = 50..2000 req/s)")
    print(f"  {'λ':>8} {'A100 total':>12} {'A100 $/yr':>12}"
          f"  {'Hetero total':>14} {'Hetero $/yr':>12} {'Δ saving':>10}")
    print(f"  {'-'*65}")

    for lam in [50, 100, 200, 500, 1000, 2000]:
        r_hom = opt_hom.sweep_analytical(cdf, lam, gammas=[1.0], verbose=False)[0]
        r_het_all = opt_het.sweep_analytical(cdf, lam, gammas=gammas, verbose=False)
        r_het = min((r for r in r_het_all if r.slo_met),
                    key=lambda r: r.cost_per_hr, default=r_het_all[0])
        sav = (r_hom.cost_per_hr - r_het.cost_per_hr) / r_hom.cost_per_hr * 100
        print(f"  {lam:>8.0f}  {r_hom.total_gpus:>10}  ${r_hom.annualised_cost_kusd:>9.1f}K"
              f"  {r_het.total_gpus:>12}  ${r_het.annualised_cost_kusd:>9.1f}K"
              f"  {sav:>+9.1f}%")


if __name__ == "__main__":
    main()
