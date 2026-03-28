#!/usr/bin/env python3
"""Validate FleetOptimizer analytical results against DES simulation.

Runs the two-phase optimizer (analytical sweep + DES verification) on
three representative workload archetypes:
  - Azure          : Archetype I  (concentrated-below CDF)
  - LMSYS-MT       : Archetype I/II boundary
  - Agent-Heavy    : Archetype II (dispersed CDF)

For each workload, prints:
  - The analytically-recommended fleet (n_s, n_l, gamma*, cost)
  - The DES-verified fleet and P99 TTFT per pool
  - The gamma sweep showing how fleet cost varies with compression bandwidth

Usage
-----
    cd src/fleet-sim
    python3 examples/optimizer_validation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_sim.gpu_profiles.profiles import A100_80GB
from fleet_sim.optimizer import FleetOptimizer

DATA_DIR = Path(__file__).parent.parent / "data"

WORKLOADS = {
    "Azure": DATA_DIR / "azure_cdf.json",
    "LMSYS-MT": DATA_DIR / "lmsys_multiturn_cdf.json",
    "Agent-Heavy": DATA_DIR / "agent_heavy_cdf.json",
}

# Fleet sizing parameters
B_SHORT = 8192  # pool boundary (tokens)
T_SLO_MS = 500.0  # P99 TTFT target (ms)
LAM = 200.0  # arrival rate (req/s)
N_SIM_REQ = 30_000  # DES requests per verification run


def load_cdf(path: Path) -> list:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    cdf = raw["cdf"] if isinstance(raw, dict) else raw
    return [(int(t), float(f)) for t, f in cdf]


def main():
    print(f"\n{'='*70}")
    print("  FleetOptimizer validation: analytical model vs DES")
    print(
        f"  λ={LAM:.0f} req/s  B_short={B_SHORT:,} tokens  SLO={T_SLO_MS:.0f}ms P99 TTFT"
    )
    print(f"{'='*70}")

    for name, cdf_path in WORKLOADS.items():
        if not cdf_path.exists():
            print(f"\n  [{name}] CDF not found: {cdf_path}")
            continue

        cdf = load_cdf(cdf_path)
        print(f"\n  Workload: {name}  ({cdf_path.name})")

        opt = FleetOptimizer(
            gpu_short=A100_80GB,
            gpu_long=A100_80GB,
            B_short=B_SHORT,
            t_slo_ms=T_SLO_MS,
            long_max_ctx=65536,
        )

        gammas = [round(1.0 + 0.1 * k, 1) for k in range(11)]
        report = opt.optimize(
            cdf=cdf,
            lam=LAM,
            gammas=gammas,
            n_sim_requests=N_SIM_REQ,
            verify_top_n=3,
            verbose=True,
        )
        report.print_report()

        # Per-gamma comparison table.
        print("\n  gamma sweep (baseline = gamma=1.0 pool routing):")
        baseline = next(
            (r for r in report.analytical if r.gamma == 1.0), report.analytical[0]
        )

        print(
            f"  {'gamma':>5} {'n_s':>5} {'n_l':>5} {'total':>7} {'saving':>8}"
            f"  {'Anal P99 s/l':>16}  {'DES P99 s/l':>16}"
        )
        print(f"  {'-'*70}")

        for sr in sorted(report.analytical, key=lambda r: r.gamma):
            sav = (baseline.cost_per_hr - sr.cost_per_hr) / baseline.cost_per_hr * 100
            src = next((s for s in report.simulated if s.gamma == sr.gamma), None)
            anal_str = f"{sr.p99_ttft_short_ms:.0f}/{sr.p99_ttft_long_ms:.0f}ms"
            des_str = (
                f"{src.p99_ttft_short_ms:.0f}/{src.p99_ttft_long_ms:.0f}ms"
                if src
                else "n/a"
            )
            anal_ok = "OK" if sr.slo_met else "  "
            des_ok = ("OK" if src and src.slo_met else "  ") if src else ""
            print(
                f"  {sr.gamma:>5.1f} {sr.n_s:>5} {sr.n_l:>5} {sr.total_gpus:>7}"
                f"  {sav:>+6.1f}%"
                f"  {anal_str:>14} {anal_ok}"
                f"  {des_str:>14} {des_ok}"
            )


if __name__ == "__main__":
    main()
