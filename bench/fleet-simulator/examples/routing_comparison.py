#!/usr/bin/env python3
"""Compare routing algorithms on real LLM inference workloads.

Evaluates four routing strategies on a fixed fleet sized for pool routing:
  1. Homogeneous     — single pool, all requests routed to long pool
  2. Pool routing    — length-based split at B_short; no compression
  3. C&R (γ=1.5)     — Compress-and-Route; borderline requests compressed
  4. Random          — uniform random assignment (lower-bound baseline)

The fleet is first sized by the analytical optimizer under pool routing
(γ=1.0), then held fixed across all routers so we compare routing quality
independent of fleet size.

Usage
-----
    cd inference-fleet-sim
    python3 examples/routing_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_sim.core.fleet import Fleet, FleetConfig, PoolConfig
from fleet_sim.gpu_profiles.profiles import A100_80GB
from fleet_sim.optimizer import FleetOptimizer
from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

DATA_DIR = Path(__file__).parent.parent / "data"

WORKLOADS = {
    "Azure":       DATA_DIR / "azure_cdf.json",
    "LMSYS-MT":    DATA_DIR / "lmsys_multiturn_cdf.json",
    "Agent-Heavy": DATA_DIR / "agent_heavy_cdf.json",
}

B_SHORT  = 8192
GAMMA    = 1.5
LAM      = 200.0
N_REQ    = 30_000
T_SLO_MS = 500.0
SEED     = 42


def load_cdf(path: Path) -> list:
    raw = json.load(open(path))
    cdf = raw["cdf"] if isinstance(raw, dict) else raw
    return [(int(t), float(f)) for t, f in cdf]


def run_routing(cdf, lam, n_s, n_l, b_short, router_type, router_kwargs,
                n_req=N_REQ, seed=SEED):
    pool_configs = [
        PoolConfig("short", A100_80GB, n_s, b_short),
        PoolConfig("long",  A100_80GB, n_l, 65536),
    ]
    fc = FleetConfig(pools=pool_configs, router_type=router_type,
                     router_kwargs=router_kwargs)
    wl_gen = CdfWorkload(cdf, seed=seed)
    workload = PoissonWorkload(lam, wl_gen, n_requests=n_req, seed=seed)
    fleet = Fleet(fc)
    result = fleet.run(workload.generate())
    return result


def main():
    print(f"\n{'='*70}")
    print(f"  Routing algorithm comparison")
    print(f"  λ={LAM:.0f} req/s  B_short={B_SHORT:,}  SLO={T_SLO_MS:.0f}ms  N={N_REQ:,} reqs")
    print(f"{'='*70}")

    for wl_name, cdf_path in WORKLOADS.items():
        if not cdf_path.exists():
            print(f"\n  [{wl_name}] Not found: {cdf_path}")
            continue

        cdf = load_cdf(cdf_path)
        print(f"\n  Workload: {wl_name}")

        # Size fleet using pool routing (γ=1.0) as the neutral baseline
        opt = FleetOptimizer(gpu_short=A100_80GB, gpu_long=A100_80GB,
                             B_short=B_SHORT, t_slo_ms=T_SLO_MS)
        res = opt.sweep_analytical(cdf, LAM, gammas=[1.0], verbose=False)
        best = res[0]
        n_s, n_l = best.n_s, max(1, best.n_l)
        print(f"  Fleet (pool routing baseline): n_s={n_s}  n_l={n_l}  total={n_s+n_l}")

        routers = [
            ("Homogeneous (all long pool)",  "LengthRouter",
             {"threshold": 99_999_999}),
            ("Pool routing (no C&R)",         "LengthRouter",
             {"threshold": B_SHORT}),
            (f"C&R (γ={GAMMA})",              "CompressAndRouteRouter",
             {"B_short": B_SHORT, "gamma": GAMMA}),
            ("Random",                        "RandomRouter", {}),
        ]

        print(f"\n  {'Router':32s} {'P99 TTFT':>10} {'P50 TTFT':>10}"
              f" {'SLO%':>8} {'Util':>7}")
        print(f"  {'-'*70}")

        for rname, rtype, rkwargs in routers:
            result = run_routing(cdf, LAM, n_s, n_l, B_SHORT, rtype, rkwargs)
            p99  = result.p99_ttft_ms()
            p50  = result.p50_ttft_ms()
            slo  = result.slo_compliance(T_SLO_MS) * 100
            util = result.mean_utilisation() * 100
            print(f"  {rname:32s} {p99:>9.1f}ms {p50:>9.1f}ms"
                  f" {slo:>7.1f}% {util:>6.1f}%")

        print()


if __name__ == "__main__":
    main()
