"""Semantic-classification-based model selection example.

Shows how to use SemanticRouter to simulate a fleet where requests are
dispatched to large or small models based on semantic content, mirroring
how a production semantic router (e.g., vLLM's semantic routing integration)
would operate.

Fleet layout
------------
  llama70b  – 20× A100-80GB  (complex queries, reasoning, long-form)
  llama8b   –  8× A10G       (simple queries, factual lookup, short answers)

Routing policy
--------------
  Three example policies are compared:
  1. Oracle length heuristic  – route short requests (<= 2K tokens) to llama8b
  2. Fixed fraction           – route 60% to llama70b, 40% to llama8b (random)
  3. All large model          – homogeneous llama70b fleet (baseline cost)

Usage
-----
  cd inference-fleet-sim
  python3 examples/semantic_routing.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_sim import A100_80GB, A10G
from fleet_sim.core.fleet import Fleet, FleetConfig, PoolConfig
from fleet_sim.core.request import Request
from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload
from fleet_sim.routing.semantic_router import SemanticRouter

DATA_DIR = Path(__file__).parent.parent / "data"
LAM = 200          # total arrival rate (req/s)
SLO_MS = 500       # P99 TTFT SLO (ms)
N_REQ = 20_000
SEED = 42

POOLS = [
    PoolConfig("llama70b", A100_80GB, 20, 8192),
    PoolConfig("llama8b",  A10G,      8, 4096),
]


def run(classify_fn, label: str, arrivals: list):
    fc = FleetConfig(
        pools=list(POOLS),
        router_type="SemanticRouter",
        router_kwargs={"classify_fn": classify_fn},
    )
    fleet = Fleet(fc)
    result = fleet.run(list(arrivals))

    p99  = result.p99_ttft_ms()
    slo  = result.slo_compliance(SLO_MS) * 100
    cost = result.annualised_cost_usd() / 1000
    print(f"  {label:45s}  P99={p99:>8.1f}ms  SLO={slo:>5.1f}%  ${cost:>6.0f}K/yr")


def main():
    cdf = json.load(open(DATA_DIR / "azure_cdf.json"))
    wl_gen = CdfWorkload(cdf, seed=SEED)
    arrivals = PoissonWorkload(LAM, wl_gen, n_requests=N_REQ, seed=SEED).generate()

    print(f"\nSemantic routing comparison  λ={LAM} req/s  SLO={SLO_MS}ms")
    print(f"  Fleet: {POOLS[0].n_gpus}× {POOLS[0].gpu.name} (llama70b) + "
          f"{POOLS[1].n_gpus}× {POOLS[1].gpu.name} (llama8b)")
    print(f"  {'Policy':45s}  {'P99 TTFT':>10}  {'SLO%':>6}  {'Cost':>10}")
    print(f"  {'-'*80}")

    # Policy 1: semantic length heuristic
    #   Simulates a classifier that sends short/simple requests to the small
    #   model.  In production, "short" could be replaced by a true semantic
    #   score (embedding similarity, intent classification, etc.).
    def length_heuristic(req: Request):
        return "llama8b" if req.l_total <= 2048 else "llama70b"

    run(length_heuristic, "Length heuristic (<=2K → llama8b)", arrivals)

    # Policy 2: fixed-fraction semantic split (60/40)
    #   Models a classifier that routes 60% of traffic to the large model
    #   and 40% to the small model, independent of content.
    rng = random.Random(SEED)
    def fixed_fraction(req: Request):
        return "llama70b" if rng.random() < 0.60 else "llama8b"

    run(fixed_fraction, "Fixed fraction (60% large / 40% small)", arrivals)

    # Policy 3: all-large baseline (homogeneous fleet, no semantic routing)
    #   Provides the cost baseline: what does a fleet look like if you never
    #   route to a small model?
    def all_large(req: Request):
        return "llama70b"

    run(all_large, "All large (homogeneous llama70b, no routing)", arrivals)

    print()
    print("  Note: P99 TTFT here may be dominated by prefill time, not queueing.")
    print("  Use 'optimize' per pool to find the right GPU count for each policy,")
    print("  then re-run this script with the updated POOLS.")


if __name__ == "__main__":
    main()
