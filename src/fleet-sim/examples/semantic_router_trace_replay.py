"""Replay a pre-labeled trace from the semantic router.

The semantic router (e.g. vLLM semantic router with RouterDC, AutoMix,
complexity classifier, or embedding-based selection) logs per-request
routing decisions.  This example shows how to:

  1. Load that trace into the simulator (preserving model_id from the
     router's decision log).
  2. Replay it with ModelRouter so the simulation honours the router's
     choices exactly — no re-classification needed.
  3. Compare fleet sizing under two routing policies by swapping the
     replay trace.

Trace format expected (JSONL, one record per request)::

    {"timestamp": 0.0,   "prompt_tokens": 512,  "generated_tokens": 64,
     "selected_model": "llama70b", "complexity": "high", "category": "reasoning"}
    {"timestamp": 0.005, "prompt_tokens": 128,  "generated_tokens": 32,
     "selected_model": "llama8b",  "complexity": "low",  "category": "factual"}

The ``selected_model`` field is the routing decision from the semantic router.
Adapt ``model_id_field`` to match your router's log field name:
  - ``"selected_model"``         (default)
  - ``"model"``                  (OpenAI-style body)
  - ``"x_vsr_selected_model"``   (vLLM semantic router header, lowercased)
  - ``"routed_to"``

Usage::

    cd src/fleet-sim
    python3 examples/semantic_router_trace_replay.py            # synthetic demo
    python3 examples/semantic_router_trace_replay.py my_trace.jsonl  # real trace

"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fleet_sim import A10G, A100_80GB
from fleet_sim.core.fleet import Fleet, FleetConfig, PoolConfig
from fleet_sim.workload.trace import TraceWorkload

SLO_MS = 500
MODEL_ID_ARG_INDEX = 2

POOLS = [
    PoolConfig("llama70b", A100_80GB, 20, 8192),
    PoolConfig("llama8b", A10G, 8, 4096),
]


def make_synthetic_trace(
    path: str,
    n: int = 10_000,
    lam: float = 200,
    frac_large: float = 0.35,
    seed: int = 42,
):
    """Generate a synthetic semantic-router trace for demo purposes.

    Simulates a complexity classifier that routes ~35% of requests to the
    large model (complex / long prompts) and ~65% to the small model.
    """
    rng = random.Random(seed)
    t = 0.0
    records = []
    for _i in range(n):
        t += rng.expovariate(lam)
        is_complex = rng.random() < frac_large

        # Complex requests: longer prompts, longer replies
        if is_complex:
            l_in = rng.randint(1024, 6144)
            l_out = rng.randint(256, 1024)
            model = "llama70b"
            complexity = "high"
            category = rng.choice(["reasoning", "coding", "analysis"])
        else:
            l_in = rng.randint(64, 1024)
            l_out = rng.randint(32, 256)
            model = "llama8b"
            complexity = "low"
            category = rng.choice(["factual", "summarization", "translation"])

        records.append(
            {
                "timestamp": round(t, 6),
                "prompt_tokens": l_in,
                "generated_tokens": l_out,
                "selected_model": model,  # <-- router's decision
                "complexity": complexity,
                "category": category,
            }
        )

    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"  Wrote {n} records to {path}")
    large_pct = 100 * sum(1 for r in records if r["selected_model"] == "llama70b") / n
    print(f"  Routing split: {large_pct:.1f}% llama70b / {100-large_pct:.1f}% llama8b")


def run_replay(trace_path: str, model_id_field: str = "selected_model"):
    """Load a pre-labeled trace and replay it with ModelRouter."""

    wl = TraceWorkload(
        path=trace_path,
        fmt="semantic_router",
        model_id_field=model_id_field,  # match your router's log field
    )
    arrivals = wl.generate()

    # Count routing split from trace
    split: dict = {}
    for _, req in arrivals:
        split[req.model_id or "unknown"] = split.get(req.model_id or "unknown", 0) + 1

    fc = FleetConfig(
        pools=list(POOLS),
        router_type="ModelRouter",  # respects request.model_id
    )
    fleet = Fleet(fc)
    result = fleet.run(arrivals)

    total_gpus = sum(p.n_gpus for p in POOLS)
    cost = result.annualised_cost_usd() / 1000
    p99 = result.p99_ttft_ms()
    slo = result.slo_compliance(SLO_MS) * 100

    print(f"\n  Trace: {trace_path}  ({len(arrivals):,} requests)")
    print(
        "  Routing split: " + " | ".join(f"{m}={n}" for m, n in sorted(split.items()))
    )
    print(
        f"  Fleet: {' + '.join(f'{p.n_gpus}x {p.gpu.name} ({p.pool_id})' for p in POOLS)}"
    )
    print(
        f"  Total GPUs={total_gpus}  Cost=${cost:.0f}K/yr  "
        f"P99={p99:.1f}ms  SLO={slo:.1f}%"
    )

    for pool_id in [p.pool_id for p in POOLS]:
        pp99 = result.p99_ttft_ms(pool_id)
        pslo = result.slo_compliance(SLO_MS, pool_id) * 100
        putil = result.mean_utilisation(pool_id) * 100
        print(
            f"    {pool_id:12s}  P99={pp99:>7.1f}ms  SLO={pslo:>5.1f}%  Util={putil:>4.1f}%"
        )


def main():
    if len(sys.argv) > 1:
        # Real trace provided on command line
        trace_path = sys.argv[1]
        model_id_field = (
            sys.argv[MODEL_ID_ARG_INDEX]
            if len(sys.argv) > MODEL_ID_ARG_INDEX
            else "selected_model"
        )
        print(f"\nReplaying pre-labeled trace: {trace_path}")
        print(f"  model_id field: '{model_id_field}'")
        run_replay(trace_path, model_id_field=model_id_field)
    else:
        # Generate and replay a synthetic demo trace
        print("\nNo trace file provided - generating synthetic demo trace.")
        print(
            "Usage: python3 semantic_router_trace_replay.py <trace.jsonl> [model_id_field]"
        )
        print("\nField name aliases for common routers:")
        print("  vLLM semantic router header : 'x_vsr_selected_model'")
        print("  OpenAI-style body           : 'model'")
        print("  Generic                     : 'selected_model' (default), 'routed_to'")

        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="w", delete=False
        ) as tmp:
            tmp_path = tmp.name

        print("\n--- Demo: complexity-based routing (35% large / 65% small) ---")
        make_synthetic_trace(tmp_path, n=20_000, frac_large=0.35)
        run_replay(tmp_path)

        print("\n--- Demo: aggressive routing (70% large / 30% small) ---")
        make_synthetic_trace(tmp_path, n=20_000, frac_large=0.70)
        run_replay(tmp_path)


if __name__ == "__main__":
    main()
