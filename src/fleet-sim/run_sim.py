#!/usr/bin/env python3
"""vllm-sr-sim  — fleet-level LLM inference simulator CLI.

Usage
-----
  # Optimize fleet for a workload CDF (two-pool, length-based routing)
  vllm-sr-sim optimize \\
      --cdf data/azure_cdf.json \\
      --lam 200 --slo 500 --b-short 6144

  # Simulate a fixed two-pool fleet
  vllm-sr-sim simulate \\
      --cdf data/azure_cdf.json \\
      --lam 200 --n-s 40 --n-l 20 --b-short 6144 --n-req 50000

  # What-if: sweep arrival rate (single GPU type)
  vllm-sr-sim whatif \\
      --cdf data/azure_cdf.json \\
      --lam-range 50 100 200 500 1000 \\
      --slo 500 --b-short 6144

  # What-if: compare GPU types at fixed arrival rate
  vllm-sr-sim whatif \\
      --cdf data/azure_cdf.json \\
      --lam-range 200 --slo 500 --b-short 6144 \\
      --gpu-compare a100 h100 a10g

  # Compare routing algorithms on a workload
  vllm-sr-sim compare-routers \\
      --cdf data/azure_cdf.json \\
      --lam 200 --n-s 50 --n-l 10 --b-short 6144

  # Pareto sweep: find the best B_short threshold from the CDF
  vllm-sr-sim pareto \\
      --cdf data/lmsys_cdf.json \\
      --lam 200 --slo 500

  # Disaggregated prefill/decode fleet: find optimal nP × nD ratio
  vllm-sr-sim disagg \\
      --cdf data/azure_cdf.json \\
      --lam 200 --slo-ttft 500 --slo-tpot 100 \\
      --gpu-prefill h100 --gpu-decode a100 --max-ctx 8192

  # Grid flexibility: power–latency trade-off under demand response (GPU-to-Grid)
  vllm-sr-sim grid-flex \\
      --cdf data/azure_cdf.json \\
      --lam 200 --slo 500 --n-gpus 40 --gpu h100

  # Simulate an arbitrary N-pool fleet (multi-model or heterogeneous)
  vllm-sr-sim simulate-fleet fleet.json \\
      --lam 200 --slo 500 --n-req 30000

  # fleet.json format (see examples/multi_model_fleet.json for a full example):
  # {
  #   "pools": [
  #     {"id": "llama70b", "gpu": "a100", "n_gpus": 20, "max_ctx": 8192},
  #     {"id": "llama8b",  "gpu": "a10g", "n_gpus": 8,  "max_ctx": 4096}
  #   ],
  #   "router": "model",          // "length" | "model" | "random" | "least_loaded"
  #   "workloads": {               // per-pool arrival streams (for model router)
  #     "llama70b": {"cdf": "data/azure_cdf.json",      "lam_frac": 0.7},
  #     "llama8b":  {"cdf": "data/lmsys_cdf.json",       "lam_frac": 0.3}
  #   }
  # }
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fleet_sim import __version__ as FLEET_SIM_VERSION
from fleet_sim.core.fleet import Fleet, FleetConfig, PoolConfig
from fleet_sim.gpu_profiles.profiles import A10G, A100_80GB, H100_80GB
from fleet_sim.optimizer import (
    FleetOptimizer,
    print_threshold_pareto,
    threshold_pareto,
)
from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

GPU_REGISTRY = {"a100": A100_80GB, "h100": H100_80GB, "a10g": A10G}


def load_cdf(path: str) -> list:
    raw = json.load(open(path))
    cdf = raw["cdf"] if isinstance(raw, dict) else raw
    return [(int(t), float(f)) for t, f in cdf]


# ── subcommands ───────────────────────────────────────────────────────────────


def cmd_optimize(args):
    cdf = load_cdf(args.cdf)
    gpu_s = GPU_REGISTRY.get(args.gpu_short, A100_80GB)
    gpu_l = GPU_REGISTRY.get(args.gpu_long, A100_80GB)

    opt = FleetOptimizer(
        gpu_short=gpu_s,
        gpu_long=gpu_l,
        B_short=args.b_short,
        t_slo_ms=args.slo,
        long_max_ctx=args.long_max_ctx,
    )
    gammas = [
        round(1.0 + 0.1 * k, 1) for k in range(int((args.gamma_max - 1.0) / 0.1) + 1)
    ]
    report = opt.optimize(
        cdf=cdf,
        lam=args.lam,
        gammas=gammas,
        n_sim_requests=args.n_sim_req,
        verify_top_n=args.verify_top,
        verbose=True,
    )
    report.print_report()

    if args.out:
        rows = [
            {
                "gamma": r.gamma,
                "n_s": r.n_s,
                "n_l": r.n_l,
                "total_gpus": r.total_gpus,
                "annualised_cost_kusd": round(r.annualised_cost_kusd, 1),
                "p99_ttft_short_ms": round(r.p99_ttft_short_ms, 1),
                "p99_ttft_long_ms": round(r.p99_ttft_long_ms, 1),
                "slo_met": r.slo_met,
                "source": r.source,
            }
            for r in report.analytical + report.simulated
        ]
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\nResults saved to {args.out}")


def cmd_simulate(args):
    cdf = load_cdf(args.cdf)
    gpu_s = GPU_REGISTRY.get(args.gpu_short, A100_80GB)
    gpu_l = GPU_REGISTRY.get(args.gpu_long, A100_80GB)

    pool_configs = [
        PoolConfig("short", gpu_s, args.n_s, args.b_short),
        PoolConfig("long", gpu_l, args.n_l, args.long_max_ctx),
    ]
    fc = FleetConfig(
        pools=pool_configs,
        router_type="CompressAndRouteRouter" if args.gamma > 1.0 else "LengthRouter",
        router_kwargs=(
            {"B_short": args.b_short, "gamma": args.gamma}
            if args.gamma > 1.0
            else {"threshold": args.b_short}
        ),
    )

    wl_gen = CdfWorkload(cdf, seed=args.seed)
    workload = PoissonWorkload(args.lam, wl_gen, n_requests=args.n_req, seed=args.seed)
    arrivals = workload.generate()

    print(f"Simulating {args.n_req:,} requests at λ={args.lam:.0f} req/s ...")
    fleet = Fleet(fc)
    result = fleet.run(arrivals, verbose=True)
    result.print_summary(t_slo_ms=args.slo)

    if args.out:
        json.dump(result.summary(args.slo), open(args.out, "w"), indent=2)
        print(f"\nResults saved to {args.out}")


def _whatif_single(cdf, lam_range, gpu_s, gpu_l, b_short, slo, long_max_ctx):
    """Return list of best SweepResult per arrival rate for one GPU type pair."""
    opt = FleetOptimizer(
        gpu_short=gpu_s,
        gpu_long=gpu_l,
        B_short=b_short,
        t_slo_ms=slo,
        long_max_ctx=long_max_ctx,
    )
    rows = []
    for lam in lam_range:
        res = opt.sweep_analytical(cdf, lam, verbose=False)
        best = min(
            (r for r in res if r.slo_met), key=lambda r: r.cost_per_hr, default=res[0]
        )
        rows.append((lam, best))
    return rows


def cmd_whatif(args):
    cdf = load_cdf(args.cdf)

    # ── GPU-type comparison mode ──────────────────────────────────────────────
    gpu_names = getattr(args, "gpu_compare", None) or []
    if gpu_names:
        gpus = [(name, GPU_REGISTRY.get(name, A100_80GB)) for name in gpu_names]
        print("\nWhat-if: GPU type comparison")
        print(
            f"  CDF: {args.cdf}  B_short={args.b_short:,}  SLO={args.slo}ms"
            f"  long_max_ctx={args.long_max_ctx:,}"
        )

        # Build results per GPU
        all_results = {}
        for name, gpu in gpus:
            all_results[name] = _whatif_single(
                cdf,
                args.lam_range,
                gpu,
                gpu,
                args.b_short,
                args.slo,
                args.long_max_ctx,
            )

        # Print side-by-side for each λ
        col_w = 28
        header_gpu = "".join(f"  {n.upper():^{col_w}}" for n, _ in gpus)
        print(f"\n  {'λ':>8}  {header_gpu}")
        sub_hdr = "".join(
            f"  {'GPUs':>5} {'$K/yr':>8} {'$/hr':>7} {'P99':>7} {'SLO':>4}"
            for _ in gpus
        )
        print(f"  {' ':>8}  {sub_hdr}")
        print(f"  {'-' * (10 + len(gpus) * (col_w + 2))}")

        all_rows = []
        for lam in args.lam_range:
            line = f"  {lam:>8.0f}  "
            row = {"lam": lam}
            for name, gpu in gpus:
                lam_rows = {r[0]: r[1] for r in all_results[name]}
                best = lam_rows.get(lam)
                if best:
                    ok = "✓" if best.slo_met else "✗"
                    cost_hr = best.cost_per_hr
                    line += (
                        f"  {best.total_gpus:>5} ${best.annualised_cost_kusd:>6.0f}K"
                        f" ${cost_hr:>5.0f}/hr"
                        f" {best.p99_ttft_short_ms:>5.0f}ms {ok:>4}"
                    )
                    row[name] = {
                        "total_gpus": best.total_gpus,
                        "cost_per_hr": round(cost_hr, 2),
                        "annualised_cost_kusd": round(best.annualised_cost_kusd, 1),
                        "p99_ttft_short_ms": round(best.p99_ttft_short_ms, 1),
                        "slo_met": best.slo_met,
                    }
                else:
                    line += f"  {'N/A':>{col_w}}"
            print(line)
            all_rows.append(row)

        # Print cost ratio vs first GPU
        if len(gpus) > 1:
            base_name = gpus[0][0]
            print(f"\n  Cost ratio vs {base_name.upper()} (at each λ):")
            for lam in args.lam_range:
                lam_rows = {r[0]: r[1] for r in all_results[base_name]}
                base = lam_rows.get(lam)
                if not base:
                    continue
                line = f"  {lam:>8.0f}  "
                for name, gpu in gpus:
                    if name == base_name:
                        line += f"  {'1.00×':>10}"
                        continue
                    other_rows = {r[0]: r[1] for r in all_results[name]}
                    other = other_rows.get(lam)
                    if other:
                        ratio = other.annualised_cost_kusd / base.annualised_cost_kusd
                        gpu_ratio = other.total_gpus / base.total_gpus
                        line += f"  {ratio:>5.2f}× cost" f" ({gpu_ratio:+.0%} GPUs)"
                    else:
                        line += f"  {'N/A':>10}"
                print(line)

        if args.out:
            json.dump(all_rows, open(args.out, "w"), indent=2)
            print(f"\nResults saved to {args.out}")
        return

    # ── Single GPU type, λ sweep (original behaviour) ─────────────────────────
    gpu_s = GPU_REGISTRY.get(args.gpu_short, A100_80GB)
    gpu_l = GPU_REGISTRY.get(args.gpu_long, A100_80GB)

    print(f"\nWhat-if analysis: sweep λ over {args.lam_range}")
    print(
        f"  B_short={args.b_short:,}  SLO={args.slo}ms  GPU: {gpu_s.name}/{gpu_l.name}"
    )
    print(
        f"\n  {'λ':>8} {'n_s':>5} {'n_l':>5} {'total':>7} {'$K/yr':>10}"
        f" {'γ*':>6} {'P99_s':>8} {'P99_l':>8}"
    )
    print(f"  {'-'*60}")

    rows = []
    for lam, best in _whatif_single(
        cdf, args.lam_range, gpu_s, gpu_l, args.b_short, args.slo, args.long_max_ctx
    ):
        print(
            f"  {lam:>8.0f} {best.n_s:>5} {best.n_l:>5} {best.total_gpus:>7}"
            f" ${best.annualised_cost_kusd:>8.1f}K {best.gamma:>5.1f}"
            f" {best.p99_ttft_short_ms:>7.1f}ms {best.p99_ttft_long_ms:>7.1f}ms"
        )
        rows.append(
            {
                "lam": lam,
                "n_s": best.n_s,
                "n_l": best.n_l,
                "total_gpus": best.total_gpus,
                "annualised_cost_kusd": round(best.annualised_cost_kusd, 1),
                "gamma_opt": best.gamma,
            }
        )

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\nResults saved to {args.out}")


def cmd_pareto(args):
    """Sweep all CDF breakpoints as candidate B_short thresholds and print the
    threshold–cost–latency Pareto frontier.

    Each row shows the two-pool fleet that would result from using that CDF
    breakpoint as the short/long split.  A star (★) marks Pareto-optimal
    points: no other threshold achieves both lower cost and lower worst-case
    P99 simultaneously.
    """
    cdf = load_cdf(args.cdf)
    gpu_s = GPU_REGISTRY.get(args.gpu_short, A100_80GB)
    gpu_l = GPU_REGISTRY.get(args.gpu_long, A100_80GB)

    # Homo baseline cost for savings computation
    from fleet_sim.optimizer import FleetOptimizer as _FO

    homo_opt = _FO(
        gpu_short=gpu_s,
        gpu_long=gpu_l,
        B_short=args.long_max_ctx,
        t_slo_ms=args.slo,
        long_max_ctx=args.long_max_ctx,
    )
    homo_sweep = homo_opt.sweep_analytical(cdf, args.lam, gammas=[1.0], verbose=False)
    homo_cost = homo_sweep[0].annualised_cost_kusd if homo_sweep else None

    print("\nPareto frontier: threshold sweep")
    print(
        f"  CDF: {args.cdf}  λ={args.lam:.0f} req/s  SLO={args.slo}ms"
        f"  GPU: {gpu_s.name}/{gpu_l.name}"
    )
    if homo_cost:
        print(
            f"  Homo baseline (B_short={args.long_max_ctx:,}): "
            f"${homo_cost:.0f}K/yr  "
            f"({homo_sweep[0].total_gpus} GPUs)"
        )

    results = threshold_pareto(
        cdf=cdf,
        lam=args.lam,
        gpu_short=gpu_s,
        gpu_long=gpu_l,
        t_slo_ms=args.slo,
        long_max_ctx=args.long_max_ctx,
    )

    if not results:
        print("  No valid threshold candidates found.")
        return

    print_threshold_pareto(results, t_slo_ms=args.slo, homo_cost_kusd=homo_cost or 0.0)

    if args.out:
        import dataclasses

        rows = [dataclasses.asdict(r) for r in results]
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\nResults saved to {args.out}")


def cmd_compare_routers(args):
    """Compare routing algorithms on the same workload and fleet."""
    cdf = load_cdf(args.cdf)
    gpu = GPU_REGISTRY.get(args.gpu_short, A100_80GB)

    pool_configs = [
        PoolConfig("short", gpu, args.n_s, args.b_short),
        PoolConfig("long", gpu, args.n_l, args.long_max_ctx),
    ]

    wl_gen = CdfWorkload(cdf, seed=args.seed)
    workload = PoissonWorkload(args.lam, wl_gen, n_requests=args.n_req, seed=args.seed)
    arrivals = workload.generate()

    routers = [
        ("LengthRouter", {"threshold": args.b_short}),
        ("CompressAndRouteRouter", {"B_short": args.b_short, "gamma": 1.5}),
        ("RandomRouter", {}),
    ]

    print(
        f"\n  Router comparison  λ={args.lam:.0f} req/s  "
        f"n_s={args.n_s} n_l={args.n_l}  SLO={args.slo}ms"
    )
    print(f"  {'Router':30s} {'P99 TTFT':>10} {'SLO%':>8} {'Util':>8}")
    print(f"  {'-'*60}")

    rows = []
    for rname, rkwargs in routers:
        fc = FleetConfig(
            pools=list(pool_configs),
            router_type=rname,
            router_kwargs=rkwargs,
        )
        fleet = Fleet(fc)
        result = fleet.run(list(arrivals))
        p99 = result.p99_ttft_ms()
        slo = result.slo_compliance(args.slo) * 100
        util = result.mean_utilisation() * 100
        print(f"  {rname:30s} {p99:>9.1f}ms {slo:>7.2f}% {util:>7.1f}%")
        rows.append(
            {
                "router": rname,
                "p99_ttft_ms": round(p99, 2),
                "slo_compliance_pct": round(slo, 2),
                "mean_utilisation_pct": round(util, 2),
            }
        )

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\nResults saved to {args.out}")


def cmd_serve(args):
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "vllm-sr-sim service mode requires API extras: pip install 'vllm-sr-sim[api]'"
        ) from exc

    os.environ.setdefault("VLLM_SR_SIM_SEED_EXAMPLE_TRACES", "true")

    print("\n  vllm-sr-sim service")
    print(f"  {'─' * 28}")
    print(f"  API docs  : http://{args.host}:{args.port}/api/docs")
    print(f"  OpenAPI   : http://{args.host}:{args.port}/api/openapi.json\n")

    uvicorn.run(
        "fleet_sim.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )


def cmd_disagg(args):
    """Disaggregated prefill/decode fleet optimizer.

    Finds the optimal (nPrefill × nDecode) worker ratio for a disaggregated
    serving system where prefill (context-ingestion) and decode (token-
    generation) phases run on separate, independently-scaled GPU pools.

    Mean ISL and OSL are derived from the CDF (75 / 25 % split) unless
    explicitly overridden via --mean-isl / --mean-osl.
    """
    from fleet_sim.optimizer.disagg import DisaggFleetOptimizer

    cdf = load_cdf(args.cdf)
    gpu_pre = GPU_REGISTRY.get(args.gpu_prefill, H100_80GB)
    gpu_dec = GPU_REGISTRY.get(args.gpu_decode, A100_80GB)

    mean_total = sum(
        t * (cdf[i][1] - (cdf[i - 1][1] if i > 0 else 0.0))
        for i, (t, _) in enumerate(cdf)
    )
    mean_isl = int(args.mean_isl) if args.mean_isl else int(mean_total * 0.75)
    mean_osl = int(args.mean_osl) if args.mean_osl else int(mean_total * 0.25)

    print("\nDisaggregated prefill/decode fleet optimizer")
    print(f"  CDF: {args.cdf}  λ={args.lam:.0f} req/s")
    print(f"  Prefill GPU: {gpu_pre.name}  Decode GPU: {gpu_dec.name}")
    print(f"  mean ISL={mean_isl} tok  mean OSL={mean_osl} tok  max_ctx={args.max_ctx}")
    print(f"  SLO: TTFT={args.slo_ttft}ms  TPOT={args.slo_tpot}ms")

    opt = DisaggFleetOptimizer(
        prefill_profile=gpu_pre,
        decode_profile=gpu_dec,
        mean_isl=mean_isl,
        mean_osl=mean_osl,
        slo_ttft_ms=args.slo_ttft,
        slo_tpot_ms=args.slo_tpot,
        max_ctx=args.max_ctx,
    )

    points = opt.sweep(max_prefill=args.max_prefill, max_decode=args.max_decode)
    feasible = [p for p in points if p.slo_met and p.system_rate >= args.lam]

    if not feasible:
        max_rate = max((p.system_rate for p in points), default=0.0)
        slo_ok = any(p.slo_met for p in points)
        print(f"\n  No valid configuration found for λ={args.lam:.0f} req/s.")
        print(f"  Max achievable throughput: {max_rate:.1f} req/s")
        if not slo_ok:
            eff = opt._prefill_ttft_ms() * opt.beta_ttft
            tpot = opt._decode_tpot_ms()
            print(
                f"  Single-instance TTFT={eff:.0f}ms  TPOT={tpot:.0f}ms  "
                f"(SLOs: {args.slo_ttft}ms / {args.slo_tpot}ms)"
            )
        return

    best = min(feasible, key=lambda p: p.cost_per_hr)

    # Pareto-efficient subset: non-dominated on (cost, total_gpus)
    pareto = [
        pt
        for pt in feasible
        if not any(
            o.cost_per_hr <= pt.cost_per_hr
            and o.total_gpus <= pt.total_gpus
            and (o.cost_per_hr < pt.cost_per_hr or o.total_gpus < pt.total_gpus)
            for o in feasible
        )
    ]
    pareto.sort(key=lambda p: p.total_gpus)

    print(f"\n  Pareto-efficient configs (SLO ✓, λ ≥ {args.lam:.0f} req/s):")
    print(
        f"  {'nP':>4} {'nD':>4} {'GPUs':>6} {'rate':>8} {'TTFT':>8} {'TPOT':>8} "
        f"{'$/hr':>7} {'$K/yr':>7}"
    )
    print(f"  {'-'*60}")
    for pt in pareto:
        marker = " ★" if pt is best else ""
        print(
            f"  {pt.n_prefill:>4} {pt.n_decode:>4} {pt.total_gpus:>6}"
            f" {pt.system_rate:>7.1f}/s {pt.ttft_ms:>7.0f}ms {pt.tpot_ms:>7.0f}ms"
            f" ${pt.cost_per_hr:>5.2f}/hr ${pt.cost_per_hr * 8760 / 1000:>5.0f}K{marker}"
        )

    # Aggregated comparison
    print(
        f"\n  Comparison vs aggregated (homogeneous) fleets at λ={args.lam:.0f} req/s:"
    )
    seen = set()
    for gname in [args.gpu_prefill, args.gpu_decode]:
        if gname in seen:
            continue
        seen.add(gname)
        g = GPU_REGISTRY[gname]
        agg_opt = FleetOptimizer(
            gpu_short=g,
            gpu_long=g,
            B_short=args.max_ctx,
            t_slo_ms=args.slo_ttft,
            long_max_ctx=args.max_ctx,
        )
        agg_results = agg_opt.sweep_analytical(
            cdf, args.lam, gammas=[1.0], verbose=False
        )
        agg_ok = [r for r in agg_results if r.slo_met]
        if agg_ok:
            agg = min(agg_ok, key=lambda r: r.annualised_cost_kusd)
            print(
                f"  All-{gname.upper():6s}  {agg.total_gpus:>4} GPUs  "
                f"${agg.annualised_cost_kusd:>5.0f}K/yr  "
                f"P99 TTFT={agg.p99_ttft_short_ms:.0f}ms"
            )
        else:
            print(f"  All-{gname.upper():6s}  No SLO-feasible config found")

    disagg_yr = best.cost_per_hr * 8760 / 1000
    print(
        f"  Disagg {gpu_pre.name}P+{gpu_dec.name}D  "
        f"{best.total_gpus:>4} GPUs ({best.n_prefill}P+{best.n_decode}D)  "
        f"${disagg_yr:>5.0f}K/yr  "
        f"TTFT={best.ttft_ms:.0f}ms  TPOT={best.tpot_ms:.0f}ms"
    )

    if args.out:
        import dataclasses

        json.dump(
            {
                "best": dataclasses.asdict(best),
                "pareto": [dataclasses.asdict(p) for p in pareto],
            },
            open(args.out, "w"),
            indent=2,
        )
        print(f"\nResults saved to {args.out}")


def cmd_simulate_fleet(args):
    """Simulate an arbitrary N-pool fleet defined in a JSON config file.

    Config schema
    -------------
    {
      "pools": [
        {"id": "<name>", "gpu": "a100|h100|a10g", "n_gpus": N, "max_ctx": M},
        ...
      ],
      "router": "length" | "model" | "random" | "least_loaded",
      // For "model" router: per-pool workload streams
      "workloads": {
        "<pool_id>": {"cdf": "<path>", "lam_frac": 0.0-1.0},
        ...
      }
    }

    When router is "model" and workloads are specified, each pool receives its
    own arrival stream (requests tagged with the pool's model_id).  When router
    is "length", all traffic shares one CDF and is routed by token length.

    The --cdf and --lam flags override the config's workloads section, sending
    all traffic through a single CDF at the given arrival rate.
    """
    cfg = json.load(open(args.fleet_config))

    # Build pool list
    pool_configs = []
    for pc in cfg["pools"]:
        gpu = GPU_REGISTRY.get(pc["gpu"], A100_80GB)
        pool_configs.append(PoolConfig(pc["id"], gpu, pc["n_gpus"], pc["max_ctx"]))

    router_name = cfg.get("router", "length")
    router_map = {
        "length": "LengthRouter",
        "model": "ModelRouter",
        "semantic": "SemanticRouter",
        "random": "RandomRouter",
        "least_loaded": "LeastLoadedRouter",
    }
    router_type = router_map.get(router_name, "LengthRouter")

    fc = FleetConfig(pools=pool_configs, router_type=router_type)

    # Build arrival stream
    if args.cdf:
        # Single CDF for all pools (ignores workloads section)
        cdf = load_cdf(args.cdf)
        wl_gen = CdfWorkload(cdf, seed=args.seed)
        workload = PoissonWorkload(
            args.lam, wl_gen, n_requests=args.n_req, seed=args.seed
        )
        arrivals = workload.generate()
    elif "workloads" in cfg and router_name == "model":
        # Per-pool workloads: merge streams and tag each request with model_id
        all_pairs: list = []
        for pool_id, wspec in cfg["workloads"].items():
            pool_cdf = load_cdf(wspec["cdf"])
            lam_p = args.lam * wspec.get("lam_frac", 1.0 / len(cfg["workloads"]))
            n_p = max(1, int(args.n_req * wspec.get("lam_frac", 1.0)))
            wl_gen = CdfWorkload(pool_cdf, seed=args.seed)
            pairs = PoissonWorkload(
                lam_p, wl_gen, n_requests=n_p, seed=args.seed
            ).generate()
            for t, req in pairs:
                req.model_id = pool_id
            all_pairs.extend(pairs)
        # Sort merged stream by arrival time and reassign req_ids
        all_pairs.sort(key=lambda x: x[0])
        arrivals = [(t, req) for t, req in all_pairs]
    else:
        p = pool_configs[0]
        print(
            f"[warn] No --cdf provided and no workloads in config; "
            f"using Poisson arrivals at λ={args.lam} req/s"
        )
        workload = PoissonWorkload(
            args.lam,
            CdfWorkload([(p.max_ctx, 1.0)], seed=args.seed),
            n_requests=args.n_req,
            seed=args.seed,
        )
        arrivals = workload.generate()

    n_arrivals = len(arrivals)
    print(f"Simulating {n_arrivals:,} requests across {len(pool_configs)} pools ...")
    print(
        "  Pools: "
        + ", ".join(
            f"{pc.pool_id}({pc.n_gpus}×{pc.gpu.name},ctx={pc.max_ctx})"
            for pc in pool_configs
        )
    )
    print(f"  Router: {router_type}")

    fleet = Fleet(fc)
    result = fleet.run(arrivals, verbose=True)
    result.print_summary(t_slo_ms=args.slo)

    if args.out:
        json.dump(result.summary(args.slo), open(args.out, "w"), indent=2)
        print(f"\nResults saved to {args.out}")


def cmd_tok_per_watt(args):
    """Compare GPU energy efficiency (tokens/watt) at the SLO-optimal fleet.

    Two modes:

    1. **Single-pool comparison** (default, ``--gpus h100 a100 a10g``):
       Each GPU receives the full CDF and the full arrival rate.  Correct
       only when comparing the *same model* on different hardware.

    2. **Two-pool routing comparison** (``--b-short B --gpu-short G --gpu-long G``):
       Splits traffic at B tokens. Short requests (≤ B) go to the short-pool
       GPU; long requests go to the long-pool GPU.  The fleet-level tok/W
       uses the correct aggregate formula — N does NOT cancel across pools.
       This models hetero fleets and semantic-router short/long routing.

    WARNING: pre-built profiles (h100, a100, a10g) are calibrated for
    different models (70B for h100/a100, 7B-class for a10g).  When using
    --gpu-short a10g --gpu-long h100, the comparison implicitly includes a
    model switch (7B → 70B), which is intentional for semantic-router use
    cases but must be flagged when reporting results.
    """
    from fleet_sim.optimizer import (
        _split_cdf,
        fleet_tpw_analysis,
        print_fleet_tpw,
        print_tpw_table,
        tpw_analysis,
    )

    cdf = load_cdf(args.cdf)

    two_pool_mode = args.b_short is not None and args.gpu_short and args.gpu_long

    if two_pool_mode:
        # ── Two-pool routing comparison ────────────────────────────────────
        b_short = args.b_short
        gpu_s = GPU_REGISTRY.get(args.gpu_short, A100_80GB)
        gpu_l = GPU_REGISTRY.get(args.gpu_long, A100_80GB)

        short_cdf, long_cdf, alpha = _split_cdf(cdf, b_short)
        lam_s = alpha * args.lam
        lam_l = (1.0 - alpha) * args.lam

        print(f"\nTwo-pool routing split at B_short={b_short} tokens:")
        print(f"  Short pool ({gpu_s.name}): α={alpha:.3f}, λ_s={lam_s:.1f} req/s")
        print(f"  Long  pool ({gpu_l.name}): 1-α={1-alpha:.3f}, λ_l={lam_l:.1f} req/s")
        if gpu_s.name != gpu_l.name:
            print(
                "\n  NOTE: different GPU types likely represent different model sizes."
            )
            print("  A10G is calibrated for 7B-class models; H100/A100 for 70B.")
            print("  Fleet tok/W reflects combined model+hardware efficiency.")

        homo_result = fleet_tpw_analysis(
            pools=[
                {
                    "gpu": gpu_l,
                    "cdf": cdf,
                    "lam": args.lam,
                    "max_ctx": args.max_ctx,
                    "label": f"homo-{gpu_l.name}",
                }
            ],
            lam_total=args.lam,
            t_slo_ms=args.slo,
            topology=f"homo {gpu_l.name} (baseline)",
        )

        routed_result = fleet_tpw_analysis(
            pools=[
                {
                    "gpu": gpu_s,
                    "cdf": short_cdf,
                    "lam": lam_s,
                    "max_ctx": args.max_ctx,
                    "label": f"short({b_short}T)-{gpu_s.name}",
                },
                {
                    "gpu": gpu_l,
                    "cdf": long_cdf,
                    "lam": lam_l,
                    "max_ctx": args.max_ctx,
                    "label": f"long-{gpu_l.name}",
                },
            ],
            lam_total=args.lam,
            t_slo_ms=args.slo,
            topology=f"two-pool {gpu_s.name}(short) + {gpu_l.name}(long)",
        )

        print(
            f"\nTokens-per-Watt fleet comparison  "
            f"(CDF: {args.cdf},  λ={args.lam} req/s,  SLO={args.slo}ms)"
        )
        print_fleet_tpw(homo_result, title="── Baseline: homo pool ──")
        print_fleet_tpw(routed_result, title="── Routed:  two-pool ──")

        delta_tpw = (routed_result.fleet_tpw / homo_result.fleet_tpw - 1) * 100
        delta_cost = (
            routed_result.fleet_cost_per_mil / homo_result.fleet_cost_per_mil - 1
        ) * 100
        delta_gpus = routed_result.total_gpus - homo_result.total_gpus
        print("\n  Routing benefit vs homo baseline:")
        sign = "+" if delta_tpw >= 0 else ""
        print(
            f"    Fleet tok/W    : {sign}{delta_tpw:+.1f}%  "
            f"({'better' if delta_tpw > 0 else 'worse'} energy efficiency)"
        )
        sign = "+" if delta_cost >= 0 else ""
        print(f"    Fleet $/1M tok : {sign}{delta_cost:+.1f}%")
        print(
            f"    Total GPUs     : {delta_gpus:+d}  "
            f"({homo_result.total_gpus} → {routed_result.total_gpus})"
        )
        print()

        if args.out:
            import json

            out_data = {
                "homo": {
                    "fleet_tpw": homo_result.fleet_tpw,
                    "fleet_cost_per_mil": homo_result.fleet_cost_per_mil,
                    "fleet_power_kw": homo_result.fleet_power_kw,
                    "total_gpus": homo_result.total_gpus,
                },
                "routed": {
                    "fleet_tpw": routed_result.fleet_tpw,
                    "fleet_cost_per_mil": routed_result.fleet_cost_per_mil,
                    "fleet_power_kw": routed_result.fleet_power_kw,
                    "total_gpus": routed_result.total_gpus,
                },
            }
            json.dump(out_data, open(args.out, "w"), indent=2)
            print(f"Results saved to {args.out}")

    else:
        # ── Single-pool per-GPU comparison (original behaviour) ────────────
        gpus = [GPU_REGISTRY.get(g, A100_80GB) for g in args.gpus]
        rho_sweep = [0.20, 0.40, 0.60, 0.80] if args.rho_sweep else None

        points = tpw_analysis(
            cdf=cdf,
            lam=args.lam,
            gpus=gpus,
            t_slo_ms=args.slo,
            max_ctx=args.max_ctx,
            rho_sweep=rho_sweep,
        )

        print(
            f"\nTokens-per-Watt comparison  "
            f"(CDF: {args.cdf},  λ={args.lam} req/s,  SLO={args.slo}ms)"
        )
        print("  NOTE: all GPUs compared with the same CDF and arrival rate.")
        print("  Pre-built profiles mix models (H100/A100=70B, A10G=7B).")
        print("  Use --b-short/--gpu-short/--gpu-long for routing comparison.")
        print_tpw_table(points)

        if args.rho_sweep:
            print("Tokens/Watt vs Utilisation sweep (SLO-optimal row marked ★):")
            hdr = (
                f"  {'Pool / GPU':22s}  {'ρ':>5}  {'n_act':>6}  "
                f"{'Tok/W':>6}  {'P/GPU':>6}  {'$/1M':>7}  {'P99ms':>7}  {'PwrQ':>4}"
            )
            print("  " + "-" * (len(hdr) - 2))
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for pt in points:
                flag = " ★" if pt.slo_optimal else "  "
                label = pt.pool_label or pt.gpu_name
                print(
                    f"  {label:22s}  {pt.rho:>5.2f}  {pt.n_active:>6.1f}  "
                    f"{pt.tokens_per_watt:>6.2f}  {pt.power_per_gpu_w:>6.0f}W  "
                    f"${pt.cost_per_mil:>6.2f}  {pt.p99_ttft_ms:>7.1f}  "
                    f"{pt.power_model_qual:>4}{flag}"
                )
            print()

        if args.out:
            import json

            data = [
                {
                    k: getattr(pt, k)
                    for k in (
                        "gpu_name",
                        "pool_label",
                        "n_gpus",
                        "rho",
                        "n_active",
                        "power_per_gpu_w",
                        "tokens_per_watt",
                        "cost_per_mil",
                        "p99_ttft_ms",
                        "power_model_qual",
                        "slo_optimal",
                    )
                }
                for pt in points
            ]
            json.dump(data, open(args.out, "w"), indent=2)
            print(f"Results saved to {args.out}")


def cmd_grid_flex(args):
    """Compute the power–latency trade-off curve for a fleet under demand response.

    Models the GPU-to-Grid (G2G) batch-size control mechanism: when the grid
    signals high load, the LLM serving engine caps the maximum in-flight batch
    size (vLLM max_num_seqs), reducing GPU power at the cost of higher
    queuing latency.

    The command sweeps flex commitment percentages (0% → 50% power reduction)
    and reports, for each level:
      - the resulting per-GPU batch-size cap (n_max)
      - estimated GPU power draw (W/GPU) and total fleet power (kW)
      - analytical P99 TTFT estimate
      - whether the SLO is still met

    The maximum safe flex depth — the largest power reduction that still
    meets the SLO — is highlighted at the bottom of the table.

    Sources
    -------
    Hassan et al. (2025) "GPU-to-Grid: Coupling LLM Inference with Power
    System Control", arXiv:2602.05116v1.
    ML.ENERGY Benchmark v3.0 (Chung et al., NeurIPS 2025 D&B track).
    """
    from fleet_sim.optimizer import grid_flex_analysis, print_grid_flex_table

    cdf = load_cdf(args.cdf)
    gpu = GPU_REGISTRY.get(args.gpu, A100_80GB)

    if gpu.power_idle_w <= 0 or gpu.power_nominal_w <= 0:
        print(f"[error] GPU profile '{gpu.name}' has no power model configured.")
        sys.exit(1)

    flex_pcts = [float(x) for x in args.flex_pcts] if args.flex_pcts else None

    if args.verify_des and args.verify_des > 0:
        print(
            f"[DES verification enabled: {args.verify_des:,} requests per flex level]"
        )

    results = grid_flex_analysis(
        cdf=cdf,
        lam=args.lam,
        n_gpus=args.n_gpus,
        gpu=gpu,
        t_slo_ms=args.slo,
        max_ctx=args.max_ctx,
        flex_pcts=flex_pcts,
        n_sim_requests=args.verify_des or 0,
        verbose=True,
    )

    print_grid_flex_table(results, t_slo_ms=args.slo, n_gpus=args.n_gpus, lam=args.lam)

    if args.out:
        data = [
            {
                "flex_pct": pt.flex_pct,
                "n_max_cap": pt.n_max_cap,
                "power_per_gpu_w": pt.power_per_gpu_w,
                "power_fleet_kw": pt.power_fleet_kw,
                "p99_ttft_ms": pt.p99_ttft_ms,
                "slo_met": pt.slo_met,
            }
            for pt in results
        ]
        json.dump(data, open(args.out, "w"), indent=2)
        print(f"\nResults saved to {args.out}")


# ── CLI setup ─────────────────────────────────────────────────────────────────


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="vllm-sr-sim",
        description="vllm-sr-sim: fleet-level LLM inference simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {FLEET_SIM_VERSION}",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── common args ───────────────────────────────────────────────────────────
    def add_common(sp):
        sp.add_argument("--cdf", required=True, help="Path to CDF JSON file")
        sp.add_argument("--lam", type=float, default=200, help="Arrival rate (req/s)")
        sp.add_argument("--slo", type=float, default=500, help="P99 TTFT SLO (ms)")
        sp.add_argument(
            "--b-short",
            type=int,
            default=4096,
            help="Short-pool context threshold (tokens)",
        )
        sp.add_argument(
            "--long-max-ctx",
            type=int,
            default=65536,
            help="Long-pool max context (tokens)",
        )
        sp.add_argument(
            "--gpu-short", default="a100", choices=list(GPU_REGISTRY.keys())
        )
        sp.add_argument("--gpu-long", default="a100", choices=list(GPU_REGISTRY.keys()))
        sp.add_argument("--out", default=None, help="Save results to JSON file")

    # ── optimize ──────────────────────────────────────────────────────────────
    sp_opt = sub.add_parser("optimize", help="Find min-cost fleet meeting SLO")
    add_common(sp_opt)
    sp_opt.add_argument("--gamma-max", type=float, default=2.0)
    sp_opt.add_argument("--n-sim-req", type=int, default=40000)
    sp_opt.add_argument("--verify-top", type=int, default=3)
    sp_opt.set_defaults(func=cmd_optimize)

    # ── simulate ──────────────────────────────────────────────────────────────
    sp_sim = sub.add_parser("simulate", help="Simulate a fixed fleet configuration")
    add_common(sp_sim)
    sp_sim.add_argument("--n-s", type=int, required=True, help="Short pool GPUs")
    sp_sim.add_argument("--n-l", type=int, required=True, help="Long pool GPUs")
    sp_sim.add_argument(
        "--gamma", type=float, default=1.0, help="C&R gamma (1.0 = pool routing only)"
    )
    sp_sim.add_argument("--n-req", type=int, default=50000)
    sp_sim.add_argument("--seed", type=int, default=42)
    sp_sim.set_defaults(func=cmd_simulate)

    # ── whatif ────────────────────────────────────────────────────────────────
    sp_wi = sub.add_parser(
        "whatif", help="Sweep arrival rate and/or GPU types (what-if analysis)"
    )
    add_common(sp_wi)
    sp_wi.add_argument(
        "--lam-range",
        type=float,
        nargs="+",
        default=[50, 100, 200, 500, 1000],
        help="List of arrival rates to sweep",
    )
    sp_wi.add_argument(
        "--gpu-compare",
        type=str,
        nargs="+",
        metavar="GPU",
        help="Compare multiple GPU types side-by-side "
        "(e.g. --gpu-compare a100 h100 a10g). "
        "When set, --gpu-short/--gpu-long are ignored.",
    )
    sp_wi.set_defaults(func=cmd_whatif)

    # ── pareto ────────────────────────────────────────────────────────────────
    sp_pa = sub.add_parser(
        "pareto",
        help="Sweep all CDF breakpoints as B_short candidates and show Pareto frontier",
    )
    add_common(sp_pa)
    sp_pa.set_defaults(func=cmd_pareto)

    # ── compare-routers ───────────────────────────────────────────────────────
    sp_cr = sub.add_parser(
        "compare-routers", help="Compare routing algorithms on same fleet"
    )
    add_common(sp_cr)
    sp_cr.add_argument("--n-s", type=int, required=True)
    sp_cr.add_argument("--n-l", type=int, required=True)
    sp_cr.add_argument("--n-req", type=int, default=30000)
    sp_cr.add_argument("--seed", type=int, default=42)
    sp_cr.set_defaults(func=cmd_compare_routers)

    # ── disagg ────────────────────────────────────────────────────────────────
    sp_dis = sub.add_parser(
        "disagg",
        help="Disaggregated prefill/decode fleet optimizer (separate P and D pools)",
    )
    sp_dis.add_argument("--cdf", required=True, help="Workload CDF JSON")
    sp_dis.add_argument(
        "--lam", type=float, default=200, help="Target arrival rate (req/s)"
    )
    sp_dis.add_argument(
        "--slo-ttft", type=float, default=500, help="TTFT SLO (ms, default 500)"
    )
    sp_dis.add_argument(
        "--slo-tpot", type=float, default=100, help="TPOT SLO (ms, default 100)"
    )
    sp_dis.add_argument(
        "--gpu-prefill",
        default="h100",
        choices=list(GPU_REGISTRY.keys()),
        help="GPU for prefill (context-ingestion) workers",
    )
    sp_dis.add_argument(
        "--gpu-decode",
        default="a100",
        choices=list(GPU_REGISTRY.keys()),
        help="GPU for decode (token-generation) workers",
    )
    sp_dis.add_argument(
        "--max-ctx",
        type=int,
        default=8192,
        help="Max context length (tokens, default 8192)",
    )
    sp_dis.add_argument(
        "--mean-isl",
        type=float,
        default=0,
        help="Mean input sequence length (0 = derive from CDF)",
    )
    sp_dis.add_argument(
        "--mean-osl",
        type=float,
        default=0,
        help="Mean output sequence length (0 = derive from CDF)",
    )
    sp_dis.add_argument(
        "--max-prefill",
        type=int,
        default=32,
        help="Max prefill workers to sweep (default 32)",
    )
    sp_dis.add_argument(
        "--max-decode",
        type=int,
        default=64,
        help="Max decode workers to sweep (default 64)",
    )
    sp_dis.add_argument("--out", default=None, help="Save results to JSON")
    sp_dis.set_defaults(func=cmd_disagg)

    # ── grid-flex ─────────────────────────────────────────────────────────────
    sp_gf = sub.add_parser(
        "grid-flex",
        help="Power–latency trade-off curve for demand-response (GPU-to-Grid)",
        description=(
            "Sweep batch-size curtailment levels and report P99 TTFT vs. power "
            "reduction.  Models the GPU-to-Grid (G2G) demand-response mechanism "
            "(Hassan et al. arXiv:2602.05116v1): vLLM caps max_num_seqs to reduce "
            "GPU power when the grid signals high load."
        ),
    )
    sp_gf.add_argument("--cdf", required=True, help="Workload CDF JSON")
    sp_gf.add_argument("--lam", type=float, default=200, help="Arrival rate (req/s)")
    sp_gf.add_argument(
        "--n-gpus", type=int, required=True, help="Fixed fleet size (GPUs) to evaluate"
    )
    sp_gf.add_argument(
        "--gpu",
        default="h100",
        choices=list(GPU_REGISTRY.keys()),
        help="GPU type (default: h100)",
    )
    sp_gf.add_argument("--slo", type=float, default=500, help="P99 TTFT SLO (ms)")
    sp_gf.add_argument(
        "--max-ctx",
        type=int,
        default=8192,
        help="Max context window (tokens, default 8192)",
    )
    sp_gf.add_argument(
        "--flex-pcts",
        type=float,
        nargs="+",
        default=None,
        metavar="PCT",
        help="Power-reduction percentages to sweep "
        "(default: 0 5 10 15 20 25 30 40 50)",
    )
    sp_gf.add_argument(
        "--verify-des",
        type=int,
        default=0,
        metavar="N",
        help="DES-verify each flex level with N simulated requests "
        "(0 = analytical only; 10000–20000 recommended)",
    )
    sp_gf.add_argument("--out", default=None, help="Save results to JSON")
    sp_gf.set_defaults(func=cmd_grid_flex)

    # ── tok-per-watt ──────────────────────────────────────────────────────────
    sp_tpw = sub.add_parser(
        "tok-per-watt",
        help="Compare GPU energy efficiency (tokens/watt) at the SLO-optimal fleet",
        description=(
            "Two modes:\n\n"
            "1. Single-pool comparison (--gpus h100 a100 a10g):\n"
            "   Each GPU receives the full CDF.  Correct only when comparing\n"
            "   the same model on different hardware.\n\n"
            "2. Two-pool routing comparison (--b-short B --gpu-short G --gpu-long G):\n"
            "   Splits traffic at B tokens. Computes the fleet-level tok/W using\n"
            "   the correct aggregate formula (N does NOT cancel across pools).\n"
            "   Models hetero-pool and semantic-router short/long routing.\n\n"
            "Power model confidence: H100=HIGH (ML.ENERGY measured), "
            "A100=FAIR (one anchor+scaling), A10G=LOW (projection only).\n\n"
            "Model calibration: H100/A100 profiles = 70B on 8-GPU TP; "
            "A10G = 7B-class on single GPU."
        ),
    )
    sp_tpw.add_argument("--cdf", required=True, help="Workload CDF JSON")
    sp_tpw.add_argument(
        "--lam", type=float, default=100, help="Arrival rate (req/s, default: 100)"
    )
    sp_tpw.add_argument(
        "--slo", type=float, default=500, help="P99 TTFT SLO (ms, default: 500)"
    )
    sp_tpw.add_argument(
        "--gpus",
        nargs="+",
        default=["h100", "a100", "a10g"],
        choices=list(GPU_REGISTRY.keys()),
        help="GPU types to compare in single-pool mode (default: h100 a100 a10g)",
    )
    sp_tpw.add_argument(
        "--max-ctx",
        type=int,
        default=8192,
        help="Max context window (tokens, default: 8192)",
    )
    sp_tpw.add_argument(
        "--rho-sweep",
        action="store_true",
        help="Also show tok/W at ρ=0.2,0.4,0.6,0.8 (single-pool mode only)",
    )
    # Two-pool routing flags
    sp_tpw.add_argument(
        "--b-short",
        type=int,
        default=None,
        metavar="TOKENS",
        help="[two-pool mode] Token-count threshold: requests ≤ B_short go to "
        "the short pool, rest go to the long pool",
    )
    sp_tpw.add_argument(
        "--gpu-short",
        default=None,
        choices=list(GPU_REGISTRY.keys()),
        help="[two-pool mode] GPU type for the short pool",
    )
    sp_tpw.add_argument(
        "--gpu-long",
        default=None,
        choices=list(GPU_REGISTRY.keys()),
        help="[two-pool mode] GPU type for the long pool",
    )
    sp_tpw.add_argument("--out", default=None, help="Save results to JSON")
    sp_tpw.set_defaults(func=cmd_tok_per_watt)

    # ── simulate-fleet ────────────────────────────────────────────────────────
    sp_sf = sub.add_parser(
        "simulate-fleet",
        help="Simulate an arbitrary N-pool fleet from a JSON config",
        description=(
            "Simulate any fleet topology: multiple models, heterogeneous GPU "
            "types, or pools not distinguished by prompt length. "
            "Pass a JSON config defining pools, router, and optionally per-pool "
            "workload CDFs. Use --cdf / --lam to override with a single stream."
        ),
    )
    sp_sf.add_argument("fleet_config", help="Path to fleet JSON config file")
    sp_sf.add_argument(
        "--cdf", default=None, help="Workload CDF JSON (overrides config workloads)"
    )
    sp_sf.add_argument(
        "--lam", type=float, default=200, help="Total arrival rate req/s (default: 200)"
    )
    sp_sf.add_argument(
        "--slo",
        type=float,
        default=500,
        help="P99 TTFT SLO ms for reporting (default: 500)",
    )
    sp_sf.add_argument(
        "--n-req",
        type=int,
        default=30000,
        help="Total requests to simulate (default: 30000)",
    )
    sp_sf.add_argument("--seed", type=int, default=42)
    sp_sf.add_argument("--out", default=None, help="Save results to JSON file")
    sp_sf.set_defaults(func=cmd_simulate_fleet)

    # ── serve ─────────────────────────────────────────────────────────────────
    sp_serve = sub.add_parser(
        "serve",
        help="Start the vllm-sr-sim HTTP service",
    )
    sp_serve.add_argument("--host", default="127.0.0.1")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.add_argument("--reload", action="store_true")
    sp_serve.add_argument("--workers", type=int, default=1)
    sp_serve.add_argument("--log-level", default="info")
    sp_serve.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
