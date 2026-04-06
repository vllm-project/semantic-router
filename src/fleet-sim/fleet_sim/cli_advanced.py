"""Advanced fleet-sim CLI commands."""

from __future__ import annotations

import dataclasses
import json
import sys

from .cli_common import GPU_REGISTRY, load_cdf, resolve_gpu, write_json_output
from .core.fleet import Fleet, FleetConfig, PoolConfig
from .gpu_profiles.profiles import A100_80GB, H100_80GB
from .optimizer import FleetOptimizer
from .workload.synthetic import CdfWorkload, PoissonWorkload


def cmd_disagg(args) -> None:
    from fleet_sim.optimizer.disagg import DisaggFleetOptimizer

    cdf = load_cdf(args.cdf)
    gpu_pre = resolve_gpu(args.gpu_prefill, H100_80GB)
    gpu_dec = resolve_gpu(args.gpu_decode, A100_80GB)
    mean_isl, mean_osl = _derive_disagg_lengths(cdf, args)
    _print_disagg_banner(args, gpu_pre, gpu_dec, mean_isl, mean_osl)
    optimizer = DisaggFleetOptimizer(
        prefill_profile=gpu_pre,
        decode_profile=gpu_dec,
        mean_isl=mean_isl,
        mean_osl=mean_osl,
        slo_ttft_ms=args.slo_ttft,
        slo_tpot_ms=args.slo_tpot,
        max_ctx=args.max_ctx,
    )
    points = optimizer.sweep(max_prefill=args.max_prefill, max_decode=args.max_decode)
    feasible = [
        point for point in points if point.slo_met and point.system_rate >= args.lam
    ]
    if not feasible:
        _print_disagg_no_solution(args, optimizer, points)
        return
    best, pareto = _select_disagg_points(feasible)
    _print_disagg_pareto_table(args, pareto, best)
    _print_disagg_aggregated_comparison(args, cdf, gpu_pre, gpu_dec, best)
    _write_disagg_output(args.out, best, pareto)


def cmd_simulate_fleet(args) -> None:
    cfg = json.loads(open(args.fleet_config).read())
    pool_configs = _build_pool_configs(cfg)
    router_type = _resolve_router_type(cfg.get("router", "length"))
    arrivals = _build_arrivals(args, cfg, pool_configs)
    print(f"Simulating {len(arrivals):,} requests across {len(pool_configs)} pools ...")
    print(
        "  Pools: "
        + ", ".join(
            f"{pool.pool_id}({pool.n_gpus}×{pool.gpu.name},ctx={pool.max_ctx})"
            for pool in pool_configs
        )
    )
    print(f"  Router: {router_type}")
    result = Fleet(FleetConfig(pools=pool_configs, router_type=router_type)).run(
        arrivals, verbose=True
    )
    result.print_summary(t_slo_ms=args.slo)
    write_json_output(args.out, result.summary(args.slo))


def cmd_tok_per_watt(args) -> None:
    cdf = load_cdf(args.cdf)
    if args.b_short is not None and args.gpu_short and args.gpu_long:
        _run_two_pool_tpw(args, cdf)
        return
    _run_single_pool_tpw(args, cdf)


def cmd_grid_flex(args) -> None:
    from fleet_sim.optimizer import grid_flex_analysis, print_grid_flex_table

    cdf = load_cdf(args.cdf)
    gpu = resolve_gpu(args.gpu, A100_80GB)
    if gpu.power_idle_w <= 0 or gpu.power_nominal_w <= 0:
        print(f"[error] GPU profile '{gpu.name}' has no power model configured.")
        sys.exit(1)
    flex_pcts = [float(value) for value in args.flex_pcts] if args.flex_pcts else None
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
    write_json_output(
        args.out,
        [
            {
                "flex_pct": point.flex_pct,
                "n_max_cap": point.n_max_cap,
                "power_per_gpu_w": point.power_per_gpu_w,
                "power_fleet_kw": point.power_fleet_kw,
                "p99_ttft_ms": point.p99_ttft_ms,
                "slo_met": point.slo_met,
            }
            for point in results
        ],
    )


def _derive_disagg_lengths(cdf, args) -> tuple[int, int]:
    mean_total = sum(
        threshold * (cdf[index][1] - (cdf[index - 1][1] if index > 0 else 0.0))
        for index, (threshold, _frac) in enumerate(cdf)
    )
    mean_isl = int(args.mean_isl) if args.mean_isl else int(mean_total * 0.75)
    mean_osl = int(args.mean_osl) if args.mean_osl else int(mean_total * 0.25)
    return mean_isl, mean_osl


def _print_disagg_banner(args, gpu_pre, gpu_dec, mean_isl: int, mean_osl: int) -> None:
    print("\nDisaggregated prefill/decode fleet optimizer")
    print(f"  CDF: {args.cdf}  λ={args.lam:.0f} req/s")
    print(f"  Prefill GPU: {gpu_pre.name}  Decode GPU: {gpu_dec.name}")
    print(f"  mean ISL={mean_isl} tok  mean OSL={mean_osl} tok  max_ctx={args.max_ctx}")
    print(f"  SLO: TTFT={args.slo_ttft}ms  TPOT={args.slo_tpot}ms")


def _print_disagg_no_solution(args, optimizer, points) -> None:
    max_rate = max((point.system_rate for point in points), default=0.0)
    print(f"\n  No valid configuration found for λ={args.lam:.0f} req/s.")
    print(f"  Max achievable throughput: {max_rate:.1f} req/s")
    if any(point.slo_met for point in points):
        return
    eff = optimizer._prefill_ttft_ms() * optimizer.beta_ttft
    tpot = optimizer._decode_tpot_ms()
    print(
        f"  Single-instance TTFT={eff:.0f}ms  TPOT={tpot:.0f}ms  "
        f"(SLOs: {args.slo_ttft}ms / {args.slo_tpot}ms)"
    )


def _select_disagg_points(feasible):
    best = min(feasible, key=lambda point: point.cost_per_hr)
    pareto = [
        point
        for point in feasible
        if not any(
            other.cost_per_hr <= point.cost_per_hr
            and other.total_gpus <= point.total_gpus
            and (
                other.cost_per_hr < point.cost_per_hr
                or other.total_gpus < point.total_gpus
            )
            for other in feasible
        )
    ]
    pareto.sort(key=lambda point: point.total_gpus)
    return best, pareto


def _print_disagg_pareto_table(args, pareto, best) -> None:
    print(f"\n  Pareto-efficient configs (SLO ✓, λ ≥ {args.lam:.0f} req/s):")
    print(
        f"  {'nP':>4} {'nD':>4} {'GPUs':>6} {'rate':>8} {'TTFT':>8} {'TPOT':>8} "
        f"{'$/hr':>7} {'$K/yr':>7}"
    )
    print(f"  {'-' * 60}")
    for point in pareto:
        marker = " ★" if point is best else ""
        print(
            f"  {point.n_prefill:>4} {point.n_decode:>4} {point.total_gpus:>6}"
            f" {point.system_rate:>7.1f}/s {point.ttft_ms:>7.0f}ms {point.tpot_ms:>7.0f}ms"
            f" ${point.cost_per_hr:>5.2f}/hr ${point.cost_per_hr * 8760 / 1000:>5.0f}K{marker}"
        )


def _print_disagg_aggregated_comparison(args, cdf, gpu_pre, gpu_dec, best) -> None:
    print(
        f"\n  Comparison vs aggregated (homogeneous) fleets at λ={args.lam:.0f} req/s:"
    )
    seen = set()
    for gpu_name in [args.gpu_prefill, args.gpu_decode]:
        if gpu_name in seen:
            continue
        seen.add(gpu_name)
        gpu = GPU_REGISTRY[gpu_name]
        optimizer = FleetOptimizer(
            gpu_short=gpu,
            gpu_long=gpu,
            B_short=args.max_ctx,
            t_slo_ms=args.slo_ttft,
            long_max_ctx=args.max_ctx,
        )
        agg_ok = [
            row
            for row in optimizer.sweep_analytical(
                cdf, args.lam, gammas=[1.0], verbose=False
            )
            if row.slo_met
        ]
        if agg_ok:
            agg = min(agg_ok, key=lambda row: row.annualised_cost_kusd)
            print(
                f"  All-{gpu_name.upper():6s}  {agg.total_gpus:>4} GPUs  "
                f"${agg.annualised_cost_kusd:>5.0f}K/yr  "
                f"P99 TTFT={agg.p99_ttft_short_ms:.0f}ms"
            )
            continue
        print(f"  All-{gpu_name.upper():6s}  No SLO-feasible config found")
    disagg_yr = best.cost_per_hr * 8760 / 1000
    print(
        f"  Disagg {gpu_pre.name}P+{gpu_dec.name}D  "
        f"{best.total_gpus:>4} GPUs ({best.n_prefill}P+{best.n_decode}D)  "
        f"${disagg_yr:>5.0f}K/yr  TTFT={best.ttft_ms:.0f}ms  TPOT={best.tpot_ms:.0f}ms"
    )


def _write_disagg_output(path: str | None, best, pareto) -> None:
    if not path:
        return
    write_json_output(
        path,
        {
            "best": dataclasses.asdict(best),
            "pareto": [dataclasses.asdict(point) for point in pareto],
        },
    )


def _build_pool_configs(cfg: dict) -> list[PoolConfig]:
    pool_configs = []
    for pool in cfg["pools"]:
        pool_configs.append(
            PoolConfig(
                pool["id"],
                resolve_gpu(pool["gpu"], A100_80GB),
                pool["n_gpus"],
                pool["max_ctx"],
            )
        )
    return pool_configs


def _resolve_router_type(router_name: str) -> str:
    return {
        "length": "LengthRouter",
        "model": "ModelRouter",
        "semantic": "SemanticRouter",
        "random": "RandomRouter",
        "least_loaded": "LeastLoadedRouter",
    }.get(router_name, "LengthRouter")


def _build_arrivals(args, cfg: dict, pool_configs: list[PoolConfig]):
    if args.cdf:
        return _build_single_cdf_arrivals(args)
    if "workloads" in cfg and cfg.get("router", "length") == "model":
        return _build_model_workload_arrivals(args, cfg)
    return _build_fallback_arrivals(args, pool_configs[0])


def _build_single_cdf_arrivals(args):
    workload = PoissonWorkload(
        args.lam,
        CdfWorkload(load_cdf(args.cdf), seed=args.seed),
        n_requests=args.n_req,
        seed=args.seed,
    )
    return workload.generate()


def _build_model_workload_arrivals(args, cfg: dict):
    all_pairs: list = []
    for pool_id, workload_spec in cfg["workloads"].items():
        lam_frac = workload_spec.get("lam_frac", 1.0 / len(cfg["workloads"]))
        workload = PoissonWorkload(
            args.lam * lam_frac,
            CdfWorkload(load_cdf(workload_spec["cdf"]), seed=args.seed),
            n_requests=max(1, int(args.n_req * workload_spec.get("lam_frac", 1.0))),
            seed=args.seed,
        )
        for time_value, request in workload.generate():
            request.model_id = pool_id
            all_pairs.append((time_value, request))
    all_pairs.sort(key=lambda pair: pair[0])
    return all_pairs


def _build_fallback_arrivals(args, pool: PoolConfig):
    print(
        f"[warn] No --cdf provided and no workloads in config; "
        f"using Poisson arrivals at λ={args.lam} req/s"
    )
    workload = PoissonWorkload(
        args.lam,
        CdfWorkload([(pool.max_ctx, 1.0)], seed=args.seed),
        n_requests=args.n_req,
        seed=args.seed,
    )
    return workload.generate()


def _run_two_pool_tpw(args, cdf) -> None:
    from fleet_sim.optimizer import _split_cdf, fleet_tpw_analysis, print_fleet_tpw

    b_short = args.b_short
    gpu_short = resolve_gpu(args.gpu_short, A100_80GB)
    gpu_long = resolve_gpu(args.gpu_long, A100_80GB)
    short_cdf, long_cdf, alpha = _split_cdf(cdf, b_short)
    lam_short = alpha * args.lam
    lam_long = (1.0 - alpha) * args.lam
    print(f"\nTwo-pool routing split at B_short={b_short} tokens:")
    print(f"  Short pool ({gpu_short.name}): α={alpha:.3f}, λ_s={lam_short:.1f} req/s")
    print(
        f"  Long  pool ({gpu_long.name}): 1-α={1-alpha:.3f}, λ_l={lam_long:.1f} req/s"
    )
    if gpu_short.name != gpu_long.name:
        print("\n  NOTE: different GPU types likely represent different model sizes.")
        print("  A10G is calibrated for 7B-class models; H100/A100 for 70B.")
        print("  Fleet tok/W reflects combined model+hardware efficiency.")
    homo_result = fleet_tpw_analysis(
        pools=[
            {
                "gpu": gpu_long,
                "cdf": cdf,
                "lam": args.lam,
                "max_ctx": args.max_ctx,
                "label": f"homo-{gpu_long.name}",
            }
        ],
        lam_total=args.lam,
        t_slo_ms=args.slo,
        topology=f"homo {gpu_long.name} (baseline)",
    )
    routed_result = fleet_tpw_analysis(
        pools=[
            {
                "gpu": gpu_short,
                "cdf": short_cdf,
                "lam": lam_short,
                "max_ctx": args.max_ctx,
                "label": f"short({b_short}T)-{gpu_short.name}",
            },
            {
                "gpu": gpu_long,
                "cdf": long_cdf,
                "lam": lam_long,
                "max_ctx": args.max_ctx,
                "label": f"long-{gpu_long.name}",
            },
        ],
        lam_total=args.lam,
        t_slo_ms=args.slo,
        topology=f"two-pool {gpu_short.name}(short) + {gpu_long.name}(long)",
    )
    print(
        f"\nTokens-per-Watt fleet comparison  "
        f"(CDF: {args.cdf},  λ={args.lam} req/s,  SLO={args.slo}ms)"
    )
    print_fleet_tpw(homo_result, title="── Baseline: homo pool ──")
    print_fleet_tpw(routed_result, title="── Routed:  two-pool ──")
    _print_tpw_delta(homo_result, routed_result)
    write_json_output(
        args.out,
        {
            "homo": _fleet_tpw_summary(homo_result),
            "routed": _fleet_tpw_summary(routed_result),
        },
    )


def _run_single_pool_tpw(args, cdf) -> None:
    from fleet_sim.optimizer import print_tpw_table, tpw_analysis

    points = tpw_analysis(
        cdf=cdf,
        lam=args.lam,
        gpus=[resolve_gpu(name, A100_80GB) for name in args.gpus],
        t_slo_ms=args.slo,
        max_ctx=args.max_ctx,
        rho_sweep=[0.20, 0.40, 0.60, 0.80] if args.rho_sweep else None,
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
        _print_rho_sweep(points)
    write_json_output(
        args.out,
        [
            {
                key: getattr(point, key)
                for key in (
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
            for point in points
        ],
    )


def _print_tpw_delta(homo_result, routed_result) -> None:
    delta_tpw = (routed_result.fleet_tpw / homo_result.fleet_tpw - 1) * 100
    delta_cost = (
        routed_result.fleet_cost_per_mil / homo_result.fleet_cost_per_mil - 1
    ) * 100
    delta_gpus = routed_result.total_gpus - homo_result.total_gpus
    print("\n  Routing benefit vs homo baseline:")
    print(
        f"    Fleet tok/W    : {delta_tpw:+.1f}%  "
        f"({'better' if delta_tpw > 0 else 'worse'} energy efficiency)"
    )
    print(f"    Fleet $/1M tok : {delta_cost:+.1f}%")
    print(
        f"    Total GPUs     : {delta_gpus:+d}  ({homo_result.total_gpus} → {routed_result.total_gpus})"
    )
    print()


def _print_rho_sweep(points) -> None:
    print("Tokens/Watt vs Utilisation sweep (SLO-optimal row marked ★):")
    header = (
        f"  {'Pool / GPU':22s}  {'ρ':>5}  {'n_act':>6}  "
        f"{'Tok/W':>6}  {'P/GPU':>6}  {'$/1M':>7}  {'P99ms':>7}  {'PwrQ':>4}"
    )
    print("  " + "-" * (len(header) - 2))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for point in points:
        label = point.pool_label or point.gpu_name
        flag = " ★" if point.slo_optimal else "  "
        print(
            f"  {label:22s}  {point.rho:>5.2f}  {point.n_active:>6.1f}  "
            f"{point.tokens_per_watt:>6.2f}  {point.power_per_gpu_w:>6.0f}W  "
            f"${point.cost_per_mil:>6.2f}  {point.p99_ttft_ms:>7.1f}  "
            f"{point.power_model_qual:>4}{flag}"
        )
    print()


def _fleet_tpw_summary(result) -> dict[str, float | int]:
    return {
        "fleet_tpw": result.fleet_tpw,
        "fleet_cost_per_mil": result.fleet_cost_per_mil,
        "fleet_power_kw": result.fleet_power_kw,
        "total_gpus": result.total_gpus,
    }
