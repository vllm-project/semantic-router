"""Core fleet-sim CLI commands."""

from __future__ import annotations

import dataclasses
import os

from .cli_common import GPU_REGISTRY, load_cdf, resolve_gpu, write_json_output
from .core.fleet import Fleet, FleetConfig, PoolConfig
from .gpu_profiles.profiles import A100_80GB
from .optimizer import FleetOptimizer, print_threshold_pareto, threshold_pareto
from .workload.synthetic import CdfWorkload, PoissonWorkload


def cmd_optimize(args) -> None:
    cdf = load_cdf(args.cdf)
    optimizer = FleetOptimizer(
        gpu_short=resolve_gpu(args.gpu_short, A100_80GB),
        gpu_long=resolve_gpu(args.gpu_long, A100_80GB),
        B_short=args.b_short,
        t_slo_ms=args.slo,
        long_max_ctx=args.long_max_ctx,
    )
    gammas = [
        round(1.0 + 0.1 * step, 1)
        for step in range(int((args.gamma_max - 1.0) / 0.1) + 1)
    ]
    report = optimizer.optimize(
        cdf=cdf,
        lam=args.lam,
        gammas=gammas,
        n_sim_requests=args.n_sim_req,
        verify_top_n=args.verify_top,
        verbose=True,
    )
    report.print_report()
    write_json_output(
        args.out,
        [_serialize_sweep_result(row) for row in report.analytical + report.simulated],
    )


def cmd_simulate(args) -> None:
    cdf = load_cdf(args.cdf)
    fleet_config = FleetConfig(
        pools=[
            PoolConfig(
                "short", resolve_gpu(args.gpu_short, A100_80GB), args.n_s, args.b_short
            ),
            PoolConfig(
                "long",
                resolve_gpu(args.gpu_long, A100_80GB),
                args.n_l,
                args.long_max_ctx,
            ),
        ],
        router_type="CompressAndRouteRouter" if args.gamma > 1.0 else "LengthRouter",
        router_kwargs=(
            {"B_short": args.b_short, "gamma": args.gamma}
            if args.gamma > 1.0
            else {"threshold": args.b_short}
        ),
    )
    workload = PoissonWorkload(
        args.lam,
        CdfWorkload(cdf, seed=args.seed),
        n_requests=args.n_req,
        seed=args.seed,
    )
    arrivals = workload.generate()
    print(f"Simulating {args.n_req:,} requests at λ={args.lam:.0f} req/s ...")
    result = Fleet(fleet_config).run(arrivals, verbose=True)
    result.print_summary(t_slo_ms=args.slo)
    write_json_output(args.out, result.summary(args.slo))


def _whatif_single(cdf, lam_range, gpu_s, gpu_l, b_short, slo, long_max_ctx):
    optimizer = FleetOptimizer(
        gpu_short=gpu_s,
        gpu_long=gpu_l,
        B_short=b_short,
        t_slo_ms=slo,
        long_max_ctx=long_max_ctx,
    )
    rows = []
    for lam in lam_range:
        results = optimizer.sweep_analytical(cdf, lam, verbose=False)
        rows.append((lam, _best_sweep_result(results)))
    return rows


def cmd_whatif(args) -> None:
    cdf = load_cdf(args.cdf)
    gpu_names = getattr(args, "gpu_compare", None) or []
    if gpu_names:
        _run_gpu_comparison_mode(args, cdf, gpu_names)
        return
    _run_lambda_sweep_mode(args, cdf)


def cmd_pareto(args) -> None:
    cdf = load_cdf(args.cdf)
    gpu_short = resolve_gpu(args.gpu_short, A100_80GB)
    gpu_long = resolve_gpu(args.gpu_long, A100_80GB)
    homo_opt = FleetOptimizer(
        gpu_short=gpu_short,
        gpu_long=gpu_long,
        B_short=args.long_max_ctx,
        t_slo_ms=args.slo,
        long_max_ctx=args.long_max_ctx,
    )
    homo_sweep = homo_opt.sweep_analytical(cdf, args.lam, gammas=[1.0], verbose=False)
    homo_cost = homo_sweep[0].annualised_cost_kusd if homo_sweep else None
    print("\nPareto frontier: threshold sweep")
    print(
        f"  CDF: {args.cdf}  λ={args.lam:.0f} req/s  SLO={args.slo}ms"
        f"  GPU: {gpu_short.name}/{gpu_long.name}"
    )
    if homo_cost:
        print(
            f"  Homo baseline (B_short={args.long_max_ctx:,}): "
            f"${homo_cost:.0f}K/yr  ({homo_sweep[0].total_gpus} GPUs)"
        )
    results = threshold_pareto(
        cdf=cdf,
        lam=args.lam,
        gpu_short=gpu_short,
        gpu_long=gpu_long,
        t_slo_ms=args.slo,
        long_max_ctx=args.long_max_ctx,
    )
    if not results:
        print("  No valid threshold candidates found.")
        return
    print_threshold_pareto(results, t_slo_ms=args.slo, homo_cost_kusd=homo_cost or 0.0)
    write_json_output(args.out, [dataclasses.asdict(result) for result in results])


def cmd_compare_routers(args) -> None:
    cdf = load_cdf(args.cdf)
    gpu = resolve_gpu(args.gpu_short, A100_80GB)
    pool_configs = [
        PoolConfig("short", gpu, args.n_s, args.b_short),
        PoolConfig("long", gpu, args.n_l, args.long_max_ctx),
    ]
    workload = PoissonWorkload(
        args.lam,
        CdfWorkload(cdf, seed=args.seed),
        n_requests=args.n_req,
        seed=args.seed,
    )
    arrivals = workload.generate()
    rows = []
    print(
        f"\n  Router comparison  λ={args.lam:.0f} req/s  "
        f"n_s={args.n_s} n_l={args.n_l}  SLO={args.slo}ms"
    )
    print(f"  {'Router':30s} {'P99 TTFT':>10} {'SLO%':>8} {'Util':>8}")
    print(f"  {'-' * 60}")
    for name, kwargs in _router_variants(args.b_short):
        fleet = Fleet(
            FleetConfig(
                pools=list(pool_configs), router_type=name, router_kwargs=kwargs
            )
        )
        result = fleet.run(list(arrivals))
        p99 = result.p99_ttft_ms()
        slo = result.slo_compliance(args.slo) * 100
        util = result.mean_utilisation() * 100
        print(f"  {name:30s} {p99:>9.1f}ms {slo:>7.2f}% {util:>7.1f}%")
        rows.append(
            {
                "router": name,
                "p99_ttft_ms": round(p99, 2),
                "slo_compliance_pct": round(slo, 2),
                "mean_utilisation_pct": round(util, 2),
            }
        )
    write_json_output(args.out, rows)


def cmd_serve(args) -> None:
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


def _best_sweep_result(results):
    return min(
        (row for row in results if row.slo_met),
        key=lambda row: row.cost_per_hr,
        default=results[0],
    )


def _run_gpu_comparison_mode(args, cdf, gpu_names: list[str]) -> None:
    gpus = [(name, resolve_gpu(name, A100_80GB)) for name in gpu_names]
    print("\nWhat-if: GPU type comparison")
    print(
        f"  CDF: {args.cdf}  B_short={args.b_short:,}  SLO={args.slo}ms"
        f"  long_max_ctx={args.long_max_ctx:,}"
    )
    all_results = {
        name: _whatif_single(
            cdf, args.lam_range, gpu, gpu, args.b_short, args.slo, args.long_max_ctx
        )
        for name, gpu in gpus
    }
    rows = _print_gpu_comparison_rows(args.lam_range, gpus, all_results)
    if len(gpus) > 1:
        _print_gpu_cost_ratios(args.lam_range, gpus, all_results)
    write_json_output(args.out, rows)


def _print_gpu_comparison_rows(lam_range, gpus, all_results):
    col_w = 28
    header_gpu = "".join(f"  {name.upper():^{col_w}}" for name, _ in gpus)
    print(f"\n  {'λ':>8}  {header_gpu}")
    print(
        f"  {' ':>8}  "
        + "".join(
            f"  {'GPUs':>5} {'$K/yr':>8} {'$/hr':>7} {'P99':>7} {'SLO':>4}"
            for _ in gpus
        )
    )
    print(f"  {'-' * (10 + len(gpus) * (col_w + 2))}")
    rows = []
    for lam in lam_range:
        line = f"  {lam:>8.0f}  "
        row = {"lam": lam}
        for name, _gpu in gpus:
            best = dict(all_results[name]).get(lam)
            if not best:
                line += f"  {'N/A':>{col_w}}"
                continue
            line += (
                f"  {best.total_gpus:>5} ${best.annualised_cost_kusd:>6.0f}K"
                f" ${best.cost_per_hr:>5.0f}/hr {best.p99_ttft_short_ms:>5.0f}ms"
                f" {'✓' if best.slo_met else '✗':>4}"
            )
            row[name] = {
                "total_gpus": best.total_gpus,
                "cost_per_hr": round(best.cost_per_hr, 2),
                "annualised_cost_kusd": round(best.annualised_cost_kusd, 1),
                "p99_ttft_short_ms": round(best.p99_ttft_short_ms, 1),
                "slo_met": best.slo_met,
            }
        print(line)
        rows.append(row)
    return rows


def _print_gpu_cost_ratios(lam_range, gpus, all_results) -> None:
    base_name = gpus[0][0]
    print(f"\n  Cost ratio vs {base_name.upper()} (at each λ):")
    for lam in lam_range:
        base = dict(all_results[base_name]).get(lam)
        if not base:
            continue
        line = f"  {lam:>8.0f}  "
        for name, _gpu in gpus:
            if name == base_name:
                line += f"  {'1.00×':>10}"
                continue
            other = dict(all_results[name]).get(lam)
            if not other:
                line += f"  {'N/A':>10}"
                continue
            ratio = other.annualised_cost_kusd / base.annualised_cost_kusd
            gpu_ratio = other.total_gpus / base.total_gpus
            line += f"  {ratio:>5.2f}× cost ({gpu_ratio:+.0%} GPUs)"
        print(line)


def _run_lambda_sweep_mode(args, cdf) -> None:
    gpu_short = resolve_gpu(args.gpu_short, A100_80GB)
    gpu_long = resolve_gpu(args.gpu_long, A100_80GB)
    print(f"\nWhat-if analysis: sweep λ over {args.lam_range}")
    print(
        f"  B_short={args.b_short:,}  SLO={args.slo}ms  GPU: {gpu_short.name}/{gpu_long.name}"
    )
    print(
        f"\n  {'λ':>8} {'n_s':>5} {'n_l':>5} {'total':>7} {'$K/yr':>10}"
        f" {'γ*':>6} {'P99_s':>8} {'P99_l':>8}"
    )
    print(f"  {'-' * 60}")
    rows = []
    for lam, best in _whatif_single(
        cdf,
        args.lam_range,
        gpu_short,
        gpu_long,
        args.b_short,
        args.slo,
        args.long_max_ctx,
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
    write_json_output(args.out, rows)


def _router_variants(b_short: int):
    return [
        ("LengthRouter", {"threshold": b_short}),
        ("CompressAndRouteRouter", {"B_short": b_short, "gamma": 1.5}),
        ("RandomRouter", {}),
    ]


def _serialize_sweep_result(result) -> dict[str, object]:
    return {
        "gamma": result.gamma,
        "n_s": result.n_s,
        "n_l": result.n_l,
        "total_gpus": result.total_gpus,
        "annualised_cost_kusd": round(result.annualised_cost_kusd, 1),
        "p99_ttft_short_ms": round(result.p99_ttft_short_ms, 1),
        "p99_ttft_long_ms": round(result.p99_ttft_long_ms, 1),
        "slo_met": result.slo_met,
        "source": result.source,
    }
