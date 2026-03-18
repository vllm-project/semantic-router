"""Async wrappers that run fleet_sim optimizer/simulator in a thread pool.

All heavy computation is executed via asyncio.to_thread so the FastAPI
event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import storage
from .models import (
    HistogramBucket,
    JobRequest,
    JobType,
    OptResult,
    PoolResult,
    SimResult,
    SweepPoint,
    WhatifPoint,
    WhatifResult,
    WorkloadRef,
)

_DATA_DIR = Path(__file__).parent.parent / "data"
_GPU_MAP: dict[str, Any] = {}  # populated lazily


def _gpu(key: str):
    """Resolve a GPU key to a GpuProfile.

    Accepts profile names from the hardware catalog
    names (h200, b200, gb200, gb300, l40s, b60).  For new hardware names a
    default ComputedProfile is returned using Llama-3.1-70B / TP=8 / fp16 as
    the representative model — callers that need a different model should call
    ProfileBuilder directly.
    """
    from fleet_sim.gpu_profiles import ProfileBuilder, ServingConfig
    from fleet_sim.gpu_profiles.profiles import A10G, A100_80GB, H100_80GB
    from fleet_sim.hardware import get_hardware
    from fleet_sim.models import LLAMA_3_1_70B

    _LEGACY = {
        "a100": A100_80GB,
        "a100_80gb": A100_80GB,
        "h100": H100_80GB,
        "h100_80gb": H100_80GB,
        "a10g": A10G,
    }

    k = key.lower().replace("-", "_").replace(" ", "_")
    if k in _LEGACY:
        return _LEGACY[k]

    # Try hardware catalog → build ComputedProfile with default model
    try:
        hw = get_hardware(k)
        return ProfileBuilder().build(
            hw=hw,
            model=LLAMA_3_1_70B,
            cfg=ServingConfig(tp=8, dtype_bytes=2),
        )
    except KeyError:
        pass

    all_keys = list(_LEGACY.keys()) + ["h200", "b200", "gb200", "gb300", "l40s", "b60"]
    raise ValueError(f"Unknown GPU profile '{key}'. Available: {all_keys}")


def _load_cdf(workload_ref: WorkloadRef) -> list:
    if workload_ref.type == "builtin":
        name = workload_ref.name or "azure"
        fname = f"{name}_cdf.json"
        path = _DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Built-in workload '{name}' not found at {path}")
        raw = json.loads(path.read_text())
        return raw["cdf"] if isinstance(raw, dict) else raw
    elif workload_ref.type == "trace":
        if not workload_ref.trace_id:
            raise ValueError("trace_id required for trace workload")
        meta = storage.get_trace_meta(workload_ref.trace_id)
        if not meta:
            raise ValueError(f"Trace '{workload_ref.trace_id}' not found")
        # For trace workloads, build a CDF from the uploaded file's token lengths
        return _cdf_from_trace(
            storage.trace_upload_path(workload_ref.trace_id),
            meta.get("format", "jsonl"),
        )
    raise ValueError(f"Unknown workload type '{workload_ref.type}'")


def _cdf_from_trace(path: Path, fmt: str) -> list:
    """Build a [[threshold, frac], ...] CDF from a trace file."""
    import csv

    totals = []
    if fmt == "csv":
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pt = int(row.get("prompt_tokens", row.get("l_in", 0)))
                ot = int(row.get("generated_tokens", row.get("l_out", 0)))
                totals.append(pt + ot)
    else:  # jsonl / semantic_router
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pt = int(r.get("prompt_tokens", r.get("l_in", 0)))
                ot = int(r.get("generated_tokens", r.get("l_out", 0)))
                totals.append(pt + ot)
    if not totals:
        raise ValueError("Trace file is empty or cannot be parsed")
    totals.sort()
    n = len(totals)
    # Build CDF with ~50 representative bins
    step = max(1, n // 50)
    cdf = []
    seen = set()
    for i in range(0, n, step):
        t = totals[i]
        if t not in seen:
            cdf.append([t, round((i + 1) / n, 6)])
            seen.add(t)
    cdf.append([totals[-1], 1.0])
    return cdf


def _make_histogram(values: list[float], n_bins: int = 20) -> list[HistogramBucket]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [HistogramBucket(lo=int(lo), hi=int(hi) + 1, count=len(values))]
    width = (hi - lo) / n_bins
    buckets: list[HistogramBucket] = []
    for i in range(n_bins):
        b_lo = lo + i * width
        b_hi = lo + (i + 1) * width
        count = sum(1 for v in values if b_lo <= v < b_hi)
        buckets.append(HistogramBucket(lo=int(b_lo), hi=int(b_hi), count=count))
    return buckets


# ── Synchronous worker functions (run in thread pool) ─────────────────────────


def _run_optimize(params) -> OptResult:
    from fleet_sim.optimizer import FleetOptimizer

    cdf = _load_cdf(params.workload)
    gpu_s = _gpu(params.gpu_short)
    gpu_l = _gpu(params.gpu_long)

    opt = FleetOptimizer(
        gpu_short=gpu_s,
        gpu_long=gpu_l,
        B_short=params.b_short,
        t_slo_ms=params.slo_ms,
        long_max_ctx=params.long_max_ctx,
        p_c=params.p_c,
        node_avail=params.node_avail,
    )
    step = params.gamma_step
    gammas = [
        round(params.gamma_min + i * step, 2)
        for i in range(int((params.gamma_max - params.gamma_min) / step) + 1)
    ]

    result = opt.optimize(
        cdf=cdf,
        lam=params.lam,
        gammas=gammas,
        n_sim_requests=params.n_sim_requests,
        verbose=False,
    )

    # OptimizationReport stores results in .analytical and .simulated lists
    all_results = result.analytical + result.simulated
    sweep_pts = [
        SweepPoint(
            gamma=r.gamma,
            n_s=r.n_s,
            n_l=r.n_l,
            total_gpus=r.total_gpus,
            annual_cost_kusd=r.annualised_cost_kusd,
            p99_short_ms=r.p99_ttft_short_ms,
            p99_long_ms=r.p99_ttft_long_ms,
            slo_met=r.slo_met,
            source=r.source,
        )
        for r in all_results
    ]

    best_r = result.best_simulated or result.best_analytical
    # Baseline = cheapest γ=1.0 analytical result (no C+R compression)
    baseline_r = next((r for r in result.analytical if r.gamma == 1.0), None)
    baseline_cost = (
        baseline_r.annualised_cost_kusd if baseline_r else best_r.annualised_cost_kusd
    )
    savings = 0.0
    if baseline_cost > 0:
        savings = round(
            (baseline_cost - best_r.annualised_cost_kusd) / baseline_cost * 100, 1
        )

    best_pt = SweepPoint(
        gamma=best_r.gamma,
        n_s=best_r.n_s,
        n_l=best_r.n_l,
        total_gpus=best_r.total_gpus,
        annual_cost_kusd=best_r.annualised_cost_kusd,
        p99_short_ms=best_r.p99_ttft_short_ms,
        p99_long_ms=best_r.p99_ttft_long_ms,
        slo_met=best_r.slo_met,
        source=best_r.source,
    )

    # DES produces SweepResult summaries, not a raw FleetSimResult, so
    # sim_validation cannot be reconstructed here without re-running the DES.
    return OptResult(
        best=best_pt,
        sweep=sweep_pts,
        baseline_annual_cost_kusd=baseline_cost,
        savings_pct=savings,
        sim_validation=None,
    )


def _run_simulate(params) -> SimResult:
    from fleet_sim import Fleet, FleetConfig, PoolConfig
    from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

    fleet_cfg = _resolve_fleet(params.fleet, params.fleet_id)
    cdf = _load_cdf(params.workload)
    workload = CdfWorkload(cdf)
    arrivals = PoissonWorkload(
        lam=params.lam, length_gen=workload, n_requests=params.n_requests
    ).generate()

    pools = [
        PoolConfig(
            pool_id=p.pool_id, gpu=_gpu(p.gpu), n_gpus=p.n_gpus, max_ctx=p.max_ctx
        )
        for p in fleet_cfg.pools
    ]
    router_type = _router_name(fleet_cfg.router, fleet_cfg.compress_gamma)
    router_kwargs = {}
    if fleet_cfg.router == "compress_route" and fleet_cfg.compress_gamma:
        router_kwargs = {"gamma": fleet_cfg.compress_gamma}

    fc = FleetConfig(pools=pools, router_type=router_type, router_kwargs=router_kwargs)
    result = Fleet(fc).run(arrivals)
    return _fleet_sim_result_to_model(result, params.slo_ms)


def _run_whatif(params) -> WhatifResult:
    from fleet_sim import Fleet, FleetConfig, PoolConfig
    from fleet_sim.workload.synthetic import CdfWorkload, PoissonWorkload

    fleet_cfg = _resolve_fleet(params.fleet, params.fleet_id)
    cdf = _load_cdf(params.workload)
    workload = CdfWorkload(cdf)
    pools = [
        PoolConfig(
            pool_id=p.pool_id, gpu=_gpu(p.gpu), n_gpus=p.n_gpus, max_ctx=p.max_ctx
        )
        for p in fleet_cfg.pools
    ]
    router_type = _router_name(fleet_cfg.router, fleet_cfg.compress_gamma)
    router_kwargs = {}
    if fleet_cfg.router == "compress_route" and fleet_cfg.compress_gamma:
        router_kwargs = {"gamma": fleet_cfg.compress_gamma}
    fc = FleetConfig(pools=pools, router_type=router_type, router_kwargs=router_kwargs)

    points = []
    slo_break = None
    for lam in sorted(params.lam_range):
        arrivals = PoissonWorkload(
            lam=lam, length_gen=workload, n_requests=params.n_requests
        ).generate()
        result = Fleet(fc).run(arrivals)
        slo_c = result.slo_compliance(params.slo_ms)
        p = WhatifPoint(
            lam=lam,
            fleet_p99_ttft_ms=result.p99_ttft_ms(),
            fleet_slo_compliance=slo_c,
            fleet_mean_utilisation=result.mean_utilisation(),
            annual_cost_kusd=result.annualised_cost_usd() / 1000,
        )
        points.append(p)
        if slo_c < 0.99 and slo_break is None:
            slo_break = lam

    return WhatifResult(points=points, slo_break_lam=slo_break)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_fleet(inline, fleet_id):
    if inline:
        return inline
    if fleet_id:
        data = storage.get_fleet(fleet_id)
        if not data:
            raise ValueError(f"Fleet '{fleet_id}' not found")
        from .models import FleetConfigIn, PoolConfigIn

        return FleetConfigIn(
            name=data["name"],
            pools=[PoolConfigIn(**p) for p in data["pools"]],
            router=data.get("router", "length"),
            compress_gamma=data.get("compress_gamma"),
        )
    raise ValueError("Either fleet_id or inline fleet config required")


def _router_name(router: str, gamma: float | None) -> str:
    mapping = {
        "length": "LengthRouter",
        "model": "ModelRouter",
        "random": "RandomRouter",
        "least_loaded": "LeastLoadedRouter",
        "compress_route": "CompressAndRouteRouter",
    }
    return mapping.get(router, "LengthRouter")


def _fleet_sim_result_to_model(result, slo_ms: float) -> SimResult:
    pool_results = []
    for pid, pool in result.pools.items():
        pool_results.append(
            PoolResult(
                pool_id=pid,
                gpu=pool.gpu.name,
                n_gpus=pool.n_gpus,
                p50_ttft_ms=result.p50_ttft_ms(pid),
                p99_ttft_ms=result.p99_ttft_ms(pid),
                p99_queue_wait_ms=result.p99_queue_wait_ms(pid),
                slo_compliance=result.slo_compliance(slo_ms, pid),
                mean_utilisation=result.mean_utilisation(pid),
                cost_per_hr=pool.cost_per_hr(),
            )
        )

    ttfts_ms = [r.ttft * 1000 for r in result.completed if r.ttft is not None]
    hist = _make_histogram(ttfts_ms, n_bins=30)

    times = [r.end_time for r in result.completed if r.end_time]
    arr_times = [r.arrival_time for r in result.completed]
    duration = (max(times) - min(arr_times)) if times and arr_times else 1.0
    actual_rps = len(times) / duration if duration > 0 else 0.0

    return SimResult(
        total_gpus=result.total_gpus(),
        annual_cost_kusd=result.annualised_cost_usd() / 1000,
        fleet_p99_ttft_ms=result.p99_ttft_ms(),
        fleet_p50_ttft_ms=result.p50_ttft_ms(),
        fleet_slo_compliance=result.slo_compliance(slo_ms),
        fleet_mean_utilisation=result.mean_utilisation(),
        pools=pool_results,
        ttft_histogram=hist,
        arrival_rate_actual=round(actual_rps, 2),
    )


# ── Async dispatch ────────────────────────────────────────────────────────────


async def run_job(job_id: str, request: JobRequest) -> None:
    """Execute a job in a thread pool and write results to storage."""

    data = storage.get_job(job_id)
    data["status"] = "running"
    data["started_at"] = datetime.now(timezone.utc).isoformat()
    storage.save_job(job_id, data)

    try:
        if request.type == JobType.optimize:
            result = await asyncio.to_thread(_run_optimize, request.optimize)
            data["result_optimize"] = result.model_dump()
        elif request.type == JobType.simulate:
            result = await asyncio.to_thread(_run_simulate, request.simulate)
            data["result_simulate"] = result.model_dump()
        elif request.type == JobType.whatif:
            result = await asyncio.to_thread(_run_whatif, request.whatif)
            data["result_whatif"] = result.model_dump()
        data["status"] = "done"
    except Exception as exc:
        data["status"] = "failed"
        data["error"] = str(exc)

        data["completed_at"] = datetime.now(timezone.utc).isoformat()
    storage.save_job(job_id, data)
