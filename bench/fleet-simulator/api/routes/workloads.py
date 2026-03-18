"""Routes for built-in CDF workloads bundled with the simulator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from ..models import BuiltinWorkload, CdfPoint, HistogramBucket, TraceStats

router = APIRouter(prefix="/workloads", tags=["Workloads"])

_DATA_DIR = Path(__file__).parent.parent.parent / "data"

_BUILTIN: dict[str, str] = {
    "azure":           "Azure production traces (~78% requests ≤2K tokens)",
    "lmsys":           "LMSYS-Chat-1M single-turn conversations",
    "lmsys_multiturn": "LMSYS-Chat-1M multi-turn conversations",
    "agent_heavy":     "Agent-heavy workload (tool calls, long context)",
}


def _load_cdf(name: str) -> list:
    path = _DATA_DIR / f"{name}_cdf.json"
    if not path.exists():
        raise FileNotFoundError(name)
    raw = json.loads(path.read_text())
    return raw["cdf"] if isinstance(raw, dict) else raw


def _cdf_stats(cdf: list, name: str) -> TraceStats:
    # Synthesise per-request stats from the CDF for display purposes
    # Expand CDF into ~1000 sample points
    samples = []
    prev_frac = 0.0
    prev_thresh = 0
    for thresh, frac in cdf:
        count = int((frac - prev_frac) * 1000)
        for _ in range(max(1, count)):
            mid = (prev_thresh + thresh) // 2
            samples.append(mid)
        prev_frac = frac
        prev_thresh = thresh
    samples.sort()
    n = len(samples)

    def pct(p):
        return samples[max(0, int(n * p / 100) - 1)]

    # Histogram (prompt tokens ≈ 80% of total)
    bins = []
    lo_val = samples[0]
    hi_val = samples[-1]
    width = max(256, (hi_val - lo_val) // 20)
    current = lo_val
    while current <= hi_val:
        nxt = current + width
        count = sum(1 for v in samples if current <= v < nxt)
        bins.append(HistogramBucket(lo=current, hi=nxt, count=count))
        current = nxt

    total_p50 = pct(50)
    return TraceStats(
        n_requests=n,
        duration_s=0.0,
        arrival_rate_rps=0.0,
        p50_prompt_tokens=int(total_p50 * 0.80),
        p95_prompt_tokens=int(pct(95) * 0.80),
        p99_prompt_tokens=int(pct(99) * 0.80),
        p50_output_tokens=int(total_p50 * 0.20),
        p99_output_tokens=int(pct(99) * 0.20),
        p50_total_tokens=total_p50,
        p99_total_tokens=pct(99),
        routing_distribution={},
        prompt_histogram=bins,
        output_histogram=[],
    )


@router.get("", response_model=List[BuiltinWorkload], summary="List built-in workloads")
async def list_workloads():
    result = []
    for name, desc in _BUILTIN.items():
        path = _DATA_DIR / f"{name}_cdf.json"
        result.append(BuiltinWorkload(
            name=name,
            description=desc,
            path=str(path),
            stats=None,
        ))
    return result


@router.get("/{name}/cdf", response_model=List[CdfPoint], summary="Get raw CDF points")
async def get_cdf(name: str):
    try:
        cdf = _load_cdf(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Workload '{name}' not found")
    return [CdfPoint(threshold=int(t), cumulative_frac=float(f)) for t, f in cdf]


@router.get("/{name}/stats", response_model=TraceStats, summary="Get workload statistics")
async def get_workload_stats(name: str):
    try:
        cdf = _load_cdf(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Workload '{name}' not found")
    return _cdf_stats(cdf, name)
