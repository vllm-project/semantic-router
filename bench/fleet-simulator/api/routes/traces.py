"""Trace management routes: upload, list, view, stats, delete."""
from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse

from ..models import (
    HistogramBucket, TraceFormat, TraceInfo, TraceSample, TraceStats,
)
from .. import storage

router = APIRouter(prefix="/traces", tags=["Traces"])


def _percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, int(len(sorted_vals) * p / 100) - 1)
    return sorted_vals[idx]


def _histogram(values: List[int], n_bins: int = 20) -> List[HistogramBucket]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [HistogramBucket(lo=lo, hi=hi + 1, count=len(values))]
    width = max(1, (hi - lo) // n_bins)
    buckets = []
    current = lo
    while current <= hi:
        next_b = current + width
        count = sum(1 for v in values if current <= v < next_b)
        buckets.append(HistogramBucket(lo=current, hi=next_b, count=count))
        current = next_b
    return buckets


def _parse_records(path: Path, fmt: str) -> list[dict]:
    records = []
    if fmt == "csv":
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def _compute_stats(records: list[dict], fmt: str) -> TraceStats:
    prompts, outputs, totals, timestamps = [], [], [], []
    routing: dict[str, int] = {}

    for r in records:
        pt = int(r.get("prompt_tokens", r.get("l_in", 0)) or 0)
        ot = int(r.get("generated_tokens", r.get("l_out", 0)) or 0)
        ts = float(r.get("timestamp", r.get("time", 0)) or 0)
        model = r.get("selected_model") or r.get("x_vsr_selected_model") \
                or r.get("model") or r.get("routed_to") or r.get("model_id")
        prompts.append(pt)
        outputs.append(ot)
        totals.append(pt + ot)
        timestamps.append(ts)
        if model:
            routing[str(model)] = routing.get(str(model), 0) + 1

    prompts.sort(); outputs.sort(); totals.sort(); timestamps.sort()
    n = len(records)
    duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 1.0
    arr_rate = n / duration if duration > 0 else 0.0
    total_count = sum(routing.values()) or 1
    routing_dist = {k: round(v / total_count, 4) for k, v in routing.items()}

    return TraceStats(
        n_requests=n,
        duration_s=round(duration, 2),
        arrival_rate_rps=round(arr_rate, 2),
        p50_prompt_tokens=int(_percentile(prompts, 50)),
        p95_prompt_tokens=int(_percentile(prompts, 95)),
        p99_prompt_tokens=int(_percentile(prompts, 99)),
        p50_output_tokens=int(_percentile(outputs, 50)),
        p99_output_tokens=int(_percentile(outputs, 99)),
        p50_total_tokens=int(_percentile(totals, 50)),
        p99_total_tokens=int(_percentile(totals, 99)),
        routing_distribution=routing_dist,
        prompt_histogram=_histogram(prompts, n_bins=25),
        output_histogram=_histogram(outputs, n_bins=25),
    )


@router.post("", response_model=TraceInfo, summary="Upload a trace file")
async def upload_trace(
    file: UploadFile = File(...),
    fmt: TraceFormat = Query(TraceFormat.jsonl, description="File format"),
):
    trace_id = storage.new_id()
    dest = storage.trace_upload_path(trace_id)
    content = await file.read()
    dest.write_bytes(content)

    # Parse and compute stats
    try:
        records = _parse_records(dest, fmt.value)
        stats = _compute_stats(records, fmt.value)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not parse trace: {exc}")

    meta = {
        "id": trace_id,
        "name": file.filename or trace_id,
        "format": fmt.value,
        "upload_time": storage.now_iso(),
        "n_requests": stats.n_requests,
        "stats": stats.model_dump(),
    }
    storage.save_trace_meta(trace_id, meta)
    return TraceInfo(**{k: v for k, v in meta.items() if k != "stats"},
                     stats=stats)


@router.get("", response_model=List[TraceInfo], summary="List all traces")
async def list_traces():
    return [TraceInfo(**{k: v for k, v in m.items() if k != "stats"},
                      stats=TraceStats(**m["stats"]) if m.get("stats") else None)
            for m in storage.list_traces()]


@router.get("/{trace_id}", response_model=TraceInfo, summary="Get trace metadata")
async def get_trace(trace_id: str):
    m = storage.get_trace_meta(trace_id)
    if not m:
        raise HTTPException(404, "Trace not found")
    return TraceInfo(**{k: v for k, v in m.items() if k != "stats"},
                     stats=TraceStats(**m["stats"]) if m.get("stats") else None)


@router.get("/{trace_id}/sample", response_model=TraceSample, summary="Get sample records")
async def sample_trace(trace_id: str, limit: int = Query(50, ge=1, le=500)):
    m = storage.get_trace_meta(trace_id)
    if not m:
        raise HTTPException(404, "Trace not found")
    path = storage.trace_upload_path(trace_id)
    records = _parse_records(path, m["format"])
    return TraceSample(records=records[:limit], total=len(records))


@router.delete("/{trace_id}", summary="Delete a trace")
async def delete_trace(trace_id: str):
    if not storage.delete_trace(trace_id):
        raise HTTPException(404, "Trace not found")
    return {"deleted": trace_id}
