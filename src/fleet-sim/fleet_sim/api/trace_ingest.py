"""Shared trace ingest helpers for uploads and startup seeding."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

from . import storage
from .models import HistogramBucket, TraceFormat, TraceStats

LOG = logging.getLogger(__name__)

_DEFAULT_SEED_TRACE_SPECS = (
    (
        "router_decisions.semantic_router.jsonl",
        "Router decisions sample",
        TraceFormat.semantic_router,
    ),
    ("generic_chat_mix.jsonl", "Generic chat mix", TraceFormat.jsonl),
    ("batch_spike_requests.csv", "Batch spike requests", TraceFormat.csv),
)


def _default_seed_trace_dir_candidates() -> list[Path]:
    """Return runtime locations that commonly contain bundled trace samples."""

    module_path = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "examples" / "trace_samples",
        Path.cwd() / "src" / "fleet-sim" / "examples" / "trace_samples",
        Path.cwd().parent / "fleet-sim" / "examples" / "trace_samples",
        module_path.parents[2] / "examples" / "trace_samples",
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, int(len(sorted_vals) * p / 100) - 1)
    return sorted_vals[idx]


def _histogram(values: list[int], n_bins: int = 20) -> list[HistogramBucket]:
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


def _parse_json_lines(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def parse_records(path: Path, fmt: str) -> list[dict]:
    records = []
    if fmt == TraceFormat.csv.value:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records
    return _parse_json_lines(path)


def compute_trace_stats(records: list[dict]) -> TraceStats:
    prompts, outputs, totals, timestamps = [], [], [], []
    routing: dict[str, int] = {}

    for record in records:
        prompt_tokens = int(record.get("prompt_tokens", record.get("l_in", 0)) or 0)
        output_tokens = int(record.get("generated_tokens", record.get("l_out", 0)) or 0)
        timestamp = float(record.get("timestamp", record.get("time", 0)) or 0)
        model = (
            record.get("selected_model")
            or record.get("x_vsr_selected_model")
            or record.get("model")
            or record.get("routed_to")
            or record.get("model_id")
        )
        prompts.append(prompt_tokens)
        outputs.append(output_tokens)
        totals.append(prompt_tokens + output_tokens)
        timestamps.append(timestamp)
        if model:
            routing[str(model)] = routing.get(str(model), 0) + 1

    prompts.sort()
    outputs.sort()
    totals.sort()
    timestamps.sort()
    n_requests = len(records)
    duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 1.0
    arrival_rate = n_requests / duration if duration > 0 else 0.0
    total_count = sum(routing.values()) or 1
    routing_dist = {
        key: round(value / total_count, 4) for key, value in routing.items()
    }

    return TraceStats(
        n_requests=n_requests,
        duration_s=round(duration, 2),
        arrival_rate_rps=round(arrival_rate, 2),
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


def persist_trace_bytes(content: bytes, fmt: TraceFormat, name: str) -> dict:
    trace_id = storage.new_id()
    dest = storage.trace_upload_path(trace_id)
    dest.write_bytes(content)
    try:
        records = parse_records(dest, fmt.value)
        stats = compute_trace_stats(records)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    meta = {
        "id": trace_id,
        "name": name,
        "format": fmt.value,
        "upload_time": storage.now_iso(),
        "n_requests": stats.n_requests,
        "stats": stats.model_dump(),
    }
    storage.save_trace_meta(trace_id, meta)
    return meta


def import_trace_file(path: Path, fmt: TraceFormat, name: str | None = None) -> dict:
    return persist_trace_bytes(path.read_bytes(), fmt, name or path.name)


def resolve_seed_trace_dir() -> Path:
    override = os.environ.get("VLLM_SR_SIM_SEED_TRACE_DIR")
    if override:
        return Path(override)
    for candidate in _default_seed_trace_dir_candidates():
        if candidate.exists():
            return candidate
    return _default_seed_trace_dir_candidates()[0]


def seed_example_traces_if_enabled() -> int:
    enabled = os.environ.get("VLLM_SR_SIM_SEED_EXAMPLE_TRACES", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return 0
    if storage.trace_seed_completed():
        return 0
    if storage.list_traces():
        storage.mark_trace_seed_completed()
        return 0

    seed_dir = resolve_seed_trace_dir()
    if not seed_dir.exists():
        checked = ", ".join(
            str(candidate) for candidate in _default_seed_trace_dir_candidates()
        )
        LOG.warning(
            "Fleet Sim trace seed directory not found: %s (checked: %s)",
            seed_dir,
            checked,
        )
        return 0

    seeded = 0
    for file_name, display_name, fmt in _DEFAULT_SEED_TRACE_SPECS:
        source = seed_dir / file_name
        if not source.exists():
            LOG.warning("Fleet Sim seed trace missing: %s", source)
            continue
        import_trace_file(source, fmt, display_name)
        seeded += 1

    if seeded:
        LOG.info("Seeded %d Fleet Sim trace samples from %s", seeded, seed_dir)
    storage.mark_trace_seed_completed()
    return seeded
