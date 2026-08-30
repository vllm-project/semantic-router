"""Safe aggregate artifact for a bounded live concurrency sweep."""

from __future__ import annotations

from collections import defaultdict

from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import percentile


def build_capacity_profile(records: list[ExecutionRecord]) -> dict[str, object]:
    by_level: dict[int, list[ExecutionRecord]] = defaultdict(list)
    for row in records:
        if row.track_id == "capacity" and row.concurrency is not None:
            by_level[row.concurrency].append(row)
    levels: list[dict[str, object]] = []
    for concurrency, rows in sorted(by_level.items()):
        latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
        successes = sum(bool(row.success) for row in rows)
        levels.append(
            {
                "concurrency": concurrency,
                "requests": len(rows),
                "successes": successes,
                "errors": len(rows) - successes,
                "elapsed_seconds": max(
                    (row.load_elapsed_seconds or 0 for row in rows), default=0
                ),
                "throughput_rps": max(
                    (row.throughput_rps or 0 for row in rows), default=0
                ),
                "latency_p50_ms": percentile(latencies, 0.50),
                "latency_p95_ms": percentile(latencies, 0.95),
                "latency_p99_ms": percentile(latencies, 0.99),
                "input_tokens": sum(row.input_tokens or 0 for row in rows),
                "output_tokens": sum(row.output_tokens or 0 for row in rows),
                "runtime_cost_usd": sum(row.runtime_cost or 0 for row in rows),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "bounded-concurrency-sweep",
        "levels": levels,
        "slo": None,
    }
