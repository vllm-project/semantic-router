"""Failure slicing for MoM evaluation diagnostics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench.mom_eval.common import write_json


def slice_failures(raw_dir: Path) -> dict[str, Any]:
    slices: list[dict[str, Any]] = []
    failures_path = raw_dir / "failures.jsonl"
    if not failures_path.is_file():
        return {"failure_slices": [], "total_failures": 0}

    by_benchmark: Counter[str] = Counter()
    by_error: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    total = 0

    for line in failures_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        total += 1
        by_benchmark[str(record.get("benchmark_id", "unknown"))] += 1
        by_error[str(record.get("error_type", "unknown"))] += 1
        by_language[str(record.get("language", "unknown"))] += 1

    for benchmark_id, count in by_benchmark.items():
        slices.append({"dimension": "benchmark_id", "value": benchmark_id, "count": count})
    for error_type, count in by_error.items():
        slices.append({"dimension": "error_type", "value": error_type, "count": count})
    for language, count in by_language.items():
        slices.append({"dimension": "language", "value": language, "count": count})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in slices:
        grouped[item["dimension"]].append(item)

    return {"failure_slices": slices, "grouped": grouped, "total_failures": total}


def write_failure_slices(output_path: Path, raw_dir: Path) -> dict[str, Any]:
    payload = slice_failures(raw_dir)
    write_json(output_path, payload)
    return payload
