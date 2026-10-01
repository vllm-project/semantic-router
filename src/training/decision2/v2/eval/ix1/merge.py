"""Merge IX1 shard results into one kit ``results.jsonl`` with row accounting.

    python3 -m v2.eval.ix1.merge --panel <rows dir>/panel.json --run <run root> --out <run root>/merged

``<run root>/shard-<k>`` holds each kit runner's ``results.jsonl``, ``environment.json`` and the
launcher's ``start_epoch`` / ``end_epoch``. The last record of a run ID is its final one (the kit
appends retried errors on resume). The merge fails unless every panel run ID has exactly one final
record and no final record is an error (``--allow-errors`` keeps errors that persisted through a
resume; both scorers count them as failures). Writes ``results.jsonl`` (final records, panel order),
``latency.json`` (private: per-request wall-time percentiles) and ``receipt.json`` (public-safe:
counts, digests and GPU-hours only).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def final_records(lines: list[str]) -> tuple[dict[str, dict[str, Any]], int]:
    """Final record per run ID, plus the number of superseded (retried) records."""
    final: dict[str, dict[str, Any]] = {}
    superseded = 0
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if record["run_id"] in final:
            if final[record["run_id"]]["status"] != "error":
                raise ValueError(f"run ID {record['run_id']} answered twice")
            superseded += 1
        final[record["run_id"]] = record
    return final, superseded


def panel_order(rows_dir: Path, panel: dict[str, Any]) -> list[tuple[str, int]]:
    order = []
    for shard in panel["shards"]:
        with gzip.open(rows_dir / shard["file"], "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    e = json.loads(line)["_evaluation"]
                    order.append((e["run_id"], e["catalog_id"]))
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="keep final errors (after a resume) as failures instead of refusing",
    )
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text())
    order = panel_order(args.panel.parent, panel)
    shard_dirs = sorted(
        args.run.glob("shard-*"), key=lambda p: int(p.name.split("-")[1])
    )
    if len(shard_dirs) != len(panel["shards"]):
        raise SystemExit(
            f"{len(shard_dirs)} shard runs for {len(panel['shards'])} shards"
        )
    final: dict[str, dict[str, Any]] = {}
    superseded = 0
    gpu_seconds = 0.0
    environments = set()
    for shard in shard_dirs:
        lines = (shard / "results.jsonl").read_text(encoding="utf-8").splitlines()
        records, retried = final_records(lines)
        if set(records) & set(final):
            raise SystemExit(f"{shard.name} repeats run IDs of another shard")
        final.update(records)
        superseded += retried
        for start_file in shard.glob("start_epoch*"):
            end_file = shard / start_file.name.replace("start", "end", 1)
            gpu_seconds += int(end_file.read_text()) - int(start_file.read_text())
        env = json.loads((shard / "environment.json").read_text())
        environments.add(json.dumps(env["model_source"], sort_keys=True))
    wanted = [run_id for run_id, _ in order]
    if len(set(wanted)) != len(wanted) or set(final) != set(wanted):
        raise SystemExit(
            f"row accounting failed: {len(set(wanted) - set(final))} missing, "
            f"{len(set(final) - set(wanted))} extra"
        )
    statuses = collections.Counter(final[r]["status"] for r in wanted)
    if statuses["error"]:
        raise SystemExit(
            f"{statuses['error']} final errors; resume the shard runs first"
        )
    if len(environments) != 1:
        raise SystemExit("shards ran different model sources")
    args.out.mkdir(parents=True, exist_ok=True)
    merged = args.out / "results.jsonl"
    digest = hashlib.sha256()
    with merged.open("x", encoding="utf-8") as stream:
        for run_id in wanted:
            line = (
                json.dumps(final[run_id], ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            stream.write(line)
            digest.update(line.encode("utf-8"))
    reasons = collections.Counter(
        final[r].get("error", "") for r in wanted if final[r]["status"] != "ok"
    )
    per_benchmark = collections.defaultdict(collections.Counter)
    for run_id, catalog in order:
        per_benchmark[str(catalog)][final[run_id]["status"]] += 1
    walls = [
        final[r]["model_request_wall_ms"] for r in wanted if final[r]["status"] == "ok"
    ]
    latency = {
        "ok_requests": len(walls),
        "median_ms": statistics.median(walls),
        "p95_ms": _percentile(walls, 0.95),
        "p99_ms": _percentile(walls, 0.99),
        "max_ms": max(walls),
        "mean_ms": statistics.fmean(walls),
    }
    (args.out / "latency.json").write_text(
        json.dumps(latency, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "schema": "ix1-run-receipt/1",
        "label": "independent provisional 0.2.1 reproduction",
        "model_source": json.loads(environments.pop()),
        "panel_run_ids_sha256": panel["run_ids_sha256"],
        "rows": len(wanted),
        "statuses": dict(sorted(statuses.items())),
        "non_ok_reasons": dict(sorted(reasons.items())),
        "superseded_error_records": superseded,
        "per_benchmark_statuses": {
            k: dict(v) for k, v in sorted(per_benchmark.items())
        },
        "shards": len(shard_dirs),
        "gpu_hours": round(gpu_seconds / 3600, 4),
        "results_sha256": digest.hexdigest(),
    }
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: receipt[k] for k in ("rows", "statuses", "gpu_hours")}))


if __name__ == "__main__":
    main()
