"""Run an Omni checkpoint over the public vision suite or a private-part proxy.

A suite directory holds ``rows.jsonl.gz`` (evaluation rows, ``vision_format`` docstring) with
image references relative to the directory. One process runs one shard on one device and appends a
kit-style record per row to ``OUT/shards/results-KKK-of-NNN.jsonl``; a restart resumes. ``merge``
checks that every row finished and writes ``OUT/results.jsonl`` (suite order) and
``OUT/scores.json`` (``d25.omni.suite.score.score_suite`` plus per-benchmark status diagnostics).

    python -m d25.omni.eval.run_suite run --ckpt CKPT --suite SUITE --out OUT --shard 0 --num-shards 8
    python -m d25.omni.eval.run_suite merge --ckpt CKPT --suite SUITE --out OUT --num-shards 8

A row is ``ok`` when every question was answered, ``unsupported`` when any question was rejected
by the checkpoint contract (scored as unanswered) and ``error`` otherwise (retried on resume).
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any

from d25.omni.common import vision_format
from d25.omni.eval import shards
from d25.omni.model import checkpoint

STEM = "results"
ENGINE_NAME = "d25-omni-vision-code-readout"


def load_rows(suite: Path, rows_file: str | None = None) -> list[dict[str, Any]]:
    path = Path(rows_file) if rows_file else suite / "rows.jsonl.gz"
    rows = list(shards.read_jsonl(path))
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate row ids")
    return rows


def row_record(row: dict[str, Any], qids: list[str], outcomes: list) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": row["id"],
        "family": row.get("family"),
        "split": row.get("split"),
        "engine": ENGINE_NAME,
    }
    if "_evaluation" in row:
        record.update(row["_evaluation"])
    statuses = {outcome.status for outcome in outcomes}
    tokens = {qid: outcome.input_tokens for qid, outcome in zip(qids, outcomes)}
    if statuses == {"ok"}:
        answers = {
            qid: vision_format.to_answer(row["questions"][qid], outcome.probabilities)
            for qid, outcome in zip(qids, outcomes)
        }
        record.update(
            status="ok",
            response={"answers": answers},
            probabilities={
                qid: outcome.probabilities for qid, outcome in zip(qids, outcomes)
            },
            input_tokens=tokens,
        )
    else:
        status = "error" if "error" in statuses else "unsupported"
        reasons = {qid: o.reason for qid, o in zip(qids, outcomes) if o.status != "ok"}
        record.update(status=status, error=reasons, input_tokens=tokens)
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    from d25.omni.eval.engine import VisionCodeReadoutModel

    suite, out = Path(args.suite), Path(args.out)
    rows = [
        row
        for row in load_rows(suite, args.rows)
        if shards.shard_of(row["id"], args.num_shards) == args.shard
    ]
    path = shards.shard_path(out, STEM, args.shard, args.num_shards)
    done = shards.finished(path)
    todo = [row for row in rows if row["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    status = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "done_before": len(done),
    }
    if not todo:
        return {**status, "ran": 0}
    engine = VisionCodeReadoutModel(
        args.ckpt,
        device=args.device,
        max_pixels=args.max_pixels,
        token_budget=args.token_budget,
        max_batch_size=args.max_batch_size,
        readout_dtype=args.readout_dtype,
    )
    counts: collections.Counter = collections.Counter()
    started = time.time()
    writer = shards.Appender(path)
    try:
        for offset in range(0, len(todo), args.chunk):
            chunk = todo[offset : offset + args.chunk]
            requests, owners = [], []
            for row in chunk:
                for qid, request in vision_format.requests(row):
                    requests.append(request)
                    owners.append((row["id"], qid))
            outcomes = engine.score(requests, root=suite)
            grouped: dict[str, tuple[list[str], list]] = {
                row["id"]: ([], []) for row in chunk
            }
            for (row_id, qid), outcome in zip(owners, outcomes):
                grouped[row_id][0].append(qid)
                grouped[row_id][1].append(outcome)
            for row in chunk:
                record = row_record(row, *grouped[row["id"]])
                record["checkpoint"] = str(args.ckpt)
                writer.write(record)
                counts[record["status"]] += 1
            shards.write_json(
                out / "shards" / f"status-{args.shard:03d}.json",
                {
                    **status,
                    "counts": dict(counts),
                    "elapsed_seconds": round(time.time() - started, 1),
                },
            )
    finally:
        writer.close()
    return {**status, "ran": sum(counts.values()), "counts": dict(counts)}


def diagnostics(
    rows: list[dict[str, Any]], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Per-benchmark status counts and mean input tokens per question."""
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        benchmark = (
            (row.get("metadata") or {}).get("benchmark")
            or row.get("family")
            or "unknown"
        )
        result = results[row["id"]]
        table[benchmark]["rows"] += 1
        table[benchmark][result["status"]] += 1
        for tokens in (result.get("input_tokens") or {}).values():
            table[benchmark]["questions"] += 1
            table[benchmark]["tokens"] += tokens
    return {
        name: {
            **{key: value for key, value in counts.items() if key != "tokens"},
            "mean_input_tokens": (
                counts["tokens"] / counts["questions"] if counts["questions"] else 0.0
            ),
        }
        for name, counts in sorted(table.items())
    }


def merge_outputs(args: argparse.Namespace) -> dict[str, Any]:
    suite, out = Path(args.suite), Path(args.out)
    rows = load_rows(suite, args.rows)
    results = shards.merge(out, STEM, args.num_shards)
    missing = [row["id"] for row in rows if row["id"] not in results]
    errors = [
        row["id"] for row in rows if results.get(row["id"], {}).get("status") == "error"
    ]
    if missing or errors:
        raise SystemExit(
            f"incomplete run: {len(missing)} rows missing, {len(errors)} errors; rerun the shards"
        )
    with (out / "results.jsonl.tmp").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(results[row["id"]], ensure_ascii=False) + "\n")
    (out / "results.jsonl.tmp").replace(out / "results.jsonl")
    summary: dict[str, Any] = {
        "checkpoint": str(args.ckpt),
        "suite": str(suite),
        "rows": len(rows),
        "status": dict(
            collections.Counter(results[row["id"]]["status"] for row in rows)
        ),
        "results_sha256": checkpoint.file_sha256(out / "results.jsonl"),
        "diagnostics": diagnostics(rows, results),
    }
    manifest = suite / "manifest.json"
    if manifest.exists():
        summary["suite_manifest_sha256"] = checkpoint.file_sha256(manifest)
    try:
        from d25.omni.suite import score as suite_score
    except ImportError:
        summary["scorer"] = (
            "unavailable: d25.omni.suite.score not installed; diagnostics only"
        )
    else:
        answers = {
            row["id"]: results[row["id"]]["response"]["answers"]
            for row in rows
            if results[row["id"]]["status"] == "ok"
        }
        try:
            summary["scores"] = suite_score.score_suite(rows, answers)
            summary["scorer"] = "d25.omni.suite.score.score_suite"
        except ValueError as error:
            summary["scorer"] = (
                f"d25.omni.suite.score refused these rows ({error}); diagnostics only"
            )
    shards.write_json(out / "scores.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "merge"):
        command = sub.add_parser(name)
        command.add_argument("--ckpt", required=True)
        command.add_argument(
            "--suite", required=True, help="directory with rows.jsonl.gz and images/"
        )
        command.add_argument(
            "--rows",
            help="row file overriding SUITE/rows.jsonl.gz (images stay relative to SUITE)",
        )
        command.add_argument("--out", required=True)
        command.add_argument("--num-shards", type=int, default=1)
    run_parser = sub.choices["run"]
    run_parser.add_argument("--shard", type=int, default=0)
    run_parser.add_argument("--device")
    run_parser.add_argument("--max-pixels", type=int, default=vision_format.MAX_PIXELS)
    run_parser.add_argument("--token-budget", type=int, default=65_536)
    run_parser.add_argument("--max-batch-size", type=int, default=64)
    run_parser.add_argument(
        "--readout-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    run_parser.add_argument(
        "--chunk",
        type=int,
        default=256,
        help="rows per scoring call (flush granularity)",
    )
    run_parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "run":
        if not 0 <= args.shard < args.num_shards:
            raise SystemExit("--shard must be in [0, --num-shards)")
        print(run(args), flush=True)
    else:
        summary = merge_outputs(args)
        print({key: summary[key] for key in ("rows", "status", "scorer")}, flush=True)


if __name__ == "__main__":
    main()
