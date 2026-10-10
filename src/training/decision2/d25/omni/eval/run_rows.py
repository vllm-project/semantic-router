"""Run an Omni (or pplx-layout) checkpoint over training-format rows, e.g. for teacher labels.

Rows follow the Omni training-row contract (one question per row, ``images`` relative to the
directory of the row file). Output per row: ``{"id", "status", "probs", "input_tokens"}`` (plus
``reason`` when the row was rejected), appended to ``OUT/shards/probs-KKK-of-NNN.jsonl`` and
resumable; ``merge`` writes ``OUT/probs.jsonl`` in input order and ``OUT/summary.json``.

    python -m d25.omni.eval.run_rows run --ckpt CKPT --rows A.jsonl.gz B.jsonl.gz --out OUT --shard 0 --num-shards 8
    python -m d25.omni.eval.run_rows merge --ckpt CKPT --rows A.jsonl.gz B.jsonl.gz --out OUT --num-shards 8
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

STEM = "probs"


def load_rows(paths: list[str]) -> list[tuple[dict[str, Any], Path]]:
    """``(row, image root)`` pairs in input order; ids must be unique across files."""
    rows: list[tuple[dict[str, Any], Path]] = []
    seen: set[str] = set()
    for name in paths:
        path = Path(name)
        for row in shards.read_jsonl(path):
            if row["id"] in seen:
                raise ValueError(f"duplicate row id {row['id']!r} ({path})")
            seen.add(row["id"])
            rows.append((row, path.parent))
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    from d25.omni.eval.engine import VisionCodeReadoutModel

    out = Path(args.out)
    rows = [
        (row, root)
        for row, root in load_rows(args.rows)
        if shards.shard_of(row["id"], args.num_shards) == args.shard
    ]
    path = shards.shard_path(out, STEM, args.shard, args.num_shards)
    done = shards.finished(path)
    todo = [(row, root) for row, root in rows if row["id"] not in done]
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
            requests = [
                {
                    "state": row.get("state"),
                    "question": row["question"],
                    "images": [
                        (
                            ref
                            if ref.startswith("data:") or Path(ref).is_absolute()
                            else str(root / ref)
                        )
                        for ref in row.get("images") or []
                    ],
                }
                for row, root in chunk
            ]
            for (row, _), outcome in zip(chunk, engine.score(requests)):
                record: dict[str, Any] = {
                    "id": row["id"],
                    "status": outcome.status,
                    "probs": outcome.probabilities,
                    "input_tokens": outcome.input_tokens,
                }
                if outcome.reason:
                    record["reason"] = outcome.reason
                writer.write(record)
                counts[outcome.status] += 1
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


def merge_outputs(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    rows = load_rows(args.rows)
    records = shards.merge(out, STEM, args.num_shards)
    missing = [row["id"] for row, _ in rows if row["id"] not in records]
    errors = [
        row["id"]
        for row, _ in rows
        if records.get(row["id"], {}).get("status") == "error"
    ]
    if missing or errors:
        raise SystemExit(
            f"incomplete run: {len(missing)} rows missing, {len(errors)} errors; rerun the shards"
        )
    target = out / "probs.jsonl"
    with target.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as stream:
        for row, _ in rows:
            stream.write(json.dumps(records[row["id"]], ensure_ascii=False) + "\n")
    target.with_suffix(".jsonl.tmp").replace(target)
    decision = checkpoint.read_decision_config(args.ckpt)
    summary = {
        "checkpoint": str(args.ckpt),
        "checkpoint_prompt": decision.get("prompt", "d25-vega"),
        "checkpoint_attention_mode": decision.get("attention_mode", "causal"),
        "rows_files_sha256": {name: checkpoint.file_sha256(name) for name in args.rows},
        "rows": len(rows),
        "status": dict(
            collections.Counter(records[row["id"]]["status"] for row, _ in rows)
        ),
        "probs_sha256": checkpoint.file_sha256(target),
    }
    shards.write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "merge"):
        command = sub.add_parser(name)
        command.add_argument("--ckpt", required=True)
        command.add_argument("--rows", nargs="+", required=True)
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
    run_parser.add_argument("--chunk", type=int, default=512)
    run_parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "run":
        if not 0 <= args.shard < args.num_shards:
            raise SystemExit("--shard must be in [0, --num-shards)")
        print(run(args), flush=True)
    else:
        print(merge_outputs(args), flush=True)


if __name__ == "__main__":
    main()
