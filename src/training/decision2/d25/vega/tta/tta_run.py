"""Paired OFF / ON answers of one d3 package for the permutation-average flag, in one process.

Every request is answered twice back to back on the same loaded model, first with
``permutation_average`` off (the released behaviour), then on, exactly as ``d3_engine.D3Engine`` answers a kit
request (prepare, run, respond; over-limit questions are ``unsupported``). Output: ``<out>/off/results.jsonl``
and ``<out>/on/results.jsonl`` as kit result records, so ``d25.vega.eval.proxy.score`` reads them.

    python -m d25.vega.tta.tta_run --package <dir> --rows a.jsonl.gz b.jsonl.gz --shard 3/12 --out <dir>

``--shard i/n`` takes the i-th of n deterministic, cost-balanced parts of the rows (largest first onto the
least-loaded part), so parallel jobs given the same rows cover them exactly once. Rows may carry ``images``
(paths relative to their rows file, or anything ``d3_runtime.load_image`` reads).
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import sys
import time
from pathlib import Path

TOKENS_PER_SECOND = 3600.0
PASS_SECONDS = 0.06


def read(path: Path) -> list[dict]:
    """Kit rows; proxy rows without ``_evaluation`` (the vision proxies) get one from their ``id``."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        row.setdefault(
            "_evaluation",
            {
                "run_id": row["id"],
                "group_id": row["id"],
                "catalog_id": path.parent.name,
                "dataset": path.parent.name,
                "track": row.get("family"),
            },
        )
    return rows


def describe(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def cost(row: dict) -> float:
    """Rough seconds for OFF + ON of one request (characters / 3.6 per token, reversed copies for choices)."""
    state = len(describe(row.get("state")))
    tokens = 0.0
    for question in row["questions"].values():
        criteria = question.get("criteria") or {}
        size = (
            state
            + len(describe(question.get("instructions") or ""))
            + len(describe(criteria))
        ) / 3.6 + 60
        many = question.get("type") == "choice" and len(criteria) >= 2
        tokens += size * (3 if many else 2)
    passes = 2 * (-(-len(row["questions"]) // 8))
    return (
        tokens / TOKENS_PER_SECOND
        + passes * PASS_SECONDS
        + 1600 * len(row.get("images") or []) / TOKENS_PER_SECOND
    )


def shard(rows: list[dict], index: int, count: int) -> list[dict]:
    order = sorted(range(len(rows)), key=lambda i: (-cost(rows[i]), i))
    heap = [(0.0, part) for part in range(count)]
    owner = {}
    for i in order:
        load, part = heapq.heappop(heap)
        owner[i] = part
        heapq.heappush(heap, (load + cost(rows[i]), part))
    return [row for i, row in enumerate(rows) if owner[i] == index]


def answer(model, row: dict, flag: bool, base: Path) -> dict:
    model.permutation_average = flag
    images = row.get("images")
    if images:
        images = [
            (
                str(base / image)
                if isinstance(image, str)
                and not image.startswith(("http:", "https:", "data:"))
                else image
            )
            for image in images
        ]
    record = {
        key: row["_evaluation"].get(key)
        for key in ("run_id", "catalog_id", "dataset", "group_id", "track")
    }
    model.synchronize()
    started = time.perf_counter()
    try:
        prepared = model.prepare(row["state"], row["questions"], images or None)
        over = [
            e["message"]
            for e in prepared.errors.values()
            if e["error"] == "max_length_exceeded"
        ]
        if over:
            record.update(status="unsupported", error=over[0])
        elif prepared.errors:
            record.update(status="error", error=json.dumps(prepared.errors)[:500])
        else:
            probabilities, tokens = model.run(prepared)
            response = model.respond(prepared, probabilities, tokens)
            failed = {
                k: a["message"] for k, a in response["answers"].items() if "error" in a
            }
            if failed:
                record.update(status="error", error=json.dumps(failed)[:500])
            else:
                record.update(status="ok", response=response)
    except Exception as exc:  # noqa: BLE001 - one failed request must not stop the run
        record.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
    model.synchronize()
    record["total_wall_ms"] = (time.perf_counter() - started) * 1000
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--rows", required=True, nargs="+", type=Path)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args(argv)
    index, count = (int(x) for x in args.shard.split("/"))
    rows, bases = [], {}
    for path in args.rows:
        for row in read(path):
            bases[row["_evaluation"]["run_id"]] = path.parent
            rows.append(row)
    mine = shard(rows, index, count)
    sys.path.insert(0, str(args.package.absolute()))
    from d3_runtime import D3

    model = D3.from_pretrained(
        str(args.package), device=args.device, permutation_average=True
    )
    warmup = model.warmup()
    for arm in ("off", "on"):
        (args.out / arm).mkdir(parents=True, exist_ok=True)
    streams = {
        arm: open(args.out / arm / "results.jsonl", "w", encoding="utf-8")
        for arm in ("off", "on")
    }
    started = time.time()
    totals = {"off": 0.0, "on": 0.0}
    for number, row in enumerate(mine, 1):
        base = bases[row["_evaluation"]["run_id"]]
        for arm, flag in (("off", False), ("on", True)):
            record = answer(model, row, flag, base)
            totals[arm] += record["total_wall_ms"]
            streams[arm].write(json.dumps(record, ensure_ascii=False) + "\n")
        if number % args.progress_every == 0 or number == len(mine):
            for stream in streams.values():
                stream.flush()
            print(
                json.dumps(
                    {
                        "done": number,
                        "of": len(mine),
                        "elapsed_s": round(time.time() - started),
                        "off_ms": round(totals["off"]),
                        "on_ms": round(totals["on"]),
                    }
                ),
                flush=True,
            )
    for stream in streams.values():
        stream.close()
    summary = {
        "package": str(args.package),
        "shard": args.shard,
        "rows_total": len(rows),
        "rows": len(mine),
        "warmup_seconds": round(warmup, 1),
        "wall_ms": {k: round(v, 1) for k, v in totals.items()},
        "provenance_on": {
            k: v
            for k, v in model.provenance().items()
            if k in ("policy", "permutation_average", "model_sha256")
        },
        "runtime": model.runtime_info(),
        "kernels": model.kernels,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        json.dumps({k: summary[k] for k in ("rows", "wall_ms", "warmup_seconds")}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
