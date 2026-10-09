"""Run an engine (anchor adapter or code-readout checkpoint) over proxy or public-sample rows.

Sharded and resumable: rows are assigned to shard ``int(sha256(run_id)[:8], 16) % shards``; each shard
appends kit-format results ``{run_id, catalog_id, status, response, total_wall_ms}`` to
``<out>/results.shard<i>.jsonl`` and skips run ids already answered (``ok`` / ``unsupported``).

    python -m d25.vega.eval.proxy.run --anchor pplx --rows DIR_OR_FILES... --out OUT --shard 0 --shards 7 --device cuda:0
    python -m d25.vega.eval.proxy.run --spawn 7 ...    # one worker per GPU (cuda:0..6) in this pod
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from d25.vega.eval.proxy.common import read_jsonl


def row_files(specs):
    out = []
    for s in specs:
        p = Path(s)
        out += sorted(p.glob("**/*.jsonl.gz")) if p.is_dir() else [p]
    return [f for f in out if "protected-items" not in f.name]


def shard_of(run_id: str, shards: int) -> int:
    return int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16) % shards


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") in ("ok", "unsupported"):
                ids.add(r["run_id"])
    return ids


def worker(a):
    from d25.vega.eval.proxy.anchors import make_engine

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / f"results.shard{a.shard}.jsonl"
    done = done_ids(res_path)
    rows = [r for f in row_files(a.rows) for r in read_jsonl(f)]
    todo = [
        r
        for r in rows
        if shard_of(r["_evaluation"]["run_id"], a.shards) == a.shard
        and r["_evaluation"]["run_id"] not in done
    ]
    if a.limit:
        todo = todo[: a.limit]
    print(
        json.dumps(
            {
                "event": "start",
                "shard": a.shard,
                "rows": len(rows),
                "todo": len(todo),
                "done": len(done),
            }
        ),
        flush=True,
    )
    if not todo:
        return
    engine = make_engine(
        a.anchor, device=a.device, options=dict(kv.split("=", 1) for kv in a.option)
    )
    t0 = time.time()
    with open(res_path, "a", encoding="utf-8") as f:

        def emit(r, status, response=None, ms=0.0, err=None):
            rec = {
                "run_id": r["_evaluation"]["run_id"],
                "catalog_id": r["_evaluation"]["catalog_id"],
                "status": status,
                "total_wall_ms": ms,
            }
            if response is not None:
                rec["response"] = response
            if err:
                rec["error"] = err[:500]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if hasattr(engine, "batch"):
            for i in range(0, len(todo), a.chunk):
                chunk = todo[i : i + a.chunk]
                t = time.time()
                for r, (status, response, err) in zip(chunk, engine.batch(chunk)):
                    emit(
                        r, status, response, 1000 * (time.time() - t) / len(chunk), err
                    )
                f.flush()
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "shard": a.shard,
                            "done": i + len(chunk),
                            "todo": len(todo),
                            "elapsed_s": round(time.time() - t0),
                        }
                    ),
                    flush=True,
                )
        else:
            for i, r in enumerate(todo):
                t = time.time()
                try:
                    response = engine(r["state"], r["questions"])
                    emit(r, "ok", response, 1000 * (time.time() - t))
                except Exception as e:  # noqa: BLE001
                    unsupported = (
                        type(e).__name__ in ("Unsupported", "UnsupportedInput")
                        or "too long" in str(e).lower()
                        or "exceeds" in str(e).lower()
                    )
                    emit(
                        r,
                        "unsupported" if unsupported else "error",
                        None,
                        1000 * (time.time() - t),
                        f"{type(e).__name__}: {e}",
                    )
                    if not unsupported and "out of memory" in str(e).lower():
                        traceback.print_exc()
                        raise
                if i % 50 == 0:
                    f.flush()
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "shard": a.shard,
                                "done": i + 1,
                                "todo": len(todo),
                                "elapsed_s": round(time.time() - t0),
                            }
                        ),
                        flush=True,
                    )
    print(
        json.dumps(
            {"event": "finished", "shard": a.shard, "seconds": round(time.time() - t0)}
        ),
        flush=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--spawn",
        type=int,
        default=0,
        help="spawn this many workers (cuda:0..n-1) and wait",
    )
    ap.add_argument("--option", action="append", default=[])
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.spawn:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        procs = []
        for i in range(a.spawn):
            args = [
                sys.executable,
                "-m",
                "d25.vega.eval.proxy.run",
                "--anchor",
                a.anchor,
                "--rows",
                *a.rows,
                "--out",
                a.out,
                "--shard",
                str(i),
                "--shards",
                str(a.spawn),
                "--device",
                f"cuda:{i}",
                "--chunk",
                str(a.chunk),
                "--limit",
                str(a.limit),
            ] + [x for kv in a.option for x in ("--option", kv)]
            log = open(Path(a.out) / f"worker{i}.log", "a")
            procs.append(
                subprocess.Popen(
                    args, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()
                )
            )
        codes = [p.wait() for p in procs]
        print(json.dumps({"event": "spawn-done", "codes": codes}), flush=True)
        sys.exit(max(codes))
    worker(a)


if __name__ == "__main__":
    main()
