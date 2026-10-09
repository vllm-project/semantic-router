"""Run a code-readout checkpoint over the Decision Index 0.3 public suite and score it with the kit.

    python -m d25.vega.eval.run_suite run --ckpt CKPT --suite-dir SUITE --kit KIT --out RUN_DIR --devices 0-6
    python -m d25.vega.eval.run_suite score --suite-dir SUITE --kit KIT --out RUN_DIR

1. plan (CPU, cached under --plan-root): every in-edition request of the suite (the 442 excluded requests
   are skipped unless --include-excluded; --index-only keeps the 37 index benchmarks), each question
   rendered and tokenized once;
2. one worker process per GPU over token-balanced shards; each appends kit-format result rows to its
   shard and skips finished requests on restart (a request is ``unsupported`` when any of its question
   prompts is over the model's input limit; nothing is truncated);
3. merge into ``RUN_DIR/results.jsonl`` with the fields and statuses of ``decision_index.runner``
   (compact rows: no payload or raw output) plus ``environment.json`` and ``status.json``;
4. score with the kit (``decision_index.pipeline.score_run``, edition 0.3), the same code as
   ``python -m decision_index score --edition 0.3``.

Batched timings are amortized: each question gets its token share of its batch's device time.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from d25.vega.eval import batching

EDITION = "0.3"
ENGINE_NAME = "d25-vega-code-readout"
LATENCY = (
    "Batched suite run: each request's time is its questions' token share of the device-synchronized wall "
    "time of the length-sorted batches they ran in; not a single-request latency."
)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def use_kit(kit: str) -> None:
    kit = str(Path(kit).resolve())
    if kit not in sys.path:
        sys.path.insert(0, kit)


def index_benchmarks() -> set[int]:
    from decision_index.scoring import index02

    return {n for area in index02.spec(EDITION)["areas"] for n in area["benchmarks"]}


def suite_rows(args):
    from decision_index.suite.io import Suite, read_jsonl

    if args.rows:
        yield from read_jsonl(args.rows)
        return
    suite = Suite(args.suite_dir, EDITION)
    yield from suite.rows(apply_exclusions=not args.include_excluded)


def plan_dir(args, codec) -> Path:
    from decision_index.suite.io import Suite

    source = (
        {"rows": str(Path(args.rows).resolve())}
        if args.rows
        else {"suite": Suite(args.suite_dir, EDITION).verify(strict=True)["sha256"]}
    )
    key = {
        "version": batching.PLAN_VERSION,
        "edition": EDITION,
        **source,
        "codec": batching.codec_key(codec),
        "index_only": args.index_only,
        "include_excluded": args.include_excluded,
        "catalog": sorted(args.catalog or []),
    }
    return Path(args.plan_root) / f"suite-{batching.stable_sha(key)[:16]}", key


def build_plan(args, log=print) -> Path:
    from d25.vega.eval.engine import PromptCodec

    codec = PromptCodec(args.ckpt, prompt=args.prompt, revision=args.revision)
    directory, key = plan_dir(args, codec)
    if (directory / "meta.json").exists():
        return directory
    tmp = directory.with_name(directory.name + f".tmp{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    keep = index_benchmarks() if args.index_only else None
    catalog = set(args.catalog or [])
    flat, requests = [], []
    for row in suite_rows(args):
        e = row["_evaluation"]
        n = e["catalog_id"]
        if (keep is not None and n not in keep) or (catalog and n not in catalog):
            continue
        qs = []
        for qk, q in row["questions"].items():
            keys = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
            qs.append([qk, q["type"], keys])
            flat.append({"state": row["state"], "question": q})
        requests.append({"e": e, "qs": qs, "q0": len(flat) - len(qs)})
    log(
        json.dumps(
            {
                "event": "plan",
                "requests": len(requests),
                "questions": len(flat),
                "dir": str(directory),
            }
        )
    )
    lengths, _ = batching.tokenize(
        flat, str(codec.dir), codec.prompt, tmp, args.cpu_workers, log=log
    )
    del flat
    with open(tmp / "requests.jsonl", "w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    per_request_max = [
        int(lengths[r["q0"] : r["q0"] + len(r["qs"])].max()) for r in requests
    ]
    bench = collections.defaultdict(lambda: [0, 0, 0])
    for r, m in zip(requests, per_request_max):
        b = bench[r["e"]["catalog_id"]]
        b[0] += 1
        b[1] += int(lengths[r["q0"] : r["q0"] + len(r["qs"])].sum())
        b[2] = max(b[2], m)
    meta = {
        "key": key,
        "requests": len(requests),
        "questions": int(len(lengths)),
        "tokens": int(lengths.sum()),
        "max_tokens": int(lengths.max()) if len(lengths) else 0,
        "requests_over": {
            str(t): sum(m > t for m in per_request_max)
            for t in (4096, 8192, 16384, 32768, 65536, 131072)
        },
        "benchmarks": {
            str(n): {"requests": v[0], "tokens": v[1], "max_tokens": v[2]}
            for n, v in sorted(bench.items())
        },
        "created_utc": stamp(),
    }
    batching.atomic_json(tmp / "meta.json", meta)
    if directory.exists():
        return directory
    tmp.rename(directory)
    return directory


def load_plan(directory: Path):
    import numpy as np

    meta = json.loads((directory / "meta.json").read_text())
    requests = list(batching.read_jsonl(directory / "requests.jsonl"))
    tokens = np.load(directory / "tokens.npy", mmap_mode="r")
    lengths = np.load(directory / "lengths.npy")
    offsets = np.load(directory / "offsets.npy")
    return meta, requests, tokens, lengths, offsets


def shard_of(requests, lengths, n: int) -> list[int]:
    costs = [int(lengths[r["q0"] : r["q0"] + len(r["qs"])].sum()) for r in requests]
    return batching.lpt_shards(costs, [r["e"]["run_id"] for r in requests], n)


def model_options(args) -> dict:
    opts = {
        "prompt": args.prompt,
        "attention_mode": args.attention_mode,
        "max_length": args.max_length,
        "temperature": args.temperature,
        "readout_dtype": args.readout_dtype,
        "revision": args.revision,
        "max_batch_tokens": args.max_batch_tokens,
        "max_batch_size": args.max_batch_size,
    }
    return {k: v for k, v in opts.items() if v is not None}


def cmd_worker(args) -> None:
    import torch

    from d25.vega.common import decision_format as df
    from d25.vega.eval.engine import CodeReadoutModel
    from decision_index.engines import validate

    directory = Path(args.plan)
    meta, requests, tokens, lengths, offsets = load_plan(directory)
    shards = shard_of(requests, lengths, args.num_shards)
    mine = [r for r, s in zip(requests, shards) if s == args.shard]
    if args.limit:
        mine = sorted(
            mine, key=lambda r: hashlib.sha256(r["e"]["run_id"].encode()).hexdigest()
        )[: args.limit]
    out = Path(args.out) / "shards" / f"shard-{args.shard:03d}"
    out.mkdir(parents=True, exist_ok=True)
    results_path, questions_path = out / "results.jsonl", out / "questions.jsonl"
    batching.truncate_partial(results_path)
    batching.truncate_partial(questions_path)
    done = {r["run_id"]: r["status"] for r in batching.read_jsonl(results_path)}
    partial = collections.defaultdict(dict)
    for q in batching.read_jsonl(questions_path):
        partial[q["r"]][q["q"]] = (q["p"], q["ms"])
    todo = [r for r in mine if done.get(r["e"]["run_id"]) not in ("ok", "unsupported")]
    started_all = time.time()

    def status(event, **kw):
        batching.atomic_json(
            out / "status.json",
            {"time": stamp(), "event": event, "shard": args.shard, **kw},
        )

    status("loading", requests=len(mine), todo=len(todo))
    model = CodeReadoutModel(args.ckpt, device=args.device, **model_options(args))
    env = {
        "engine": args.engine_name,
        "engine_options": {"checkpoint": args.ckpt, **model_options(args)},
        "model_source": model.provenance(),
        **model.runtime(),
        "loaded_seconds": model.loaded_seconds,
        "plan": str(directory),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "latency": LATENCY,
    }
    batching.atomic_json(out / "environment.json", env)
    results_f = open(results_path, "a", encoding="utf-8")
    questions_f = open(questions_path, "a", encoding="utf-8")
    counts = collections.Counter(done.values())

    def write_row(r, row_status, response=None, error=None, wall_ms=0.0, first=None):
        row = {
            **r["e"],
            "started_utc": first or stamp(),
            "engine": args.engine_name,
            "status": row_status,
        }
        if response is not None:
            row["response"] = response
        if error is not None:
            row["error"] = error
        row.update(
            completed_utc=stamp(), total_wall_ms=wall_ms, model_request_wall_ms=wall_ms
        )
        results_f.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        counts[row_status] += 1

    pending, items = {}, []
    for r in todo:
        rid = r["e"]["run_id"]
        qlens = [int(lengths[r["q0"] + j]) for j in range(len(r["qs"]))]
        longest = max(qlens)
        if model.codec.over_limit(longest):
            write_row(
                r,
                "unsupported",
                error=f"a question prompt has {longest} tokens, over the {model.max_length}-token "
                "limit; no input was truncated",
            )
            continue
        got = partial.get(rid, {})
        pending[rid] = {
            "r": r,
            "left": sum(1 for q in r["qs"] if q[0] not in got),
            "probs": dict(got),
            "first": None,
        }
        for j, q in enumerate(r["qs"]):
            if q[0] not in got:
                items.append((rid, j))
    results_f.flush()

    def finish(rid):
        p = pending.pop(rid)
        r = p["r"]
        answers, wall = {}, 0.0
        questions = {}
        for qk, qtype, keys in r["qs"]:
            probs, ms = p["probs"][qk]
            wall += ms
            q = (
                {"type": qtype, "criteria": {k: None for k in keys}}
                if qtype == "choice"
                else {"type": "noul"}
            )
            questions[qk] = q
            answers[qk] = df.to_answer(q, probs)
        tokens_used = sum(int(lengths[r["q0"] + j]) for j in range(len(r["qs"])))
        response = {
            "model": args.engine_name,
            "answers": answers,
            "usage": {"input_tokens": tokens_used},
        }
        try:
            validate(questions, response)
            write_row(
                r, "ok", response=response, wall_ms=round(wall, 3), first=p["first"]
            )
        except Exception as exc:  # noqa: BLE001
            write_row(
                r,
                "error",
                error=f"{type(exc).__name__}: {exc}",
                wall_ms=round(wall, 3),
                first=p["first"],
            )

    for rid in [rid for rid, p in pending.items() if p["left"] == 0]:
        finish(rid)
    results_f.flush()
    seq_len = [int(lengths[pending[rid]["r"]["q0"] + j]) for rid, j in items]
    batches = model.batches(seq_len)
    status(
        "running", requests=len(mine), todo_questions=len(items), batches=len(batches)
    )
    processed_tokens, device_seconds, last = 0, 0.0, time.time()

    def run_batch(batch):
        nonlocal processed_tokens, device_seconds
        seqs, counts_ = [], []
        for b in batch:
            rid, j = items[b]
            r = pending[rid]["r"]
            k = r["q0"] + j
            seqs.append(tokens[offsets[k] : offsets[k] + lengths[k]])
            counts_.append(len(r["qs"][j][2]))
        t0 = time.perf_counter()
        try:
            probs = model.probabilities(seqs, counts_)
            model.synchronize()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                rid, _ = items[batch[0]]
                p = pending.pop(rid)
                write_row(
                    p["r"], "error", error="OutOfMemoryError on a single question"
                )
                return
            half = len(batch) // 2
            run_batch(batch[:half])
            run_batch(batch[half:])
            return
        elapsed = time.perf_counter() - t0
        device_seconds += elapsed
        total = sum(len(s) for s in seqs)
        processed_tokens += total
        now = stamp()
        for b, pr, s in zip(batch, probs, seqs):
            rid, j = items[b]
            if rid not in pending:
                continue
            p = pending[rid]
            qk = p["r"]["qs"][j][0]
            ms = elapsed * 1000 * len(s) / total
            p["probs"][qk] = (pr, ms)
            p["first"] = p["first"] or now
            p["left"] -= 1
            questions_f.write(
                json.dumps(
                    {"r": rid, "q": qk, "p": pr, "ms": round(ms, 3)},
                    separators=(",", ":"),
                )
                + "\n"
            )
            if p["left"] == 0:
                finish(rid)
        questions_f.flush()
        results_f.flush()

    for n, batch in enumerate(batches):
        run_batch(batch)
        if time.time() - last > 30 or n == len(batches) - 1:
            last = time.time()
            status(
                "progress",
                batches_done=n + 1,
                batches=len(batches),
                counts=dict(counts),
                tokens=processed_tokens,
                device_seconds=round(device_seconds, 1),
                tokens_per_second=round(processed_tokens / max(device_seconds, 1e-9)),
                wall_seconds=round(time.time() - started_all, 1),
            )
    for rid in list(pending):
        write_row(
            pending.pop(rid)["r"],
            "error",
            error="question results missing after the shard finished",
        )
    results_f.close()
    questions_f.close()
    summary = {
        "requests": len(mine),
        "counts": dict(counts),
        "tokens": processed_tokens,
        "device_seconds": round(device_seconds, 1),
        "wall_seconds": round(time.time() - started_all, 1),
        "load_seconds": round(model.loaded_seconds, 1),
    }
    status("complete", **summary)
    if not counts.get("error"):
        batching.atomic_json(out / "DONE", summary)


def merge(args, meta, requests, shards) -> dict:
    out = Path(args.out)
    rows, envs, summaries = {}, [], []
    for s in sorted(set(shards)):
        d = out / "shards" / f"shard-{s:03d}"
        for row in batching.read_jsonl(d / "results.jsonl"):
            if row["run_id"] not in rows or rows[row["run_id"]]["status"] == "error":
                rows[row["run_id"]] = row
        if (d / "environment.json").exists():
            envs.append(json.loads((d / "environment.json").read_text()))
        if (d / "status.json").exists():
            summaries.append(json.loads((d / "status.json").read_text()))
    missing = [r["e"]["run_id"] for r in requests if r["e"]["run_id"] not in rows]
    tmp = out / "results.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in requests:
            row = rows.get(r["e"]["run_id"])
            if row is not None:
                f.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    tmp.replace(out / "results.jsonl")
    counts = collections.Counter(row["status"] for row in rows.values())
    env = dict(envs[0]) if envs else {}
    env.update(
        rows_path=meta["key"],
        shards=len(set(shards)),
        devices=[e.get("device") for e in envs],
        loaded_seconds=max((e.get("loaded_seconds", 0) for e in envs), default=0),
    )
    batching.atomic_json(out / "environment.json", env)
    device_seconds = sum(s.get("device_seconds", 0) for s in summaries)
    wall = max((s.get("wall_seconds", 0) for s in summaries), default=0)
    load = max((s.get("load_seconds", 0) for s in summaries), default=0)
    state = {
        "time": stamp(),
        "event": "complete" if not missing and not counts.get("error") else "partial",
        "engine": args.engine_name,
        "completed": len(rows),
        "planned": len(requests),
        "missing": len(missing),
        "counts": dict(counts),
        "device_seconds": round(device_seconds, 1),
        "worker_wall_seconds": round(wall, 1),
        "load_seconds": round(load, 1),
        "gpus": len(set(shards)),
        "gpu_hours": round(
            sum(s.get("wall_seconds", 0) + s.get("load_seconds", 0) for s in summaries)
            / 3600,
            3,
        ),
    }
    batching.atomic_json(out / "status.json", state)
    return state


def score(args) -> dict:
    from decision_index.pipeline import score_run
    from decision_index.suite.io import Suite

    out = Path(args.out)
    suite = Suite(args.suite_dir, EDITION)
    scores = score_run(suite, out / "results.jsonl", args.engine_name, out)
    summary = {
        "edition": scores["edition"],
        "decision_index": scores["decision_index"],
        "raw_index": scores["raw_index"],
        "complete": scores["complete"],
        "completed": scores["completed"],
        "counts": scores["counts"],
        "areas": {
            a["id"]: {k: a[k] for k in ("skill", "raw", "coverage")}
            for a in scores["areas"]
        },
        "benchmarks": {
            n: {
                k: b.get(k)
                for k in (
                    "dataset",
                    "requests",
                    "answered",
                    "unsupported",
                    "errors",
                    "index_raw",
                    "index_skill",
                    "coverage",
                    "chance",
                    "in_index",
                )
            }
            for n, b in scores["benchmarks"].items()
        },
    }
    if (out / "status.json").exists():
        summary["run"] = json.loads((out / "status.json").read_text())
    batching.atomic_json(out / "summary.json", summary)
    return summary


def cmd_run(args) -> None:
    started = time.time()
    directory = build_plan(args)
    meta, requests, _, lengths, _ = load_plan(directory)
    print(
        json.dumps(
            {
                "event": "plan_ready",
                "dir": str(directory),
                **{
                    k: meta[k]
                    for k in (
                        "requests",
                        "questions",
                        "tokens",
                        "max_tokens",
                        "requests_over",
                    )
                },
            }
        ),
        flush=True,
    )
    if args.plan_only:
        return
    devices = [f"cuda:{d}" for d in batching.parse_ids(args.devices, 1)]
    num_shards = args.num_shards or len(devices)
    shards = batching.parse_ids(args.shards, num_shards)
    if len(shards) > len(devices):
        raise SystemExit("more shards than devices in this pod")
    common = [
        "--plan",
        str(directory),
        "--out",
        args.out,
        "--num-shards",
        str(num_shards),
        "--ckpt",
        args.ckpt,
        "--engine-name",
        args.engine_name,
        "--kit",
        args.kit,
    ]
    for flag in (
        "prompt",
        "attention_mode",
        "max_length",
        "temperature",
        "readout_dtype",
        "revision",
        "max_batch_tokens",
        "max_batch_size",
        "limit",
    ):
        value = getattr(args, flag)
        if value is not None:
            common += ["--" + flag.replace("_", "-"), str(value)]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pending = [
        s
        for s in shards
        if not (Path(args.out) / "shards" / f"shard-{s:03d}" / "DONE").exists()
    ]
    codes = (
        batching.spawn_workers(
            "d25.vega.eval.run_suite",
            pending,
            devices[: len(pending)],
            common,
            Path(args.out) / "logs",
        )
        if pending
        else []
    )
    all_shards = list(range(num_shards))
    if not all(
        (Path(args.out) / "shards" / f"shard-{s:03d}" / "DONE").exists()
        for s in all_shards
    ):
        print(json.dumps({"event": "shards_incomplete", "codes": codes}), flush=True)
        raise SystemExit(1 if any(codes) else 0)
    state = merge(args, meta, requests, shard_of(requests, lengths, num_shards))
    state["pod_wall_seconds"] = round(time.time() - started, 1)
    print(json.dumps(state), flush=True)
    if not args.no_score:
        print(json.dumps(score(args), indent=1), flush=True)


def cmd_merge(args) -> None:
    meta, requests, _, lengths, _ = load_plan(Path(args.plan))
    print(
        json.dumps(
            merge(args, meta, requests, shard_of(requests, lengths, args.num_shards))
        ),
        flush=True,
    )
    if not args.no_score:
        print(json.dumps(score(args), indent=1), flush=True)


def cmd_score(args) -> None:
    print(json.dumps(score(args), indent=1), flush=True)


def add_model_args(p) -> None:
    p.add_argument(
        "--ckpt", required=True, help="checkpoint dir, stock base dir, or Hub id"
    )
    p.add_argument("--revision")
    p.add_argument("--prompt", choices=("d25-vega", "pplx"))
    p.add_argument("--attention-mode", choices=("causal", "noncausal_full_attention"))
    p.add_argument(
        "--max-length",
        type=int,
        help="input limit in tokens (default: the checkpoint's)",
    )
    p.add_argument("--temperature", type=float)
    p.add_argument("--readout-dtype", choices=("float32", "bfloat16"))
    p.add_argument("--max-batch-tokens", type=int)
    p.add_argument("--max-batch-size", type=int)
    p.add_argument("--limit", type=int, help="smoke test: at most N requests per shard")
    p.add_argument("--engine-name", default=ENGINE_NAME)


def main(argv=None) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    add_model_args(r)
    r.add_argument("--kit", required=True)
    r.add_argument("--suite-dir", required=True)
    r.add_argument("--rows", help="kit rows file instead of the suite (smoke tests)")
    r.add_argument("--out", required=True)
    r.add_argument("--plan-root", default="/data/d25/vega/eval/plans")
    r.add_argument(
        "--index-only",
        action="store_true",
        help="only the 37 benchmarks of the public index",
    )
    r.add_argument(
        "--include-excluded",
        action="store_true",
        help="also run the 442 excluded requests",
    )
    r.add_argument("--catalog", type=int, nargs="*", help="only these catalog ids")
    r.add_argument("--devices", default="0", help="GPU indices in this pod, e.g. 0-6")
    r.add_argument(
        "--num-shards",
        type=int,
        help="total shards across pods (default: number of devices)",
    )
    r.add_argument("--shards", help="shard ids this pod runs, e.g. 0-6 (default: all)")
    r.add_argument("--cpu-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r.add_argument("--plan-only", action="store_true")
    r.add_argument("--no-score", action="store_true")
    r.set_defaults(func=cmd_run)
    w = sub.add_parser("worker")
    add_model_args(w)
    w.add_argument("--kit", required=True)
    w.add_argument("--plan", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--shard", type=int, required=True)
    w.add_argument("--num-shards", type=int, required=True)
    w.add_argument("--device", required=True)
    w.set_defaults(func=cmd_worker)
    m = sub.add_parser("merge")
    m.add_argument("--kit", required=True)
    m.add_argument("--suite-dir", required=True)
    m.add_argument("--plan", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--num-shards", type=int, required=True)
    m.add_argument("--engine-name", default=ENGINE_NAME)
    m.add_argument("--no-score", action="store_true")
    m.set_defaults(func=cmd_merge)
    s = sub.add_parser("score")
    s.add_argument("--kit", required=True)
    s.add_argument("--suite-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--engine-name", default=ENGINE_NAME)
    s.set_defaults(func=cmd_score)
    args = ap.parse_args(argv)
    use_kit(args.kit)
    args.func(args)


if __name__ == "__main__":
    main()
