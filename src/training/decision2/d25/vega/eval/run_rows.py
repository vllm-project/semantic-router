"""Run a code-readout checkpoint over training-format rows (teacher labelling, proxies).

    python -m d25.vega.eval.run_rows run --ckpt CKPT --rows ROWS.jsonl.gz [MORE ...] --out OUT --devices 0-6

Rows follow the ``d25.vega.common.decision_format`` contract; only ``id``, ``state`` and ``question`` are
read (ids must be unique across the inputs). Output ``OUT/probs.jsonl`` (input order), one line per row:

    {"id": ..., "status": "ok", "probs": [p_0, ..., p_{K-1}], "tokens": n}           # options() order
    {"id": ..., "status": "unsupported", "probs": null, "tokens": n}                 # over --max-length

``--write-logits`` adds ``logits`` (masked code logits before temperature). Probabilities use the
checkpoint temperature unless ``--temperature`` is given (``--temperature 1`` for raw softmax).
Plans (tokenized prompts) live in ``OUT/plan-<key>/``; workers append to ``OUT/shards/shard-XXX.jsonl``
and skip finished ids on restart; ``OUT/summary.json`` records counts, throughput and provenance.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path

from d25.vega.eval import batching


def build_plan(args) -> Path:
    import numpy as np

    from d25.vega.eval.engine import PromptCodec, iter_rows, option_count

    codec = PromptCodec(args.ckpt, prompt=args.prompt, revision=args.revision)
    sources = [
        {
            "path": str(Path(p).resolve()),
            "size": Path(p).stat().st_size,
            "mtime": int(Path(p).stat().st_mtime),
        }
        for p in args.rows
    ]
    key = {
        "version": batching.PLAN_VERSION,
        "rows": sources,
        "codec": batching.codec_key(codec),
    }
    directory = Path(args.out) / f"plan-{batching.stable_sha(key)[:16]}"
    if (directory / "meta.json").exists():
        return directory
    tmp = directory.with_name(directory.name + f".tmp{os.getpid()}")
    ids, rows, counts, seen = [], [], [], set()
    for row in iter_rows(args.rows):
        rid = str(row["id"])
        if rid in seen:
            raise ValueError(f"duplicate row id {rid!r}")
        seen.add(rid)
        ids.append(rid)
        counts.append(option_count(row["question"]))
        rows.append({"state": row["state"], "question": row["question"]})
    lengths, _ = batching.tokenize(
        rows, str(codec.dir), codec.prompt, tmp, args.cpu_workers
    )
    del rows
    with open(tmp / "ids.jsonl", "w", encoding="utf-8") as f:
        for rid in ids:
            f.write(json.dumps(rid, ensure_ascii=False) + "\n")
    np.save(tmp / "counts.npy", np.asarray(counts, dtype=np.int32))
    meta = {
        "key": key,
        "rows": len(ids),
        "tokens": int(lengths.sum()),
        "max_tokens": int(lengths.max()) if len(ids) else 0,
        "rows_over": {
            str(t): int((lengths > t).sum()) for t in (4096, 8192, 16384, 32768)
        },
    }
    batching.atomic_json(tmp / "meta.json", meta)
    if not directory.exists():
        tmp.rename(directory)
    return directory


def load_plan(directory: Path):
    import numpy as np

    meta = json.loads((directory / "meta.json").read_text())
    ids = [json.loads(line) for line in open(directory / "ids.jsonl", encoding="utf-8")]
    return (
        meta,
        ids,
        np.load(directory / "tokens.npy", mmap_mode="r"),
        np.load(directory / "lengths.npy"),
        np.load(directory / "offsets.npy"),
        np.load(directory / "counts.npy"),
    )


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

    from d25.vega.eval.engine import CodeReadoutModel

    meta, ids, tokens, lengths, offsets, counts = load_plan(Path(args.plan))
    shards = batching.lpt_shards([int(x) for x in lengths], ids, args.num_shards)
    mine = [i for i, s in enumerate(shards) if s == args.shard]
    out = Path(args.out) / "shards"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard-{args.shard:03d}.jsonl"
    batching.truncate_partial(path)
    done = {r["id"] for r in batching.read_jsonl(path)}
    todo = [i for i in mine if ids[i] not in done]
    model = CodeReadoutModel(args.ckpt, device=args.device, **model_options(args))
    if args.shard == 0:
        batching.atomic_json(
            Path(args.out) / "provenance.json",
            {
                "model_source": model.provenance(),
                **model.runtime(),
                "options": model_options(args),
            },
        )
    f = open(path, "a", encoding="utf-8")
    started, device_seconds, processed = time.time(), 0.0, 0
    run = []
    for i in todo:
        n = int(lengths[i])
        if model.codec.over_limit(n):
            f.write(
                json.dumps(
                    {"id": ids[i], "status": "unsupported", "probs": None, "tokens": n},
                    ensure_ascii=False,
                )
                + "\n"
            )
        else:
            run.append(i)
    f.flush()

    def do(batch):
        nonlocal device_seconds, processed
        seqs = [tokens[offsets[i] : offsets[i] + lengths[i]] for i in batch]
        cnt = [int(counts[i]) for i in batch]
        t0 = time.perf_counter()
        try:
            logits = model.logits(seqs, cnt)
            probs = (logits / model.temperature).softmax(-1).cpu().tolist()
            raw = logits.cpu().tolist() if args.write_logits else None
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            do(batch[: len(batch) // 2])
            do(batch[len(batch) // 2 :])
            return
        device_seconds += time.perf_counter() - t0
        processed += sum(int(lengths[i]) for i in batch)
        for j, i in enumerate(batch):
            line = {
                "id": ids[i],
                "status": "ok",
                "probs": probs[j][: cnt[j]],
                "tokens": int(lengths[i]),
            }
            if raw is not None:
                line["logits"] = raw[j][: cnt[j]]
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        f.flush()

    last = time.time()
    batches = model.batches([int(lengths[i]) for i in run])
    for n, b in enumerate(batches):
        do([run[j] for j in b])
        if time.time() - last > 30 or n == len(batches) - 1:
            last = time.time()
            batching.atomic_json(
                out / f"shard-{args.shard:03d}.status.json",
                {
                    "batches_done": n + 1,
                    "batches": len(batches),
                    "tokens": processed,
                    "device_seconds": round(device_seconds, 1),
                    "tokens_per_second": round(processed / max(device_seconds, 1e-9)),
                    "wall_seconds": round(time.time() - started, 1),
                },
            )
    f.close()
    batching.atomic_json(
        out / f"shard-{args.shard:03d}.DONE",
        {
            "rows": len(mine),
            "tokens": processed,
            "device_seconds": round(device_seconds, 1),
            "wall_seconds": round(time.time() - started, 1),
            "load_seconds": round(model.loaded_seconds, 1),
        },
    )


def merge(args, directory: Path) -> dict:
    meta, ids, *_ = load_plan(directory)
    got = {}
    for path in sorted((Path(args.out) / "shards").glob("shard-*.jsonl")):
        for r in batching.read_jsonl(path):
            got[r["id"]] = r
    missing = [i for i in ids if i not in got]
    tmp = Path(args.out) / "probs.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i in ids:
            if i in got:
                f.write(json.dumps(got[i], ensure_ascii=False) + "\n")
    tmp.replace(Path(args.out) / "probs.jsonl")
    done = [
        json.loads(p.read_text())
        for p in (Path(args.out) / "shards").glob("shard-*.DONE")
    ]
    status = collections.Counter(r["status"] for r in got.values())
    summary = {
        "rows": len(ids),
        "written": len(got),
        "missing": len(missing),
        "status": dict(status),
        "tokens": meta["tokens"],
        "device_seconds": round(sum(d["device_seconds"] for d in done), 1),
        "gpu_hours": round(
            sum(d["wall_seconds"] + d["load_seconds"] for d in done) / 3600, 3
        ),
        "plan": str(directory),
    }
    prov = Path(args.out) / "provenance.json"
    if prov.exists():
        summary["provenance"] = json.loads(prov.read_text())
    batching.atomic_json(Path(args.out) / "summary.json", summary)
    return summary


def cmd_run(args) -> None:
    directory = build_plan(args)
    meta = json.loads((directory / "meta.json").read_text())
    print(
        json.dumps(
            {
                "event": "plan_ready",
                "dir": str(directory),
                **{k: meta[k] for k in ("rows", "tokens", "max_tokens", "rows_over")},
            }
        ),
        flush=True,
    )
    if args.plan_only:
        return
    devices = [f"cuda:{d}" for d in batching.parse_ids(args.devices, 1)]
    num_shards = args.num_shards or len(devices)
    shards = batching.parse_ids(args.shards, num_shards)
    common = [
        "--plan",
        str(directory),
        "--out",
        args.out,
        "--num-shards",
        str(num_shards),
        "--ckpt",
        args.ckpt,
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
    ):
        value = getattr(args, flag)
        if value is not None:
            common += ["--" + flag.replace("_", "-"), str(value)]
    if args.write_logits:
        common.append("--write-logits")
    pending = [
        s
        for s in shards
        if not (Path(args.out) / "shards" / f"shard-{s:03d}.DONE").exists()
    ]
    codes = (
        batching.spawn_workers(
            "d25.vega.eval.run_rows",
            pending,
            devices[: len(pending)],
            common,
            Path(args.out) / "logs",
        )
        if pending
        else []
    )
    if all(
        (Path(args.out) / "shards" / f"shard-{s:03d}.DONE").exists()
        for s in range(num_shards)
    ):
        print(json.dumps(merge(args, directory)), flush=True)
    else:
        print(json.dumps({"event": "shards_incomplete", "codes": codes}), flush=True)
        raise SystemExit(1 if any(codes) else 0)


def add_model_args(p) -> None:
    p.add_argument("--ckpt", required=True)
    p.add_argument("--revision")
    p.add_argument("--prompt", choices=("d25-vega", "pplx"))
    p.add_argument("--attention-mode", choices=("causal", "noncausal_full_attention"))
    p.add_argument("--max-length", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--readout-dtype", choices=("float32", "bfloat16"))
    p.add_argument("--max-batch-tokens", type=int)
    p.add_argument("--max-batch-size", type=int)
    p.add_argument("--write-logits", action="store_true")


def main(argv=None) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    add_model_args(r)
    r.add_argument("--rows", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--devices", default="0")
    r.add_argument("--num-shards", type=int)
    r.add_argument("--shards")
    r.add_argument("--cpu-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r.add_argument("--plan-only", action="store_true")
    r.set_defaults(func=cmd_run)
    w = sub.add_parser("worker")
    add_model_args(w)
    w.add_argument("--plan", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--shard", type=int, required=True)
    w.add_argument("--num-shards", type=int, required=True)
    w.add_argument("--device", required=True)
    w.set_defaults(func=cmd_worker)
    m = sub.add_parser("merge")
    m.add_argument("--plan", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=lambda a: print(json.dumps(merge(a, Path(a.plan)))))
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
