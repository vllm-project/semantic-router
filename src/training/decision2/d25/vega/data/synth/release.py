"""SYN1 release: training rows with soft targets (SYN1T) and the wave-2 mixture M3T = M2T-v5 + SYN1T.

    python -m d25.vega.data.synth.release syn1t --assembled DIR --audit DIR --out DIR --tokenizer QWEN3.8_DIR
    python -m d25.vega.data.synth.release m3t --base /data/d25/shared/data/v1/M2T-v5 --syn1t DIR --out DIR --name M3T

syn1t: every assembled (accepted, deduplicated, decontaminated) SYN1 row gets
       target = 0.5 * gold + 0.5 * verifier code probabilities (``meta.teachers.<verifier>``, the generator-family
       model reading the row with the training prompt), gold kept in ``meta.gold_target``; prompt token counts with
       the d25-vega prompt (rows over 8,160 tokens dropped); a group-disjoint dev split of ~500 rows (whole seeds).
m3t:   train = base train + SYN1T train, dev = base dev + SYN1T dev, deterministic shuffle, 50k-row shards.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from d25.vega.common import decision_format as df
from d25.vega.data import tokens as tk
from d25.vega.data.mixture import MAX_PROMPT_TOKENS, composition, stats
from d25.vega.data.util import rank, read_jsonl, sha256_file, write_json, write_jsonl

SHARD_ROWS = 50_000


def write_shards(
    out: Path, train: list[dict[str, Any]], dev: list[dict[str, Any]]
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("train-*.jsonl.gz"):
        old.unlink()
    shards = max(1, math.ceil(len(train) / SHARD_ROWS))
    files = {}
    for i in range(shards):
        name = f"train-{i:05d}-of-{shards:05d}.jsonl.gz"
        write_jsonl(out / name, train[i * SHARD_ROWS : (i + 1) * SHARD_ROWS])
        files[name] = {
            "rows": min(SHARD_ROWS, len(train) - i * SHARD_ROWS),
            "sha256": sha256_file(out / name),
            "bytes": (out / name).stat().st_size,
        }
    write_jsonl(out / "dev.jsonl.gz", dev)
    files["dev.jsonl.gz"] = {
        "rows": len(dev),
        "sha256": sha256_file(out / "dev.jsonl.gz"),
        "bytes": (out / "dev.jsonl.gz").stat().st_size,
    }
    return files


def soft_target(
    row: dict[str, Any], gold_weight: float
) -> tuple[list[float], str | None]:
    gold = list(row["meta"].get("gold_target") or row["target"])
    teachers = row["meta"].get("teachers") or {}
    name, probs = next(iter(teachers.items()), (None, None))
    if not probs or len(probs) != len(gold) or sum(probs) <= 0:
        return gold, None
    total = sum(probs)
    mixed = [
        gold_weight * g + (1 - gold_weight) * p / total for g, p in zip(gold, probs)
    ]
    s = sum(mixed)
    target = [round(v / s, 6) for v in mixed]
    best = max(range(len(target)), key=target.__getitem__)
    target[best] = round(target[best] + 1.0 - sum(target), 6)
    return target, name


def syn1t(
    assembled: Path,
    audit: Path | None,
    out: Path,
    tokenizer: str,
    gold_weight: float,
    dev_rows: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    started = time.time()
    rows = [
        r
        for path in sorted(assembled.glob("train-*.jsonl.gz"))
        for r in read_jsonl(path)
    ]
    counts: Counter = Counter()
    for row in rows:
        row["meta"]["gold_target"] = list(
            row["meta"].get("gold_target") or row["target"]
        )
        target, teacher = soft_target(row, gold_weight)
        row["target"] = target
        row["meta"]["part"] = "syn1"
        row["meta"][
            "dataset"
        ] = "SYN1 (synthetic; generator and verifier Qwen/Qwen3.5-397B-A17B-FP8@ea5b4f81, Apache-2.0)"
        if teacher:
            row["meta"]["teacher"] = teacher
            row["meta"][
                "target_rule"
            ] = f"{gold_weight}*gold+{1 - gold_weight}*{teacher}"
            counts["soft"] += 1
        else:
            counts["gold_only"] += 1
        df.validate_row(row)
    chunks = [rows[i : i + 512] for i in range(0, len(rows), 512)]
    with Pool(workers, initializer=tk._init, initargs=(tokenizer,)) as pool:
        lengths = {
            item["id"]: item["n"]
            for part in pool.map(tk.count, chunks, chunksize=1)
            for item in part
        }
    kept = []
    for row in rows:
        n = lengths.get(row["id"], -1)
        if n < 0 or n > MAX_PROMPT_TOKENS:
            counts["overlong_or_unrenderable"] += 1
            continue
        row["meta"]["n_tokens"] = n
        kept.append(row)
    seeds = sorted(
        {r["meta"]["seed_id"] for r in kept}, key=lambda s: rank(s, f"{seed}:dev")
    )
    by_seed: dict[str, list[dict[str, Any]]] = {}
    for row in kept:
        by_seed.setdefault(row["meta"]["seed_id"], []).append(row)
    dev, dev_seeds = [], set()
    for s in seeds:
        if len(dev) >= dev_rows:
            break
        dev.extend(by_seed[s])
        dev_seeds.add(s)
    train = [r for r in kept if r["meta"]["seed_id"] not in dev_seeds]
    train.sort(key=lambda r: rank(r["id"], seed))
    dev.sort(key=lambda r: rank(r["id"], seed))
    files = write_shards(out, train, dev)
    assembled_manifest = json.loads((assembled / "manifest.json").read_text())
    audit_report = (
        json.loads((audit / "report.json").read_text())
        if audit and (audit / "report.json").exists()
        else None
    )
    manifest = {
        "corpus": "SYN1T",
        "format": {
            "row_contract": "d25.vega.common.decision_format",
            "format_id": df.FORMAT_ID,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
        },
        "target_rule": f"target = {gold_weight} * gold + {1 - gold_weight} * verifier code probabilities "
        "(meta.teachers); gold in meta.gold_target",
        "rows": {"train": len(train), "dev": len(dev)},
        "counts": dict(counts),
        "files": files,
        "seed": seed,
        "composition": {"train": composition(train), "dev": composition(dev)},
        "by_archetype": dict(
            Counter(r["meta"]["archetype"] for r in train).most_common()
        ),
        "by_condition": dict(
            Counter(r["meta"]["condition"] for r in train).most_common()
        ),
        "tokens": {
            "train": stats([r["meta"]["n_tokens"] for r in train]),
            "dev": stats([r["meta"]["n_tokens"] for r in dev]),
        },
        "assembled": {
            k: assembled_manifest.get(k)
            for k in (
                "pipeline",
                "generator",
                "verifier",
                "prompts",
                "seeds",
                "domains",
                "teacher_argmax_agrees_with_gold",
                "files",
            )
        },
        "audit": audit_report,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.time() - started, 1),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def m3t(base: Path, syn: Path, out: Path, name: str, seed: int) -> dict[str, Any]:
    started = time.time()
    base_manifest = json.loads((base / "manifest.json").read_text())
    syn_manifest = json.loads((syn / "manifest.json").read_text())
    train, dev = [], []
    for directory, manifest in ((base, base_manifest), (syn, syn_manifest)):
        for file in manifest["files"]:
            target = dev if file == "dev.jsonl.gz" else train
            target.extend(read_jsonl(directory / file))
    ids = Counter(r["id"] for r in train + dev)
    duplicates = sum(c - 1 for c in ids.values() if c > 1)
    if duplicates:
        raise SystemExit(f"{duplicates} duplicate ids between {base} and {syn}")
    train.sort(key=lambda r: rank(r["id"], seed))
    dev.sort(key=lambda r: rank(r["id"], seed))
    files = write_shards(out, train, dev)
    manifest = {
        "mix": name,
        "name": name,
        "corpus": "v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "description": f"{name} = {base_manifest.get('name') or base_manifest.get('mix')} (all rows, targets unchanged) + "
        f"SYN1T (synthetic decisions, 0.5 gold + 0.5 verifier probabilities), reshuffled.",
        "rows": {"train": len(train), "dev": len(dev)},
        "files": files,
        "parts": {"base": base_manifest["rows"], "syn1t": syn_manifest["rows"]},
        "base": {
            "path": str(base),
            "name": base_manifest.get("name"),
            "files": {k: v["sha256"] for k, v in base_manifest["files"].items()},
        },
        "syn1t": {
            "path": str(syn),
            "files": {k: v["sha256"] for k, v in syn_manifest["files"].items()},
            "counts": syn_manifest.get("counts"),
            "audit_overall": (syn_manifest.get("audit") or {}).get("overall"),
        },
        "target_rule": {
            "base": base_manifest.get("target_rule"),
            "syn1t": syn_manifest.get("target_rule"),
        },
        "composition": {"train": composition(train), "dev": composition(dev)},
        "tokens": {
            "train": stats([r["meta"]["n_tokens"] for r in train]),
            "dev": stats([r["meta"]["n_tokens"] for r in dev]),
        },
        "decontamination": base_manifest.get("decontamination"),
        "holdouts": base_manifest.get("holdouts"),
        "seconds": round(time.time() - started, 1),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("syn1t")
    a.add_argument("--assembled", type=Path, required=True)
    a.add_argument("--audit", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--tokenizer", default="/models/Qwen3.8-27B")
    a.add_argument("--gold-weight", type=float, default=0.5)
    a.add_argument("--dev-rows", type=int, default=500)
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--seed", type=int, default=20261012)
    b = sub.add_parser("m3t")
    b.add_argument("--base", type=Path, required=True)
    b.add_argument("--syn1t", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    if args.cmd == "syn1t":
        m = syn1t(
            args.assembled,
            args.audit,
            args.out,
            args.tokenizer,
            args.gold_weight,
            args.dev_rows,
            args.workers,
            args.seed,
        )
        print(
            json.dumps(
                {k: m[k] for k in ("rows", "counts", "tokens", "by_archetype")},
                indent=1,
            )[:3000]
        )
    else:
        m = m3t(args.base, args.syn1t, args.out, args.name, args.seed)
        print(
            json.dumps({k: m[k] for k in ("rows", "parts", "tokens")}, indent=1)[:2000]
        )


if __name__ == "__main__":
    main()
