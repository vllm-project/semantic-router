"""Decoder M15 multilingual-preserving TRAIN (prereg dec-m15-prereg-2026-10-01.md, "Data").

Input: an M12 arm (train.jsonl + train.ids.jsonl, blocks base / ib1 / ib2, IB copies `~c<k>`) and the tier's M13
self-distillation targets. Output (in file order): the arm's base rows whole; its IB lines whole, or with --ib-share a
subset whose tokens are floor(share * T) (copy levels in order, each kept whole while it fits, the last sampled by whole
groups stratified by family with M11's sample_groups); then one copy (id suffix `~m2`) of chosen released multilingual
groups (every row non-`en`), stratified by language, adding U = (s(T+I) - ML - I_ml) / (1 - s) tokens so the TRAIN
multilingual token share equals the released share s. The build fails if the built share is off s by more than TOL.
The teacher file is the M13 targets plus one line per typed copy (its original's targets under the copy id).

usage: m15_data.py --arm-train F --arm-sha S --arm-ids F --teacher F --teacher-sha S --name NAME [--ib-share X]
         --seed N --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "dec-m15-data/1"
ML_SUFFIX = "~m2"
TOL = 0.002
OPS = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def sample_groups() -> Any:
    spec = importlib.util.spec_from_file_location(
        "m11_s2data", OPS / "m11" / "m11_s2data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_groups


def copy_level(row_id: str) -> int:
    return int(row_id.rsplit("~c", 1)[1]) if "~c" in row_id else 1


def ib_subset(rows: list[dict[str, Any]], target: int, seed: int) -> set[int]:
    """Indices of IB rows kept: copy levels in order, whole while they fit, the last level sampled."""
    levels: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        levels[copy_level(r["id"])].append(i)
    kept: set[int] = set()
    left = target
    sampler = sample_groups()
    for level in sorted(levels):
        idx = levels[level]
        size = sum(rows[i]["tokens"] for i in idx)
        if size <= left:
            kept.update(idx)
            left -= size
            continue
        sub = [rows[i] for i in idx]
        chosen, _ = sampler(sub, left, seed + level)
        kept.update(idx[j] for j in chosen)
        break
    return kept


def ml_groups(
    base: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, list[int]]], Counter]:
    """Multilingual groups by language (groups whose rows are all non-en) and multilingual tokens per language."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(base):
        by_group[r["group_id"]].append(i)
    lang_tokens: Counter = Counter()
    groups: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for g, idx in by_group.items():
        langs = Counter()
        for i in idx:
            if base[i]["language"] != "en":
                lang_tokens[base[i]["language"]] += base[i]["tokens"]
            langs[base[i]["language"]] += base[i]["tokens"]
        if "en" in langs:
            continue
        groups[langs.most_common(1)[0][0]][g] = idx
    return groups, lang_tokens


def upsample(base: list[dict[str, Any]], target: int, seed: int) -> list[int]:
    """Base indices to copy once: whole multilingual groups, per language quota target * ML_L / ML."""
    groups, lang_tokens = ml_groups(base)
    ml = sum(lang_tokens.values())
    chosen: list[int] = []
    for lang in sorted(groups):
        quota = target * lang_tokens[lang] / ml
        rng = random.Random(f"{seed}:{lang}")
        order = sorted(groups[lang])
        rng.shuffle(order)
        used = 0
        for g in order:
            size = sum(base[i]["tokens"] for i in groups[lang][g])
            if used + size > quota:
                continue
            chosen.extend(groups[lang][g])
            used += size
    return sorted(chosen)


def copy_line(line: bytes) -> bytes:
    row = json.loads(line)
    row["id"] = f"{row['id']}{ML_SUFFIX}"
    return json.dumps(row, ensure_ascii=False).encode() + b"\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    if sha256(args.arm_train) != args.arm_sha:
        raise ValueError(f"{args.arm_train} is not {args.arm_sha}")
    if sha256(args.teacher) != args.teacher_sha:
        raise ValueError(f"{args.teacher} is not {args.teacher_sha}")
    lines = args.arm_train.read_bytes().splitlines(keepends=True)
    ids = [json.loads(x) for x in args.arm_ids.read_text().splitlines()]
    if len(ids) != len(lines):
        raise ValueError("arm TRAIN and ids differ in length")
    rows = []
    for line, meta in zip(lines, ids):
        r = json.loads(line)
        if r["id"] != meta["id"]:
            raise ValueError(f"ids file out of order at {r['id']}")
        rows.append(
            {
                "id": r["id"],
                "group_id": r["group_id"],
                "family": r["family"],
                "language": r["language"],
                "block": meta["block"],
                "tokens": meta["tokens"],
            }
        )
    base_idx = [i for i, r in enumerate(rows) if r["block"] == "base"]
    ib_idx = [i for i, r in enumerate(rows) if r["block"] != "base"]
    base = [rows[i] for i in base_idx]
    T = sum(r["tokens"] for r in base)
    ML = sum(r["tokens"] for r in base if r["language"] != "en")
    s = ML / T
    if args.ib_share is None:
        keep_ib = list(ib_idx)
    else:
        ib_rows = [rows[i] for i in ib_idx]
        kept = ib_subset(ib_rows, int(args.ib_share * T), args.seed)
        keep_ib = [ib_idx[j] for j in sorted(kept)]
    I = sum(rows[i]["tokens"] for i in keep_ib)
    I_ml = sum(rows[i]["tokens"] for i in keep_ib if rows[i]["language"] != "en")
    U = max(0, round((s * (T + I) - ML - I_ml) / (1 - s)))
    copies = upsample(base, U, args.seed)
    added = sum(base[j]["tokens"] for j in copies)
    share = (ML + I_ml + added) / (T + I + added)
    if abs(share - s) > TOL:
        raise ValueError(f"built multilingual share {share:.4f} is off {s:.4f}")
    teacher = {}
    with args.teacher.open("rb") as stream:
        for line in stream:
            teacher[json.loads(line)["id"]] = line
    out = args.output / args.name
    out.mkdir(parents=True)
    keep = sorted(set(base_idx) | set(keep_ib))
    with (out / "train.jsonl").open("wb") as t, (out / "train.ids.jsonl").open(
        "w"
    ) as m:
        for i in keep:
            t.write(lines[i])
            m.write(json.dumps({k: rows[i][k] for k in ("id", "block", "tokens")}))
            m.write("\n")
        for j in copies:
            i = base_idx[j]
            t.write(copy_line(lines[i]))
            m.write(
                json.dumps(
                    {
                        "id": rows[i]["id"] + ML_SUFFIX,
                        "block": "ml",
                        "tokens": rows[i]["tokens"],
                    }
                )
            )
            m.write("\n")
    taught = 0
    with (out / "teacher-ml.jsonl").open("wb") as stream:
        for line in teacher.values():
            stream.write(line)
        for j in copies:
            rid = base[j]["id"]
            if rid not in teacher:
                continue
            entry = json.loads(teacher[rid])
            entry["id"] = rid + ML_SUFFIX
            stream.write(json.dumps(entry, ensure_ascii=False).encode() + b"\n")
            taught += 1
    copy_langs = Counter()
    for j in copies:
        copy_langs[base[j]["language"]] += base[j]["tokens"]
    report = {
        "schema": SCHEMA,
        "name": args.name,
        "seed": args.seed,
        "arm_train_sha256": args.arm_sha,
        "teacher_sha256": args.teacher_sha,
        "ib_share": args.ib_share,
        "base": {"rows": len(base_idx), "tokens": T, "ml_tokens": ML, "ml_share": s},
        "ib": {
            "rows": len(keep_ib),
            "rows_in_arm": len(ib_idx),
            "tokens": I,
            "ml_tokens": I_ml,
            "dose": I / T,
            "by_level": dict(
                sorted(Counter(copy_level(rows[i]["id"]) for i in keep_ib).items())
            ),
        },
        "upsample": {
            "target_tokens": U,
            "tokens": added,
            "rows": len(copies),
            "groups": len({base[j]["group_id"] for j in copies}),
            "tokens_by_language": dict(copy_langs.most_common()),
            "teacher_copies": taught,
        },
        "train": {
            "rows": len(keep) + len(copies),
            "tokens": T + I + added,
            "ml_share": share,
            "sha256": sha256(out / "train.jsonl"),
        },
        "teacher": {
            "rows": len(teacher) + taught,
            "sha256": sha256(out / "teacher-ml.jsonl"),
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-train", type=Path, required=True)
    p.add_argument("--arm-sha", required=True)
    p.add_argument("--arm-ids", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--teacher-sha", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--ib-share", type=float)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    report = build(args)
    print(
        json.dumps(
            {
                "name": report["name"],
                "train": report["train"],
                "ib": report["ib"]["tokens"],
                "upsample": report["upsample"]["tokens"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
