"""Decoder M17 TRAIN files (prereg dec-m17-prereg-2026-10-02.md, "Data"): IB swapped in at LH's token count.

Inputs: LH's released TRAIN (--base); an M12 arm whose base block is that file byte for byte, for the per-row tokens
as the trainer counts them (--base-train / --base-ids: M12 `4b-LHA10`); the IB pool's arm (--pool-train / --pool-ids:
M12 `4b-LHA`, its IB block); M13's self-distillation targets for LH's TRAIN (--teacher).

Per arm NAME=SHARE (T = LH's TRAIN tokens):
  IB       whole groups of the pool (the pool arm's IB rows without the --drop families), stratified by family in
           proportion to the family's pool tokens (M11's sample_groups), target floor(SHARE * T); R = its tokens.
  cut      candidates are LH rows of groups whose every row is `en`; removed: (family, group) units stratified by
           family (sample_groups, target R), then one fill pass over the remaining units in a seeded order
           (random.Random("<seed>:fill:<NAME>")) while the removed total stays <= R. The build fails unless
           0 <= TRAIN tokens - T <= TOL * T.
Output <output>/<NAME>/: train.jsonl (the kept LH lines, then the IB lines, byte for byte), train.ids.jsonl (id,
block, tokens), teacher-s.jsonl (the target lines of the kept LH ids, the teacher file's bytes and order) and
report.json. Host python3, standard library only.

usage: m17_data.py --base F --base-sha S --base-train F --base-train-sha S --base-ids F --pool-train F --pool-sha S \
         --pool-ids F --teacher F --teacher-sha S [--drop sentfin] --arm NAME=SHARE [...] --seed 20261002 --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "dec-m17-data/1"
TOL = 0.0001
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


def read_rows(train: Path, ids: Path) -> list[dict[str, Any]]:
    """Rows of an M12 arm with their ids-file block and tokens, in file order."""
    lines = train.read_bytes().splitlines(keepends=True)
    meta = [json.loads(x) for x in ids.read_text().splitlines()]
    if len(meta) != len(lines):
        raise ValueError(f"{train} and {ids} differ in length")
    rows = []
    for line, m in zip(lines, meta, strict=True):
        r = json.loads(line)
        if r["id"] != m["id"]:
            raise ValueError(f"{ids} out of order at {r['id']}")
        rows.append(
            {
                "id": r["id"],
                "group_id": r["group_id"],
                "family": r["family"],
                "language": r["language"],
                "type": r.get("task_type"),
                "block": m["block"],
                "tokens": int(m["tokens"]),
                "line": line if line.endswith(b"\n") else line + b"\n",
            }
        )
    return rows


def units_of(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    """(family, group) units, the granularity of sample_groups."""
    units: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        units[(r["family"], r["group_id"])].append(i)
    return units


def english_cut(
    cand: list[dict[str, Any]], target: int, seed: int, name: str
) -> tuple[set[int], dict[str, Any]]:
    """Indices of candidate rows removed: stratified (family, group) units, then the fill pass."""
    removed, _ = sample_groups()(cand, target, seed)
    stratified = sum(cand[i]["tokens"] for i in removed)
    units = units_of(cand)
    left = sorted(u for u, idx in units.items() if not removed.intersection(idx))
    rng = random.Random(f"{seed}:fill:{name}")
    rng.shuffle(left)
    used = stratified
    filled = 0
    for u in left:
        size = sum(cand[i]["tokens"] for i in units[u])
        if used + size > target:
            continue
        removed.update(units[u])
        used += size
        filled += 1
    return removed, {
        "target": target,
        "stratified_tokens": stratified,
        "fill_units": filled,
        "removed_tokens": used,
        "shortfall": target - used,
    }


def by(
    rows: list[dict[str, Any]], key: str, idx: set[int] | list[int]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "tokens": 0})
    for i in idx:
        out[str(rows[i][key])]["rows"] += 1
        out[str(rows[i][key])]["tokens"] += rows[i]["tokens"]
    return dict(sorted(out.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path, want in (
        (args.base, args.base_sha),
        (args.base_train, args.base_train_sha),
        (args.pool_train, args.pool_sha),
        (args.teacher, args.teacher_sha),
    ):
        if sha256(path) != want:
            raise ValueError(f"{path} is not {want}")
    base_lines = args.base.read_bytes().splitlines(keepends=True)
    counted = read_rows(args.base_train, args.base_ids)
    base = [r for r in counted if r["block"] == "base"]
    if [r["line"] for r in base] != [
        x if x.endswith(b"\n") else x + b"\n" for x in base_lines
    ]:
        raise ValueError("the counted arm's base block is not --base byte for byte")
    pool_arm = read_rows(args.pool_train, args.pool_ids)
    if [r["line"] for r in pool_arm if r["block"] == "base"] != [
        r["line"] for r in base
    ]:
        raise ValueError("the pool arm's base block is not --base byte for byte")
    drop = set(args.drop or [])
    ib_all = [r for r in pool_arm if r["block"] != "base"]
    missing = drop - {r["family"] for r in ib_all}
    if missing:
        raise ValueError(f"--drop families not in the pool: {sorted(missing)}")
    pool = [r for r in ib_all if r["family"] not in drop]
    ids = [r["id"] for r in base] + [r["id"] for r in pool]
    if len(ids) != len(set(ids)):
        raise ValueError("repeated row id across LH TRAIN and the IB pool")
    T = sum(r["tokens"] for r in base)
    en_tokens = sum(r["tokens"] for r in base if r["language"] == "en")
    langs: dict[str, set[str]] = defaultdict(set)
    for r in base:
        langs[r["group_id"]].add(r["language"])
    cand_idx = [i for i, r in enumerate(base) if langs[r["group_id"]] == {"en"}]
    cand = [base[i] for i in cand_idx]
    teacher: dict[str, bytes] = {}
    order: list[str] = []
    with args.teacher.open("rb") as stream:
        for line in stream:
            rid = json.loads(line)["id"]
            if rid in teacher:
                raise ValueError(f"repeated teacher id {rid}")
            teacher[rid] = line if line.endswith(b"\n") else line + b"\n"
            order.append(rid)
    if set(teacher) != {r["id"] for r in base}:
        raise ValueError("the teacher file does not cover LH's TRAIN ids exactly")
    sampler = sample_groups()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "seed": args.seed,
        "inputs": {
            "base": {"path": str(args.base), "sha256": args.base_sha},
            "base_train": {"path": str(args.base_train), "sha256": args.base_train_sha},
            "base_ids": {"path": str(args.base_ids), "sha256": sha256(args.base_ids)},
            "pool_train": {"path": str(args.pool_train), "sha256": args.pool_sha},
            "pool_ids": {"path": str(args.pool_ids), "sha256": sha256(args.pool_ids)},
            "teacher": {"path": str(args.teacher), "sha256": args.teacher_sha},
        },
        "token_unit": "the M12 ids files' tokens (training.model.decision_model.encode, head readout)",
        "T": T,
        "base": {
            "rows": len(base),
            "en_rows": sum(1 for r in base if r["language"] == "en"),
            "en_tokens": en_tokens,
            "ml_tokens": T - en_tokens,
            "ml_share": (T - en_tokens) / T,
            "groups": len(langs),
            "en_only_groups": sum(1 for v in langs.values() if v == {"en"}),
            "mixed_language_groups": sum(
                1 for v in langs.values() if "en" in v and len(v) > 1
            ),
        },
        "candidates": {
            "rows": len(cand),
            "tokens": sum(r["tokens"] for r in cand),
            "groups": len({r["group_id"] for r in cand}),
            "units": len(units_of(cand)),
        },
        "pool": {
            "dropped_families": sorted(drop),
            "dropped": by(
                ib_all,
                "family",
                [i for i, r in enumerate(ib_all) if r["family"] in drop],
            ),
            "rows": len(pool),
            "tokens": sum(r["tokens"] for r in pool),
            "families": len({r["family"] for r in pool}),
        },
        "arms": {},
    }
    for spec in args.arm:
        name, share_text = spec.split("=", 1)
        share = float(share_text)
        if not 0 < share < 1:
            raise ValueError(f"{spec}: share must be in (0, 1)")
        target = math.floor(share * T)
        kept, _ = sampler(pool, target, args.seed)
        ib = [pool[i] for i in sorted(kept)]
        R = sum(r["tokens"] for r in ib)
        removed_local, cut = english_cut(cand, R, args.seed, name)
        removed = {cand_idx[i] for i in removed_local}
        kept_base = [r for i, r in enumerate(base) if i not in removed]
        rows = kept_base + ib
        tokens = sum(r["tokens"] for r in rows)
        if not 0 <= tokens - T <= TOL * T:
            raise ValueError(
                f"{name}: TRAIN tokens {tokens} vs T {T} outside [0, {TOL} T]"
            )
        if any(base[i]["language"] != "en" for i in removed):
            raise ValueError(f"{name}: a non-English row was removed")
        out = args.output / name
        out.mkdir(parents=True)
        with (out / "train.jsonl").open("xb") as t, (out / "train.ids.jsonl").open(
            "x"
        ) as m:
            for r in rows:
                t.write(r["line"])
                m.write(
                    json.dumps(
                        {"id": r["id"], "block": r["block"], "tokens": r["tokens"]}
                    )
                    + "\n"
                )
        kept_ids = {r["id"] for r in kept_base}
        with (out / "teacher-s.jsonl").open("xb") as stream:
            for rid in order:
                if rid in kept_ids:
                    stream.write(teacher[rid])
        ml = sum(r["tokens"] for r in rows if r["language"] != "en")
        removed_tokens = sum(base[i]["tokens"] for i in removed)
        report["arms"][name] = {
            "share": share,
            "ib": {
                "target": target,
                "rows": len(ib),
                "tokens": R,
                "groups": len({r["group_id"] for r in ib}),
                "share_of_train": R / tokens,
                "families": by(ib, "family", range(len(ib))),
                "blocks": dict(sorted(Counter(r["block"] for r in ib).items())),
                "ml_tokens": sum(r["tokens"] for r in ib if r["language"] != "en"),
            },
            "removed": {
                **cut,
                "rows": len(removed),
                "groups": len({base[i]["group_id"] for i in removed}),
                "share_of_en_tokens": removed_tokens / en_tokens,
                "families": by(base, "family", sorted(removed)),
                "types": by(base, "type", sorted(removed)),
            },
            "train": {
                "rows": len(rows),
                "tokens": tokens,
                "minus_T": tokens - T,
                "updates_64": math.ceil(len(rows) / 64),
                "ml_tokens": ml,
                "ml_share": ml / tokens,
                "sha256": sha256(out / "train.jsonl"),
                "ids_sha256": sha256(out / "train.ids.jsonl"),
            },
            "teacher": {
                "rows": len(kept_ids),
                "sha256": sha256(out / "teacher-s.jsonl"),
            },
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "base-train", "pool-train", "teacher"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--base-sha", required=True)
    p.add_argument("--base-train-sha", required=True)
    p.add_argument("--pool-sha", required=True)
    p.add_argument("--teacher-sha", required=True)
    p.add_argument("--base-ids", type=Path, required=True)
    p.add_argument("--pool-ids", type=Path, required=True)
    p.add_argument("--drop", action="append")
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = build(args)
    print(
        json.dumps(
            {
                k: {
                    "rows": v["train"]["rows"],
                    "tokens": v["train"]["tokens"],
                    "minus_T": v["train"]["minus_T"],
                    "ib_share": round(v["ib"]["share_of_train"], 4),
                    "en_removed": round(v["removed"]["share_of_en_tokens"], 4),
                    "ml_share": round(v["train"]["ml_share"], 4),
                    "train_sha256": v["train"]["sha256"],
                    "teacher_sha256": v["teacher"]["sha256"],
                }
                for k, v in report["arms"].items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
