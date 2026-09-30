"""Decoder M8-small teacher files and data lock (prereg dec-m8s-prereg-2026-09-30.md), host Python on node B,
stdlib only. It never writes READY files.

  teachers  split the A20r label shards (m8s_label.py label) into each tier's D1 file (every top-up row) and D2 file
            (the human rows only; typed rows then train on gold, --teacher-partial), keyed by (id, input_sha256), in
            TRAIN order: <root>/teacher/<tier>-D1/teacher.jsonl, <root>/teacher/<tier>-D2/teacher.jsonl
  check     per tier: TRAIN hash = compose.json, unique ids, quarantine absent, parts file aligned with TRAIN, the
            exposure receipt (r2 payload 2194716a) lists 0 groups, C1 registry source keys give 0 hits, teacher
            coverage (D1 every row; D2 exactly the human rows; 2B C every row), option keys match, A20r argmax
            agreement with gold by part and type -> lock-<tier>.json, PASS / FAIL

usage: python3 m8s_lock.py teachers --root R --labels SHARD... [--tiers 2b 08b]
       python3 m8s_lock.py check --root R --tier 2b|08b --out OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "v2" / "9b"))
from lux9b.m3_data import c1_keys, denied_hits  # noqa: E402

C1_REGISTRY = CODE / "v2/eval/records/sealed-c1-source-registry-2026-09-28.json"
QUARANTINE = CODE / "v2/dec/ops/m7/specs/m7-quarantine-groups.json"
CONTROL_TEACHER = {"2b": "teacher/2b-C/teacher.jsonl", "08b": None}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    r, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
    return sha(path)


def label_map(paths: list[Path]) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for path in paths:
        for r in read_jsonl(path):
            key = (r["id"], r["input_sha256"])
            if key in out:
                raise ValueError(f"{path}: repeated label {key}")
            out[key] = r["teacher_probs"]
    return out


def split_teachers(
    train: list[dict[str, Any]],
    parts: dict[str, str],
    labels: dict[tuple[str, str], dict[str, float]],
) -> dict[str, list[dict[str, Any]]]:
    d1, d2 = [], []
    for row in train:
        probs = labels.get((row["id"], row["input_sha256"]))
        if probs is None:
            raise ValueError(f"{row['id']}: no A20r label")
        if set(probs) != {o["key"] for o in row["options"]}:
            raise ValueError(f"{row['id']}: label keys differ from the options")
        rec = {
            "id": row["id"],
            "input_sha256": row["input_sha256"],
            "teacher_probs": probs,
        }
        d1.append(rec)
        if parts[row["id"]] == "human":
            d2.append(rec)
    return {"D1": d1, "D2": d2}


def teachers(a: argparse.Namespace) -> int:
    labels = label_map(a.labels)
    for tier in a.tiers:
        data = a.root / "data" / tier
        train = read_jsonl(data / "train.jsonl")
        parts = {r["id"]: r["part"] for r in read_jsonl(data / "train.parts.jsonl")}
        for arm, rows in split_teachers(train, parts, labels).items():
            digest = write_jsonl(
                a.root / "teacher" / f"{tier}-{arm}" / "teacher.jsonl", rows
            )
            print(
                json.dumps(
                    {"tier": tier, "arm": arm, "rows": len(rows), "sha256": digest}
                )
            )
    return 0


def argmax(probs: dict[str, float], keys: list[str]) -> int:
    values = [probs[k] for k in keys]
    return max(range(len(values)), key=values.__getitem__)


def check(a: argparse.Namespace) -> int:
    root, tier = a.root, a.tier
    data = root / "data" / tier
    comp = json.loads((data / "compose.json").read_text())
    fails: list[str] = []
    train_path = data / "train.jsonl"
    train = read_jsonl(train_path)
    parts_rows = read_jsonl(data / "train.parts.jsonl")
    digest = sha(train_path)
    if digest != comp["files"]["train.jsonl"]:
        fails.append("TRAIN hash differs from compose.json")
    ids = [r["id"] for r in train]
    if len(set(ids)) != len(ids):
        fails.append("repeated TRAIN ids")
    quarantine = set(json.loads(QUARANTINE.read_text())["group_ids"])
    if any(r["group_id"] in quarantine for r in train):
        fails.append("quarantined group present")
    if [(p["id"], p["input_sha256"]) for p in parts_rows] != [
        (r["id"], r["input_sha256"]) for r in train
    ]:
        fails.append("parts file not aligned with TRAIN")
    parts = {p["id"]: p["part"] for p in parts_rows}
    exposure_path = root / "exposure" / f"{tier}" / f"exposure-{tier}.json"
    exposure = (
        json.loads(exposure_path.read_text()) if exposure_path.is_file() else None
    )
    if exposure is None:
        fails.append("exposure receipt missing")
    elif exposure.get("groups") or [
        f.get("sha256") for f in exposure.get("files", [])
    ] != [digest]:
        fails.append("exposure receipt lists groups or names another TRAIN file")
    keys = c1_keys(json.loads(C1_REGISTRY.read_text()))
    sources = sorted({r["source"] for r in train})
    families = sorted({r.get("family", "") for r in train} - {""})
    c1 = {s: h for s in sources if (h := denied_hits(s, keys, []))}
    c1_family = {f: h for f in families if (h := denied_hits(f, keys, []))}
    if c1:
        fails.append(f"C1 registry source hits {sorted(c1)}")
    teacher_report: dict[str, Any] = {}
    expected = {
        "D1": {r["id"] for r in train},
        "D2": {r["id"] for r in train if parts[r["id"]] == "human"},
    }
    control = CONTROL_TEACHER[tier]
    if control:
        expected["C"] = {r["id"] for r in train}
    by_id = {r["id"]: r for r in train}
    for arm, want in expected.items():
        path = root / (control if arm == "C" else f"teacher/{tier}-{arm}/teacher.jsonl")
        if not path.is_file():
            fails.append(f"{arm} teacher missing ({path})")
            continue
        recs = read_jsonl(path)
        got = {r["id"] for r in recs}
        bad = [
            r["id"]
            for r in recs
            if r["id"] not in by_id
            or r["input_sha256"] != by_id[r["id"]]["input_sha256"]
            or set(r["teacher_probs"]) != {o["key"] for o in by_id[r["id"]]["options"]}
        ]
        if got != want or len(recs) != len(want) or bad:
            fails.append(
                f"{arm} teacher coverage: {len(got & want)}/{len(want)} rows, {len(bad)} bad"
            )
        agree: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in recs:
            row = by_id.get(r["id"])
            if row is None:
                continue
            keys_ = [o["key"] for o in row["options"]]
            cell = agree[f"{parts[row['id']]}:{row['task_type']}"]
            cell[0] += int(argmax(r["teacher_probs"], keys_) == row["label"])
            cell[1] += 1
        teacher_report[arm] = {
            "path": str(path),
            "sha256": sha(path),
            "rows": len(recs),
            "argmax_gold_agreement": {
                k: {"correct": c, "n": n, "accuracy": c / n}
                for k, (c, n) in sorted(agree.items())
            },
        }
    doc = {
        "schema": "dec-m8s-lock/1",
        "tier": tier,
        "status": "FAIL" if fails else "PASS",
        "fails": fails,
        "train": {"path": str(train_path), "sha256": digest, "rows": len(train)},
        "parts": {
            "path": str(data / "train.parts.jsonl"),
            "sha256": sha(data / "train.parts.jsonl"),
            "rows_by_part": dict(sorted(Counter(parts.values()).items())),
        },
        "compose": {
            "sha256": sha(data / "compose.json"),
            "tokens_by_part": comp["report"]["tokens_by_part"],
            "tokens": comp["report"]["tokens"],
            "seed": comp["seed"],
            "recipe": comp["recipe"],
        },
        "exposure": {
            "path": str(exposure_path),
            "sha256": sha(exposure_path) if exposure else None,
            "groups": (exposure or {}).get("groups"),
        },
        "c1_registry_sha256": sha(C1_REGISTRY),
        "c1_keys": len(keys),
        "c1_registry_source_hits": c1,
        "c1_registry_family_name_hits": c1_family,
        "teachers": teacher_report,
    }
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"{doc['status']} {tier} {'; '.join(fails)}".rstrip())
    return 0 if not fails else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teachers")
    t.add_argument("--root", type=Path, required=True)
    t.add_argument("--labels", type=Path, nargs="+", required=True)
    t.add_argument("--tiers", nargs="+", default=["2b", "08b"])
    c = sub.add_parser("check")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--tier", choices=("2b", "08b"), required=True)
    c.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    return teachers(a) if a.cmd == "teachers" else check(a)


if __name__ == "__main__":
    sys.exit(main())
