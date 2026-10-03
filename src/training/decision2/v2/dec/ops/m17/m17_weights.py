"""Decoder M17 stage 2 per-row loss weights (prereg dec-m17-stage2-prereg-2026-10-02.md, "Arms"): typed upweight UP.

An UP arm trains on an M17 arm's locked TRAIN file unchanged; the only change is a weights file for
``train_dec --example-weights``. m17_data.py's ids file names each line's block: ``base`` = a kept line of LH's
released (typed) TRAIN, anything else = an IB line. Every ``base`` row gets --released-weight, every other row
--ib-weight (M14's 4B recipe: 1.5 / 1.0).

Checks: TRAIN hashes to --train-sha; the ids file and TRAIN agree line by line (same count, same id); ids are unique.
The report gives rows and weight totals per block and the released rows' share of the total weight.

usage: m17_weights.py --train F --train-sha S --ids F [--released-weight 1.5] [--ib-weight 1.0] --name ARM --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SCHEMA = "dec-m17-weights/1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def build(
    train: Path, ids: Path, released: float, ib: float
) -> tuple[list[dict], dict]:
    if not (math.isfinite(released) and math.isfinite(ib) and released > 0 and ib > 0):
        raise SystemExit("weights must be positive and finite")
    lines = train.read_bytes().splitlines()
    meta = [json.loads(x) for x in ids.read_text().splitlines()]
    if len(meta) != len(lines):
        raise SystemExit(f"ids file has {len(meta)} lines, TRAIN {len(lines)}")
    out, rows, weight, seen = [], Counter(), Counter(), set()
    for line, m in zip(lines, meta):
        row_id = json.loads(line)["id"]
        if row_id != m["id"]:
            raise SystemExit(f"ids file and TRAIN disagree at {row_id} / {m['id']}")
        if row_id in seen:
            raise SystemExit(f"duplicate id {row_id}")
        seen.add(row_id)
        block = "base" if m["block"] == "base" else "ib"
        w = released if block == "base" else ib
        out.append({"id": row_id, "weight": w})
        rows[block] += 1
        weight[block] += w
    total = sum(weight.values())
    report = {
        "rows": dict(rows),
        "weight": dict(weight),
        "released_row_share": rows["base"] / len(out),
        "released_weight_share": weight["base"] / total,
    }
    return out, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--train-sha", required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--released-weight", type=float, default=1.5)
    p.add_argument("--ib-weight", type=float, default=1.0)
    p.add_argument("--name", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if sha256(a.train) != a.train_sha:
        raise SystemExit(f"{a.train} is not {a.train_sha}")
    weights, report = build(a.train, a.ids, a.released_weight, a.ib_weight)
    out = a.output / a.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "weights.jsonl"
    path.write_text("".join(json.dumps(w) + "\n" for w in weights))
    report = {
        "schema": SCHEMA,
        "name": a.name,
        "train": {"path": str(a.train), "sha256": a.train_sha},
        "ids": {"path": str(a.ids), "sha256": sha256(a.ids)},
        "released_weight": a.released_weight,
        "ib_weight": a.ib_weight,
        **report,
        "weights_sha256": sha256(path),
    }
    (out / "weights-report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
