"""Decoder M14 per-row loss weights (prereg dec-m14-prereg-2026-10-01.md, "Weights").

An M14 arm trains on its M12 arm's locked TRAIN file unchanged; the only change is a weights file for
``train_dec --example-weights``. M12's builder wrote the tier's released TRAIN first (byte for byte, file order) and
the IB rows after it, and its ``train.ids.jsonl`` names each line's block (``base`` = released). Here every released
(typed) row gets the preregistered weight of its task type and every IB row (copies included) gets the IB weight.

Checks: the ids file and TRAIN agree line by line; the released block is exactly the first N lines, its bytes hash to
the released TRAIN's SHA-256, and no released line follows an IB line. The report gives rows and weight totals per
block and type and the released rows' share of the total weight (the trainer normalizes weights within each update
window, so this share is what moves).

usage: m14_weights.py --train F --train-sha S --ids F --ids-sha S --released-rows N --released-sha S \
         --weight choice=1.5,noul=1.5,score=2.0 [--ib-weight 1.0] --name ARM --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "dec-m14-weights/1"
TASK_TYPES = ("choice", "noul", "score")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_weights(spec: str) -> dict[str, float]:
    weights = {}
    for part in spec.split(","):
        kind, value = part.split("=", 1)
        if kind not in TASK_TYPES or kind in weights:
            raise ValueError(f"{spec}: unknown or repeated type {kind}")
        weights[kind] = float(value)
    if set(weights) != set(TASK_TYPES):
        raise ValueError(f"{spec}: needs a weight for each of {TASK_TYPES}")
    if any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError(f"{spec}: weights must be positive and finite")
    return weights


def build(
    train_lines: list[bytes],
    id_lines: list[bytes],
    released_rows: int,
    released_sha: str,
    typed: dict[str, float],
    ib_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Weight records in TRAIN order and the report body."""
    if len(train_lines) != len(id_lines):
        raise ValueError("TRAIN and ids files differ in length")
    if not math.isfinite(ib_weight) or ib_weight <= 0:
        raise ValueError("IB weight must be positive and finite")
    released_digest = hashlib.sha256()
    records = []
    totals: dict[str, dict[str, dict[str, float]]] = {"released": {}, "ib": {}}
    seen_ib = False
    for number, (line, id_line) in enumerate(zip(train_lines, id_lines), 1):
        row, ids = json.loads(line), json.loads(id_line)
        if row["id"] != ids["id"]:
            raise ValueError(
                f"line {number}: TRAIN id {row['id']} != ids file {ids['id']}"
            )
        kind = row["task_type"]
        if kind not in TASK_TYPES:
            raise ValueError(f"line {number}: unknown task type {kind}")
        if ids["block"] == "base":
            if seen_ib:
                raise ValueError(f"line {number}: released row after an IB row")
            block, weight = "released", typed[kind]
            released_digest.update(line)
        else:
            seen_ib = True
            block, weight = "ib", ib_weight
        cell = totals[block].setdefault(kind, {"rows": 0, "weight": 0.0})
        cell["rows"] += 1
        cell["weight"] += weight
        records.append({"id": row["id"], "weight": weight})
    got = sum(c["rows"] for c in totals["released"].values())
    if got != released_rows:
        raise ValueError(f"{got} released rows, expected {released_rows}")
    if released_digest.hexdigest() != released_sha:
        raise ValueError("the released block does not hash to the released TRAIN")
    weight_total = math.fsum(r["weight"] for r in records)
    released_weight = math.fsum(c["weight"] for c in totals["released"].values())
    rows_total = len(records)
    report = {
        "rows": rows_total,
        "released_rows": released_rows,
        "ib_rows": rows_total - released_rows,
        "typed_weights": typed,
        "ib_weight": ib_weight,
        "by_block_type": totals,
        "released_weight_share": released_weight / weight_total,
        "released_row_share": released_rows / rows_total,
        "type_weight_share": {
            kind: math.fsum(totals[b].get(kind, {}).get("weight", 0.0) for b in totals)
            / weight_total
            for kind in TASK_TYPES
        },
    }
    return records, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--train-sha", required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--ids-sha", required=True)
    p.add_argument("--released-rows", type=int, required=True)
    p.add_argument("--released-sha", required=True)
    p.add_argument(
        "--weight", required=True, help="choice=W,noul=W,score=W for released rows"
    )
    p.add_argument("--ib-weight", type=float, default=1.0)
    p.add_argument("--name", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    for path, want in ((a.train, a.train_sha), (a.ids, a.ids_sha)):
        if sha256(path) != want:
            raise ValueError(f"{path}: sha256 is not {want}")
    with a.train.open("rb") as stream:
        train_lines = stream.readlines()
    with a.ids.open("rb") as stream:
        id_lines = stream.readlines()
    records, body = build(
        train_lines,
        id_lines,
        a.released_rows,
        a.released_sha,
        parse_weights(a.weight),
        a.ib_weight,
    )
    out = a.output / a.name
    out.mkdir(parents=True, exist_ok=False)
    weights = out / "weights.jsonl"
    with weights.open("x", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "schema": SCHEMA,
        "arm": a.name,
        "train": {"path": str(a.train), "sha256": a.train_sha},
        "ids": {"path": str(a.ids), "sha256": a.ids_sha},
        "released_sha256": a.released_sha,
        "weights_sha256": sha256(weights),
        **body,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                "arm": a.name,
                "weights_sha256": report["weights_sha256"],
                "rows": body["rows"],
                "released_weight_share": round(body["released_weight_share"], 4),
                "released_row_share": round(body["released_row_share"], 4),
                "type_weight_share": {
                    k: round(v, 4) for k, v in body["type_weight_share"].items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
