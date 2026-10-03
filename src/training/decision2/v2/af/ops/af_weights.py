"""Arm factory per-row loss weights for ``train_dec --example-weights`` (M17 / M10's weights format).

A weights arm trains on another arm's locked TRAIN file unchanged; the only change is the weights file. Every TRAIN
row whose id appears in one of the --up files gets --up-weight, every other row 1.0 (KIB4W2: the IB4 phase 1 rows
x2 on KIB4's TRAIN). Checks: TRAIN hashes to --train-sha; TRAIN ids are unique; every --up id that the TRAIN holds
is counted, and the report gives rows and weight totals per block.

usage: af_weights.py --train F --train-sha S --up F [--up F ...] --up-weight 2.0 --output weights.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--train-sha", required=True)
    p.add_argument("--up", type=Path, action="append", required=True)
    p.add_argument("--up-weight", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.up_weight > 0:
        raise SystemExit("--up-weight must be positive")
    if sha256(a.train) != a.train_sha:
        raise SystemExit(f"{a.train} is not {a.train_sha}")
    up_ids: set[str] = set()
    for path in a.up:
        for line in path.read_text().splitlines():
            if line.strip():
                up_ids.add(json.loads(line)["id"])
    seen: set[str] = set()
    rows = {"up": 0, "other": 0}
    weight = {"up": 0.0, "other": 0.0}
    families: dict[str, int] = {}
    out = []
    for line in a.train.read_bytes().splitlines():
        row = json.loads(line)
        rid = row["id"]
        if rid in seen:
            raise SystemExit(f"duplicate id {rid}")
        seen.add(rid)
        block = "up" if rid in up_ids else "other"
        w = a.up_weight if block == "up" else 1.0
        if block == "up":
            fam = str(row.get("family", "?"))
            families[fam] = families.get(fam, 0) + 1
        rows[block] += 1
        weight[block] += w
        out.append(json.dumps({"id": rid, "weight": w}))
    if rows["up"] == 0:
        raise SystemExit("no TRAIN row matched the --up files")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        f.write("\n".join(out) + "\n")
    print(
        json.dumps(
            {
                "schema": "af-weights/1",
                "train_sha256": a.train_sha,
                "up_ids": len(up_ids),
                "rows": rows,
                "weight": weight,
                "up_families": families,
                "up_weight_share": weight["up"] / (weight["up"] + weight["other"]),
                "weights_sha256": sha256(a.output),
            }
        )
    )


if __name__ == "__main__":
    main()
