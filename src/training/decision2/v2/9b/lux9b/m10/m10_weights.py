"""9B M10 per-row loss weights (the M14 "UP" lever on a 9B stage-3 TRAIN).

The stage-3 builds (``lux9b/m9_data.py --match-tokens``) write the kept x60 lines first and the IB lines after them,
and the manifest records ``x60_rows_kept``. Every x60 (released-recipe) row gets ``--x60-weight``, every IB row
weight 1. The trainer normalizes the weights within each 64-row update window (``train_dec --example-weights``), so
the TRAIN bytes, windows, update count and LR schedule equal the unweighted arm's for the same seed.

The x60 block is checked against the released x60 TRAIN: every one of the first ``x60_rows_kept`` ids is an x60 id
and no later id is.

usage: m10_weights.py --train-dir D --x60-train F --x60-weight W --output F
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--x60-train", type=Path, required=True)
    parser.add_argument("--x60-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.x60_weight) or args.x60_weight <= 0:
        parser.error("--x60-weight must be positive and finite")
    manifest = json.loads((args.train_dir / "manifest.json").read_text())
    train = args.train_dir / "train.jsonl"
    if sha256(train) != manifest["train_sha256"]:
        raise SystemExit("train.jsonl differs from its manifest")
    n_x60 = int(manifest["x60_rows_kept"])
    with args.x60_train.open(encoding="utf-8") as stream:
        x60_ids = {json.loads(line)["id"] for line in stream if line.strip()}
    rows = x60_w = ib_w = 0.0
    with train.open(encoding="utf-8") as stream, args.output.open(
        "x", encoding="utf-8"
    ) as out:
        for i, line in enumerate(stream):
            rid = json.loads(line)["id"]
            in_x60 = rid in x60_ids
            if in_x60 != (i < n_x60):
                raise SystemExit(
                    f"row {i} ({rid}): x60 membership does not match the block layout"
                )
            w = args.x60_weight if in_x60 else 1.0
            out.write(json.dumps({"id": rid, "weight": w}) + "\n")
            rows += 1
            x60_w += w if in_x60 else 0.0
            ib_w += 0.0 if in_x60 else w
    report = {
        "schema": "lux9b-m10-weights/1",
        "train_sha256": manifest["train_sha256"],
        "x60_train_sha256": sha256(args.x60_train),
        "x60_weight": args.x60_weight,
        "rows": int(rows),
        "x60_rows": n_x60,
        "ib_rows": int(rows) - n_x60,
        "x60_weight_share": round(x60_w / (x60_w + ib_w), 4),
        "weights_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
