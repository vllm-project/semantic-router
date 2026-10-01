"""Private Index input of the K-a13IB release card (schema dev2-card-index/1; values never committed).

The card round-2 input (make_index.py of dev2-card2-2026-10-02) with the 9B point replaced by the independent kit
run on exactly the released weights (IX1 run K-a13IB-bf16: kit index.json, balanced skill and area skills x100, as
round 2 took them) and the footnote of the coordinator decision 2026-10-02 02:05. Every other point is unchanged.

    python3 make_index_ka13ib.py --base decision-index-card.json --kit-index index.json \
        --model-sha256 SHA --parameters N --out decision-index-card.json

Prints only the output's SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v2.release import card_index

FOOTNOTE = (
    "Decision 2.0: independent reproduction with the official {edition} kit on the released weights; training data "
    "has no overlap with Index test items; others: public board snapshot, {snapshot}."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--kit-index", type=Path, required=True)
    ap.add_argument("--model-sha256", required=True)
    ap.add_argument("--parameters", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = json.loads(args.base.read_text())
    kit = json.loads(args.kit_index.read_text())
    if kit["edition"] != base["edition"]:
        raise SystemExit(f"kit edition {kit['edition']} is not {base['edition']}")
    entry = next(p for p in base["family"] if p["tier"] == "9B")
    if entry["parameters"] != args.parameters:
        raise SystemExit("loaded parameter count differs from the round-2 9B point")
    entry.update(
        {
            "balanced_skill": kit["scores"]["balanced_skill"],
            "areas": {a["id"]: 100 * a["skill"] for a in kit["areas"]},
            "model_sha256": args.model_sha256,
            "kit_index_sha256": hashlib.sha256(args.kit_index.read_bytes()).hexdigest(),
        }
    )
    base["footnote"] = FOOTNOTE.format(
        edition=base["edition"], snapshot=base["snapshot"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(base, indent=2) + "\n")
    print(
        json.dumps(
            {"out": str(args.out), "sha256": card_index.load(args.out)["sha256"]}
        )
    )


if __name__ == "__main__":
    main()
