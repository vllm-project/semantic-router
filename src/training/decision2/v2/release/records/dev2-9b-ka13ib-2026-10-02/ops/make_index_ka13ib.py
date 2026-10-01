"""Private Index input of the K-a13IB release card (schema dev2-card-index/1; values never committed).

The card round-3 input (dev2-card3-2026-10-02: board-served parameter counts, the audited footnote of
card_index.FOOTNOTE) with the 9B point's scores replaced by the independent kit run on exactly the released weights
(IX1 run K-a13IB-bf16: kit index.json, balanced skill and area skills x100, as round 2 took them). The 9B point keeps
its board-served and loaded parameter counts (same architecture); every other point is unchanged.

    python3 make_index_ka13ib.py --base decision-index-card.json --kit-index index.json \
        --model-sha256 SHA --loaded-parameters N --out decision-index-card.json

Prints only the output's SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v2.release import card_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--kit-index", type=Path, required=True)
    ap.add_argument("--model-sha256", required=True)
    ap.add_argument("--loaded-parameters", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    card_index.load(args.base)
    base = json.loads(args.base.read_text())
    kit = json.loads(args.kit_index.read_text())
    if kit["edition"] != base["edition"]:
        raise SystemExit(f"kit edition {kit['edition']} is not {base['edition']}")
    entry = next(p for p in base["family"] if p["tier"] == "9B")
    if entry["loaded_parameters"] != args.loaded_parameters:
        raise SystemExit("loaded parameter count differs from the round-3 9B point")
    entry.update(
        {
            "balanced_skill": kit["scores"]["balanced_skill"],
            "areas": {a["id"]: 100 * a["skill"] for a in kit["areas"]},
            "model_sha256": args.model_sha256,
            "kit_index_sha256": hashlib.sha256(args.kit_index.read_bytes()).hexdigest(),
        }
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
