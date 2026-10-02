"""Kit-run records for ``python -m v2.release.card_index --runs`` from IX1 run directories (stdlib; private output).

Each NAME=DIR names an IX1 run whose merged/receipt.json (ix1-run-receipt/1) gives the scored weights and whose
merged/kit/index.json is the official kit's Index output. The output is one JSON object keyed by name with
model_id, model_sha256, edition, balanced_skill, areas (area id -> skill x 100) and index_sha256, the format
card_index reads. Prints only counts and the output's SHA-256, never a value.

    python3 index_runs.py --run DEV2.0-0.6B=/data/dev2/private/eval/index021/ix1/runs/DEV2.0-0.6B ... --out runs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def record(run: Path) -> dict:
    receipt = json.loads((run / "merged/receipt.json").read_text())
    if receipt.get("schema") != "ix1-run-receipt/1":
        raise SystemExit(f"{run}: not an ix1 run receipt")
    index_path = run / "merged/kit/index.json"
    index = json.loads(index_path.read_text())
    source = receipt["model_source"]
    return {
        "model_id": source["model_id"],
        "model_sha256": source["model_sha256"],
        "edition": index["edition"],
        "balanced_skill": index["scores"]["balanced_skill"],
        "areas": {a["id"]: 100 * a["skill"] for a in index["areas"]},
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "results_sha256": receipt["results_sha256"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs = {}
    for item in args.run:
        name, _, path = item.partition("=")
        if not name or not path or name in runs:
            raise SystemExit(f"bad or repeated --run {item!r}")
        runs[name] = record(Path(path))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        args.out.write_text(json.dumps(runs, sort_keys=True) + "\n")
    finally:
        os.umask(old)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "runs": len(runs),
                "weights": {n: r["model_sha256"][:12] for n, r in runs.items()},
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
