"""Decoder M7 HT-DEV v2 diagnostic: one point against the tier reference <tier>-I, paired on the 1,944 items.

M7's preregistration predates HT-DEV v2 and did not adopt it before its first development readout, so under
COORDINATION 2026-09-30 04:10 it is reported as a diagnostic: never selected on, never a gate. Both prediction files
come from the same M7 collection path (m7-htdev2.sh: v2.dec.infer_dec on the tier's node, image and kernel path,
16,384 tokens, T = 1). Scoring is the eval track's `v2.eval.htdev2.score` (task macro-F1, H_dev2 = mean over the nine
tasks, item bootstrap within tasks, 2,000 draws, seed 20260930); the verdict follows the 04:10 rule: FLAG at
delta <= -0.02, GAIN at delta >= +0.02, TIE between (±0.045 is the preregistered 10%-risk band).

usage: python3 m7_htdev2.py --gold <ht-dev2.gold.jsonl> --left <preds> --left-name <point> \
           --right <preds> --right-name <tier>-I --output <diag>/<point>.htdev2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "dec-m7-htdev2/1"
TIE = 0.02
BAND = 0.045
ROLE = (
    "diagnostic (M7 prereg predates HT-DEV v2, not adopted before the first development readout; "
    "COORDINATION 2026-09-30 04:10); never selected on; development readout, never a release score"
)


def verdict(delta: float) -> str:
    return "FLAG" if delta <= -TIE else "GAIN" if delta >= TIE else "TIE"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(
    htdev2: Any,
    gold_path: Path,
    left_path: Path,
    left_name: str,
    right_path: Path,
    right_name: str,
) -> dict[str, Any]:
    gold = htdev2.read_jsonl(gold_path)
    preds = {
        name: {r["id"]: r for r in htdev2.read_jsonl(path)}
        for name, path in ((left_name, left_path), (right_name, right_path))
    }
    reports = {n: htdev2.report(gold, p, replicates=1) for n, p in preds.items()}
    paired = htdev2.bootstrap(
        htdev2.outcomes(gold, preds[left_name]),
        htdev2.REPLICATES,
        htdev2.SEED,
        other=htdev2.outcomes(gold, preds[right_name]),
    )
    left, right = reports[left_name], reports[right_name]
    delta = left["H_dev2"] - right["H_dev2"]
    return {
        "schema": SCHEMA,
        "role": ROLE,
        "left": left_name,
        "right": right_name,
        "H_dev2": {left_name: left["H_dev2"], right_name: right["H_dev2"]},
        "H_dev2_median": {
            left_name: left["H_dev2_median"],
            right_name: right["H_dev2_median"],
        },
        "delta": delta,
        "ci95": paired["H_dev2"]["ci95"],
        "p_le_0": paired["H_dev2"]["p_le_0"],
        "verdict": verdict(delta),
        "tie_band": TIE,
        "band_10pct_risk": BAND,
        "tasks": {
            task: {
                left_name: left["tasks"][task]["macro_f1_all"],
                right_name: right["tasks"][task]["macro_f1_all"],
            }
            for task in sorted(left["tasks"])
        },
        "invalid_or_missing": {n: r["items"] - r["valid"] for n, r in reports.items()},
        "items": left["items"],
        "files": {
            "gold": sha(gold_path),
            left_name: sha(left_path),
            right_name: sha(right_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--left-name", required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--right-name", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    from v2.eval.htdev2 import score as htdev2

    out = evaluate(htdev2, a.gold, a.left, a.left_name, a.right, a.right_name)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "left": a.left_name,
                "right": a.right_name,
                "delta": round(out["delta"], 4),
                "ci95": [round(x, 4) for x in out["ci95"]],
                "verdict": out["verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
