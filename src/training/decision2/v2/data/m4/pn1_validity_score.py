"""Score the PN1 dev-slice validity check (prereg sec. 7).

Predicted yes means P(true) > 0.5. VALID iff, pooled over the six PAWS-X languages,
(i) yes(A) - yes(B) >= 0.03 with the paired group-bootstrap 95% CI lower bound > 0, and
(ii) the 95% CI lower bound of yes(A) - gold is > 0 (A = N4XF, B = Nox 1.0).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

PAWSX = ("de", "es", "fr", "ja", "ko", "zh")
RESAMPLES = 10_000
MIN_GAP = 0.03


def load_predictions(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            out[record["id"]] = float(record["answers"]["decision"]["noul"])
    return out


def load_gold(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            g = json.loads(line)
            rows.append(
                {
                    "id": g["id"],
                    "group": g["group_id"],
                    "language": g["language"],
                    "family": g["task"].split("/", 1)[1],
                    "gold": bool(g["gold"]["decision"]["value"]),
                }
            )
    return rows


def cell(rows, pa, pb) -> dict[str, Any]:
    n = len(rows)
    ya = sum(pa[r["id"]] > 0.5 for r in rows)
    yb = sum(pb[r["id"]] > 0.5 for r in rows)
    gold = sum(r["gold"] for r in rows)
    acc_a = sum((pa[r["id"]] > 0.5) == r["gold"] for r in rows)
    acc_b = sum((pb[r["id"]] > 0.5) == r["gold"] for r in rows)
    return {
        "n": n,
        "gold_rate": round(gold / n, 4),
        "yes_a": round(ya / n, 4),
        "yes_b": round(yb / n, 4),
        "acc_a": round(acc_a / n, 4),
        "acc_b": round(acc_b / n, 4),
        "gold_no_recall_a": round(
            sum(pa[r["id"]] <= 0.5 for r in rows if not r["gold"]) / max(1, n - gold),
            4,
        ),
        "gold_no_recall_b": round(
            sum(pb[r["id"]] <= 0.5 for r in rows if not r["gold"]) / max(1, n - gold),
            4,
        ),
    }


def bootstrap(rows, pa, pb, seed: int = 20260929) -> dict[str, Any]:
    groups: dict[str, list[tuple[int, int, int]]] = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(
            (int(pa[r["id"]] > 0.5), int(pb[r["id"]] > 0.5), int(r["gold"]))
        )
    keys = sorted(groups)
    sums = [tuple(map(sum, zip(*groups[k]))) + (len(groups[k]),) for k in keys]
    rng = random.Random(seed)
    diff_ab, diff_ag = [], []
    for _ in range(RESAMPLES):
        a = b = g = n = 0
        for _ in keys:
            sa, sb, sg, sn = sums[rng.randrange(len(keys))]
            a, b, g, n = a + sa, b + sb, g + sg, n + sn
        diff_ab.append((a - b) / n)
        diff_ag.append((a - g) / n)
    diff_ab.sort()
    diff_ag.sort()

    def ci(values):
        return [
            round(values[int(0.025 * RESAMPLES)], 4),
            round(values[int(0.975 * RESAMPLES) - 1], 4),
        ]

    total = [sum(col) for col in zip(*sums)]
    return {
        "groups": len(keys),
        "yes_a_minus_yes_b": round((total[0] - total[1]) / total[3], 4),
        "yes_a_minus_yes_b_ci95": ci(diff_ab),
        "yes_a_minus_gold": round((total[0] - total[2]) / total[3], 4),
        "yes_a_minus_gold_ci95": ci(diff_ag),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True, help="N4XF predictions")
    parser.add_argument("--b", type=Path, required=True, help="Nox 1.0 predictions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    gold = load_gold(args.gold)
    pa, pb = load_predictions(args.a), load_predictions(args.b)
    missing = [r["id"] for r in gold if r["id"] not in pa or r["id"] not in pb]
    if missing:
        raise SystemExit(f"{len(missing)} gold ids without predictions")
    pawsx = [r for r in gold if r["language"] in PAWSX]
    report: dict[str, Any] = {
        "a": "N4XF",
        "b": "Nox 1.0",
        "predicted_yes": "P(true) > 0.5",
        "per_language": {
            lang: cell([r for r in gold if r["language"] == lang], pa, pb)
            for lang in sorted({r["language"] for r in gold})
        },
        "per_family": {
            fam: cell([r for r in pawsx if r["family"] == fam], pa, pb)
            for fam in sorted({r["family"] for r in gold})
        },
        "pooled_pawsx6": cell(pawsx, pa, pb),
        "pooled_all8": cell(gold, pa, pb),
        "bootstrap_pawsx6": bootstrap(pawsx, pa, pb),
        "bootstrap_all8": bootstrap(gold, pa, pb),
    }
    boot = report["bootstrap_pawsx6"]
    report["valid"] = bool(
        boot["yes_a_minus_yes_b"] >= MIN_GAP
        and boot["yes_a_minus_yes_b_ci95"][0] > 0
        and boot["yes_a_minus_gold_ci95"][0] > 0
    )
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"valid": report["valid"], "pawsx6": boot}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
