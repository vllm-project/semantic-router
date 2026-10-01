"""Per-family difference between two IX1 runs of one size (private inputs and outputs; no values here).

    python3 -m v2.eval.ix1.family_delta --base A=compare.json --new B=compare.json --out delta.json

Both inputs are ``compare.json`` files written by ``v2.eval.ix1.compare`` over the same panel. Per
benchmark: each run's skill x 100, the difference and its weighted contribution to the headline
(the board weight the compare file carries); per task family (``gap.FAMILIES``): the summed weighted
difference and the benchmarks that moved by more than one skill point; then the port and kit
balanced-skill headlines of both runs and the row counts (ok / error / pending).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .gap import FAMILIES, LOSS


def load(spec: str) -> tuple[str, dict[str, Any]]:
    label, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"expected LABEL=compare.json, got {spec!r}")
    return label, json.loads(Path(path).read_text())


def delta(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old = {r["benchmark"]: r for r in base["benchmarks"]}
    cur = {r["benchmark"]: r for r in new["benchmarks"]}
    if set(old) != set(cur):
        raise SystemExit("the two runs cover different benchmarks")
    rows = {}
    for name in sorted(old):
        weight = old[name]["weight"]
        change = cur[name]["ours"] - old[name]["ours"]
        rows[name] = {
            "base": old[name]["ours"],
            "new": cur[name]["ours"],
            "delta": round(change, 3),
            "weighted_delta": round(weight * change, 3),
        }
    families = []
    for family, names in FAMILIES.items():
        families.append(
            {
                "family": family,
                "weighted_delta": round(
                    sum(rows[n]["weighted_delta"] for n in names), 3
                ),
                "up": [n for n in names if rows[n]["delta"] > LOSS],
                "down": [n for n in names if rows[n]["delta"] < -LOSS],
            }
        )
    families.sort(key=lambda f: -f["weighted_delta"])
    headline = {
        key: {
            "base": base["headline"][key],
            "new": new["headline"][key],
            "delta": round(new["headline"][key] - base["headline"][key], 3),
        }
        for key in ("port_balanced_skill", "kit_balanced_skill")
    }
    return {
        "headline": headline,
        "weighted_delta_sum": round(sum(r["weighted_delta"] for r in rows.values()), 3),
        "families": families,
        "benchmarks": rows,
        "counts": {"base": base["counts"], "new": new["counts"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, help="LABEL=compare.json")
    parser.add_argument("--new", required=True, help="LABEL=compare.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base_label, base = load(args.base)
    new_label, new = load(args.new)
    result = {"base": base_label, "new": new_label, **delta(base, new)}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"{new_label} vs {base_label}: written to {args.out}")


if __name__ == "__main__":
    main()
