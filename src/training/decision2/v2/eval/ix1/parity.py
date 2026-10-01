"""86-request parity gate: kit-runner results with the adapter vs the package's own entry point.

    python3 -m v2.eval.ix1.parity --kit <run>/results.jsonl --ref ref.jsonl --out parity.json

Gate (IX1 prereg §4): every request ``ok`` or declared ``unsupported`` on both sides, the same
status and chosen key for every request and question, and max |Δp| <= 1e-4 over every option
probability and Noul probability. The receipt holds counts, the max |Δp| and mismatching run IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TOLERANCE = 1e-4


def _final(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[record["run_id"]] = record
    return records


def compare(kit: dict[str, dict], ref: dict[str, dict]) -> dict[str, Any]:
    mismatches: list[str] = []
    max_dp = 0.0
    statuses: dict[str, int] = {}
    questions = 0
    if set(kit) != set(ref):
        mismatches.extend(sorted(set(kit) ^ set(ref)))
    for run_id in sorted(set(kit) & set(ref)):
        a, b = kit[run_id], ref[run_id]
        statuses[a["status"]] = statuses.get(a["status"], 0) + 1
        if a["status"] != b["status"] or a["status"] not in ("ok", "unsupported"):
            mismatches.append(run_id)
            continue
        if a["status"] != "ok":
            continue
        left, right = a["response"]["answers"], b["answers"]
        if set(left) != set(right):
            mismatches.append(run_id)
            continue
        for key, x in left.items():
            y = right[key]
            questions += 1
            if x.get("type") != y.get("type") or x.get("choice") != y.get("choice"):
                mismatches.append(run_id)
                break
            if x["type"] == "noul":
                max_dp = max(max_dp, abs(x["noul"] - y["noul"]))
                continue
            if set(x["probabilities"]) != set(y["probabilities"]):
                mismatches.append(run_id)
                break
            for option, p in x["probabilities"].items():
                max_dp = max(max_dp, abs(p - y["probabilities"][option]))
    return {
        "requests": len(set(kit) | set(ref)),
        "statuses": statuses,
        "questions_compared": questions,
        "max_abs_dp": max_dp,
        "tolerance": TOLERANCE,
        "mismatched_run_ids": sorted(set(mismatches)),
        "pass": not mismatches and max_dp <= TOLERANCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "ix1-parity/1", **compare(_final(args.kit), _final(args.ref))}
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("requests", "statuses", "max_abs_dp", "pass")}
        )
    )
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
