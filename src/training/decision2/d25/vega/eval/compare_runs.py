"""Per-question decision agreement between two kit result files on their shared ok requests.

    python -m d25.vega.eval.compare_runs A/results.jsonl B/results.jsonl [--out report.json]

A decision is the choice key (``choice`` questions) or ``noul >= 0.5``; the scorer reads nothing else.
Reports agreement overall and per benchmark, the largest probability difference, and the disagreeing
question ids (useful for batch-composition, ROCm-vs-CUDA and answer-validation checks).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path


def load(path: str) -> dict:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    out = {}
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    out[r["run_id"]] = r
    return out


def decision(answer: dict):
    return answer["choice"] if answer["type"] == "choice" else answer["noul"] >= 0.5


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    a, b = load(args.a), load(args.b)
    shared = sorted(set(a) & set(b))
    per = collections.defaultdict(lambda: [0, 0])
    flips, max_diff = [], 0.0
    for rid in shared:
        ra, rb = a[rid], b[rid]
        bench = ra.get("catalog_id")
        for key, qa in ra["response"]["answers"].items():
            qb = rb["response"]["answers"][key]
            same = decision(qa) == decision(qb)
            per[bench][0] += same
            per[bench][1] += 1
            if qa["type"] == "choice":
                diff = max(
                    abs(qa["probabilities"][k] - qb["probabilities"][k])
                    for k in qa["probabilities"]
                )
            else:
                diff = abs(qa["noul"] - qb["noul"])
            max_diff = max(max_diff, diff)
            if not same:
                flips.append({"run_id": rid, "question": key, "catalog_id": bench})
    agree = sum(v[0] for v in per.values())
    total = sum(v[1] for v in per.values())
    report = {
        "requests_shared": len(shared),
        "questions": total,
        "agree": agree,
        "agreement": round(agree / total, 6) if total else None,
        "max_abs_prob_diff": max_diff,
        "per_benchmark": {
            str(k): {"agree": v[0], "questions": v[1]} for k, v in sorted(per.items())
        },
        "flips": flips[:200],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "requests_shared",
                    "questions",
                    "agree",
                    "agreement",
                    "max_abs_prob_diff",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
