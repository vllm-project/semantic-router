"""Public-sample harness check: an entrant's answers from our run of its server vs its published public rows.

    python -m d25.vega.eval.proxy.harness_check --ours SAMPLE_RESULTS.jsonl --published RESULTS.jsonl.gz --out OUT.json
Per question: top-1 agreement (choice key, noul side, score argmax) and |delta p| of the published top option.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics as st
from pathlib import Path


def records(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def top(a):
    if not isinstance(a, dict):
        return None, None
    probs = a.get("probabilities") or a.get("distribution")
    if "choice" in a:
        k = a["choice"]
        return ("choice", k), (probs or {}).get(k) if isinstance(probs, dict) else None
    if "noul" in a:
        return ("noul", a["noul"] >= 0.5), a["noul"]
    if isinstance(probs, dict) and probs:
        k = max(probs, key=probs.get)
        return ("score", k), probs[k]
    if "score" in a:
        return ("score", a["score"]), None
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ours = {r["run_id"]: r for r in records(a.ours)}
    pub = {}
    for r in records(a.published):
        if r.get("run_id") in ours:
            pub[r["run_id"]] = r
    agree, dp, by_bench = [], [], collections.defaultdict(list)
    status = collections.Counter(r["status"] for r in ours.values())
    for rid, r in ours.items():
        if r["status"] != "ok" or rid not in pub:
            continue
        pa = pub[rid]["response"]["answers"]
        for k, ans in r["response"]["answers"].items():
            t1, p1 = top(ans)
            t2, p2 = top(pa.get(k))
            if t1 is None or t2 is None:
                continue
            same = t1 == t2
            agree.append(same)
            by_bench[r.get("catalog_id")].append(same)
            if p1 is not None and p2 is not None:
                dp.append(abs(p1 - p2))
    res = {
        "requests": len(ours),
        "statuses": dict(status),
        "matched_published": len(pub),
        "questions": len(agree),
        "top1_agreement": round(sum(agree) / len(agree), 4) if agree else None,
        "mean_abs_dp_top_option": round(st.mean(dp), 4) if dp else None,
        "median_abs_dp_top_option": round(st.median(dp), 4) if dp else None,
        "benchmarks_below_95pct": {
            str(b): round(sum(v) / len(v), 3)
            for b, v in sorted(by_bench.items(), key=lambda kv: str(kv[0]))
            if sum(v) / len(v) < 0.95
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
