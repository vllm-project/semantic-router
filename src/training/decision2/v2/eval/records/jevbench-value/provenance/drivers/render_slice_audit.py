"""Accuracy on rendering-sensitive slices of public 231 (aggregates only).

Stdlib only; run on node A via stdin. Arg: JSON {"runs_root", "targets", "prompts", "runs": [..]}.
Slices: question type; Score level count; dict vs string state; Noul criteria key order.
Uses the same accuracy rule as jev_arena/jevbench_public.py (verified identical to the
upstream-faithful rule by rescore_upstream_faithful.py).
"""

import json
import math
import os
import sys
from collections import Counter


def predicted(ans, t):
    if not isinstance(ans, dict) or ans.get("type", t["task_type"]) != t["task_type"]:
        return None
    if t["task_type"] == "noul":
        p = ans.get("noul")
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            return None
        probs = {"no": 1 - p, "yes": p}
    else:
        pr = ans.get("probabilities")
        if not isinstance(pr, dict) or set(pr) != set(t["labels"]):
            return None
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in pr.values()
        ):
            return None
        s = sum(pr.values())
        if s <= 0 or abs(s - 1) > 0.02:
            return None
        probs = pr
    return min(t["labels"], key=lambda k: (-probs[k], k))


def main():
    a = json.loads(sys.argv[1])
    targets = {}
    for line in open(a["targets"]):
        t = json.loads(line)
        targets[t["id"]] = t
    meta = {}
    for line in open(a["prompts"]):
        p = json.loads(line)
        q = p["questions"]["decision"]
        crit = q.get("criteria")
        m = {"dict_state": isinstance(p["state"], dict)}
        if q["type"] == "score":
            m["levels"] = len(crit)
        if q["type"] == "noul":
            m["noul_order"] = "true_first" if list(crit)[0] == "true" else "false_first"
        meta[p["id"]] = m
    panel = Counter()
    for tid, t in targets.items():
        m = meta[tid]
        panel["type_" + t["task_type"]] += 1
        panel["dict_state_" + str(m["dict_state"])] += 1
        if "levels" in m:
            panel[f"score_levels_{m['levels']}"] += 1
        if "noul_order" in m:
            panel["noul_" + m["noul_order"]] += 1
    out = {"panel": dict(panel), "runs": {}}
    for rel in a["runs"]:
        rows = {}
        for line in open(
            os.path.join(a["runs_root"], rel, "output", "public231.predictions.jsonl")
        ):
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
        c = Counter()
        for tid, t in targets.items():
            m = meta[tid]
            ok = predicted(
                ((rows.get(tid) or {}).get("answers") or {}).get("decision"), t
            ) == str(t["expected"])
            keys = [
                "type_" + t["task_type"],
                "dict_state_" + str(m["dict_state"]),
                f"{t['tier']}_type_{t['task_type']}",
            ]
            if "levels" in m:
                keys.append(f"score_levels_{m['levels']}")
            if "noul_order" in m:
                keys.append("noul_" + m["noul_order"])
            for k in keys:
                c[k + "_n"] += 1
                c[k + "_correct"] += ok
        out["runs"][rel] = dict(sorted(c.items()))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
