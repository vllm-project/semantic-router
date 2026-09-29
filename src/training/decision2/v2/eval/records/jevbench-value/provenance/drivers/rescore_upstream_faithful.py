"""Re-score stored public-231 predictions: our scorer vs an upstream-faithful variant.

Stdlib only; run on node A via stdin. Emits aggregates only (no item ids/text/gold).
Arg: JSON {"runs_root": ..., "targets": ..., "prompts": ...}.

Upstream-faithful = jevbench/adapters/typesafe.py answer mapping + jevbench/scoring.py:
  * answer must be a dict whose "type" equals the question type (missing type -> invalid)
  * noul: numeric non-bool p in [0,1] -> {"yes": p, "no": 1-p}
  * choice: answer["choice"] must be one of the labels, probabilities must be a dict
  * score: probabilities must be a dict
  * validate_probs: exact key set, finite numbers in [0,1], |sum-1| <= 0.02 (renormalize)
  * argmax with lexicographically smallest label on ties; invalid -> wrong
"""

import json
import math
import os
import sys
from collections import Counter

RENORM_TOL = 0.02


def is_num(v):
    return not isinstance(v, bool) and isinstance(v, (int, float))


def valid_probs(raw, labels):
    if not isinstance(raw, dict) or set(raw) != set(labels):
        return None
    out = {}
    for k in labels:
        v = raw[k]
        if not is_num(v) or not math.isfinite(float(v)) or not 0 <= v <= 1:
            return None
        out[k] = float(v)
    total = sum(out.values())
    if total <= 0 or abs(total - 1) > RENORM_TOL:
        return None
    return {k: v / total for k, v in out.items()}


def argmax(probs):
    best, best_p = None, -1.0
    for k in sorted(probs):
        if probs[k] > best_p:
            best, best_p = k, probs[k]
    return best


def ours(answer, t):
    # Mirror of jev_arena/jevbench_public.py _evaluate (accuracy-relevant part).
    if not isinstance(answer, dict):
        return None
    q = t["task_type"]
    if answer.get("type", q) != q:
        return None
    if q == "noul":
        p = answer.get("noul")
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            return None
        probs = {"no": 1 - float(p), "yes": float(p)}
    else:
        probs = valid_probs(answer.get("probabilities"), t["labels"])
        if probs is None:
            return None
    return min(t["labels"], key=lambda k: (-probs[k], k))


def upstream(answer, t):
    if not isinstance(answer, dict):
        return None
    q = t["task_type"]
    if answer.get("type") != q:
        return None
    if q == "noul":
        p = answer.get("noul")
        if not is_num(p) or not 0.0 <= float(p) <= 1.0:
            return None
        probs = {"yes": float(p), "no": 1.0 - float(p)}
        probs = valid_probs(probs, t["labels"])
    elif q == "choice":
        if answer.get("choice") not in t["labels"]:
            return None
        probs = valid_probs(answer.get("probabilities"), t["labels"])
    else:
        probs = valid_probs(answer.get("probabilities"), t["labels"])
    if probs is None:
        return None
    return argmax(probs)


def over_budget(row):
    if row.get("native_error"):
        return True
    errs = row.get("adapter_errors") or {}
    if any("max_length" in str(v) or "budget" in str(v) for v in errs.values()):
        return True
    for a in (row.get("answers") or {}).values():
        if isinstance(a, dict) and (
            "budget" in str(a.get("error", ""))
            or "max_length" in str(a.get("error", ""))
        ):
            return True
    return False


def main():
    args = json.loads(sys.argv[1])
    targets = {}
    with open(args["targets"]) as f:
        for line in f:
            t = json.loads(line)
            targets[t["id"]] = t
    lengths = {}
    with open(args["prompts"]) as f:
        for line in f:
            p = json.loads(line)
            s = p["state"]
            lengths[p["id"]] = len(
                s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
            )
    out = []
    for dirpath, dirnames, filenames in os.walk(args["runs_root"]):
        if "private" in dirpath.split(os.sep):
            continue
        if "REPORT.json" not in filenames:
            continue
        pred = os.path.join(dirpath, "output", "public231.predictions.jsonl")
        if not os.path.isfile(pred):
            continue
        try:
            rep = json.load(open(os.path.join(dirpath, "REPORT.json")))
        except Exception:
            continue
        pub = (rep.get("panels") or {}).get("public231")
        if not pub:
            continue
        rows = {}
        with open(pred) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r.get("id")] = r
        agg = {
            "run": os.path.relpath(dirpath, args["runs_root"]),
            "label": (rep.get("model") or {}).get("label"),
            "report_correct": pub.get("correct"),
            "report_tiers": {
                k: v.get("correct") for k, v in (pub.get("tiers") or {}).items()
            },
            "report_valid": pub.get("valid"),
        }
        c = Counter()
        for tid, t in targets.items():
            tier = t["tier"]
            row = rows.get(tid)
            ans = ((row or {}).get("answers") or {}).get("decision")
            exp = str(t["expected"])
            po, pu = ours(ans, t), upstream(ans, t)
            c["ours_correct"] += po == exp
            c["ours_correct_" + tier] += po == exp
            c["up_correct"] += pu == exp
            c["up_correct_" + tier] += pu == exp
            c["ours_invalid"] += po is None
            c["up_invalid"] += pu is None
            c["outcome_flips"] += (po == exp) != (pu == exp)
            if row is None:
                c["missing_row"] += 1
            elif over_budget(row):
                c["over_budget"] += 1
                c["over_budget_" + tier] += 1
                c["over_budget_state_chars_min"] = min(
                    c.get("over_budget_state_chars_min", 10**9), lengths[tid]
                )
            if isinstance(ans, dict):
                if "type" not in ans:
                    c["answer_missing_type"] += 1
                if t["task_type"] == "choice":
                    ch = ans.get("choice")
                    if ch is None:
                        c["choice_point_missing"] += 1
                    elif ch not in t["labels"]:
                        c["choice_point_not_label"] += 1
                    elif po is not None and ch != po:
                        c["choice_point_ne_argmax"] += 1
                if (
                    t["task_type"] == "noul"
                    and is_num(ans.get("noul"))
                    and ans.get("noul") == 0.5
                ):
                    c["noul_exact_half"] += 1
                pr = ans.get("probabilities")
                if isinstance(pr, dict) and t["task_type"] != "noul":
                    vals = [v for v in pr.values() if is_num(v)]
                    if vals and abs(sum(vals) - 1) > 0.001:
                        c["renormalized_or_outside"] += 1
            elif row is not None:
                c["answer_null"] += 1
        agg.update(dict(c))
        out.append(agg)
    out.sort(key=lambda a: a["run"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
