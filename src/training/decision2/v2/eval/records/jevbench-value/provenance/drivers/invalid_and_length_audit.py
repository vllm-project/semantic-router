"""Aggregate audit of invalid answers and input lengths on public 231.

Stdlib only; run on node A via stdin. Aggregates only (no ids/text/gold).
Arg: JSON {"runs_root": ..., "targets": ..., "prompts": ..., "runs": [relative run dirs]}.
For each run: invalid-answer shapes (answer keys, error strings, row-level error fields),
invalid counts by tier/type and by state-length band, and max usage.input_tokens.
Also the panel's state-length distribution (chars) by tier.
"""

import json
import math
import os
import sys
from collections import Counter


def slen(state):
    return len(
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    )


def band(n):
    for hi in (1000, 2000, 4000, 8000):
        if n <= hi:
            return f"<={hi}"
    return ">8000"


def is_valid(ans, t):
    if not isinstance(ans, dict) or ans.get("type", t["task_type"]) != t["task_type"]:
        return False
    if t["task_type"] == "noul":
        p = ans.get("noul")
        return type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
    pr = ans.get("probabilities")
    if not isinstance(pr, dict) or set(pr) != set(t["labels"]):
        return False
    vals = list(pr.values())
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
        for v in vals
    ):
        return False
    return abs(sum(vals) - 1) <= 0.02 and sum(vals) > 0


def main():
    args = json.loads(sys.argv[1])
    targets = {}
    for line in open(args["targets"]):
        t = json.loads(line)
        targets[t["id"]] = t
    lengths, types_by_len = {}, {}
    for line in open(args["prompts"]):
        p = json.loads(line)
        lengths[p["id"]] = slen(p["state"])
    panel = {}
    for tier in ("easy", "standard", "hard"):
        ls = sorted(lengths[i] for i, t in targets.items() if t["tier"] == tier)
        panel[tier] = {
            "n": len(ls),
            "median_chars": ls[len(ls) // 2],
            "max_chars": ls[-1],
            "bands": dict(Counter(band(n) for n in ls)),
        }
    out = {"panel_state_chars": panel, "runs": {}}
    for rel in args["runs"]:
        pred = os.path.join(
            args["runs_root"], rel, "output", "public231.predictions.jsonl"
        )
        rows = {}
        for line in open(pred):
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
        c = Counter()
        shapes, errors, rowkeys = Counter(), Counter(), Counter()
        max_in, inv_chars = 0, []
        for tid, t in targets.items():
            row = rows.get(tid, {})
            u = row.get("usage") or {}
            for k in ("input_tokens", "prompt_tokens"):
                if isinstance(u.get(k), int):
                    max_in = max(max_in, u[k])
            ans = (row.get("answers") or {}).get("decision")
            if is_valid(ans, t):
                continue
            c["invalid"] += 1
            c["invalid_" + t["tier"]] += 1
            c["invalid_type_" + t["task_type"]] += 1
            c["invalid_band_" + band(lengths[tid])] += 1
            inv_chars.append(lengths[tid])
            shapes[
                (
                    ",".join(sorted(ans.keys()))
                    if isinstance(ans, dict)
                    else repr(type(ans).__name__)
                )
            ] += 1
            if isinstance(ans, dict):
                for k in ("error", "reason", "status"):
                    if k in ans:
                        errors[str(ans[k])[:80]] += 1
            for k in (
                "native_error",
                "adapter_errors",
                "error",
                "errors",
                "status",
                "adapter_status",
                "overflow",
                "invalid_reason",
            ):
                if row.get(k):
                    v = row[k]
                    if isinstance(v, dict):
                        v = (
                            ",".join(
                                sorted(str(x)[:40] for x in v.get("kind", v).values())
                            )
                            if "kind" not in v
                            else v["kind"]
                        )
                    rowkeys[f"{k}={str(v)[:80]}"] += 1
        inv_chars.sort()
        out["runs"][rel] = {
            **dict(c),
            "max_usage_input_tokens": max_in or None,
            "invalid_answer_shapes": dict(shapes),
            "invalid_answer_errors": dict(errors),
            "invalid_row_fields": dict(rowkeys),
            "invalid_state_chars_min": inv_chars[0] if inv_chars else None,
            "invalid_state_chars_median": (
                inv_chars[len(inv_chars) // 2] if inv_chars else None
            ),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
