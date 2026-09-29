"""Gold audit, step 3: aggregates and pair-delta sensitivity (node A, stdlib).

Reads the private judgments (gold_audit_judgments.json: id -> grade a/b/c) and
recomputes the key pair deltas with (c), (b)+(c), and, as a worst case, every
audited item excluded. Also re-measures the A7 planted-quote rate with a looser
upper-bound pattern. Stdout carries aggregates only.
"""

import collections
import glob
import json
import math
import os
import random
import re
import sys

ARGS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
PRIV = "/data/dev2/private/eval/jevbench-value/gap"
PAIRS = [
    ("N4XF", "nox1-adopt"),
    ("N4XF", "decider4b"),
    ("K-a13", "lux1-adopt"),
    ("F1", "eikos27b"),
    ("F1", "autojev27"),
]
LOOSE = re.compile(
    r"\b(wrote|writes|note|notes|comment|remark|draft|recommend\w*|says|said|claims?|"
    r"believes?|suggests?|thinks?|chat|assumption|summary)\b[^\n]{0,120}?"
    r"([\"\u201c]|:\s*\S)",
    re.I,
)


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def binom(b, c):
    n = b + c
    if not n:
        return 1.0
    pm = [math.comb(n, k) * 0.5**n for k in range(n + 1)]
    return min(1.0, sum(x for x in pm if x <= pm[b] * (1 + 1e-9)))


mat = {r["id"]: r for r in jl(PRIV + "/per_item_matrix.jsonl")}
judg = json.load(open(PRIV + "/gold_audit_judgments.json"))
flags = json.load(open(PRIV + "/cue_flags.json"))
ids = list(mat)

out = {
    "grades_by_tier_family": dict(
        collections.Counter(
            f"{mat[i]['tier']}:{mat[i]['family']}:{j['grade']}" for i, j in judg.items()
        )
    ),
    "grades": dict(collections.Counter(j["grade"] for j in judg.values())),
    "audited_cue_flagged": sum(flags[i] for i in judg),
    "audited_n": len(judg),
}

sets = {
    "all_231": set(),
    "excl_c": {i for i, j in judg.items() if j["grade"] == "c"},
    "excl_b_c": {i for i, j in judg.items() if j["grade"] in ("b", "c")},
    "excl_all_audited_worst_case": set(judg),
}
sens = {}
for name, drop in sets.items():
    keep = [i for i in ids if i not in drop]
    row = {"n": len(keep)}
    for a, b in PAIRS:
        bo = sum(mat[i]["runs"][a]["c"] and not mat[i]["runs"][b]["c"] for i in keep)
        co = sum(mat[i]["runs"][b]["c"] and not mat[i]["runs"][a]["c"] for i in keep)
        row[f"{a}__vs__{b}"] = {
            "b": bo,
            "c": co,
            "delta": bo - co,
            "p": round(binom(bo, co), 4),
        }
    sens[name] = row
out["sensitivity"] = sens

snap = ARGS.get("snap")
shape = {}
if snap:
    for path in sorted(glob.glob(os.path.join(snap, "v2/a7/arms/*/train.jsonl"))):
        rnd, res, seen = random.Random(0), [], 0
        for line in open(path):
            if not line.strip():
                continue
            seen += 1
            if len(res) < 4000:
                res.append(line)
            else:
                k = rnd.randrange(seen)
                if k < 4000:
                    res[k] = line
        hits = 0
        for line in res:
            st = json.loads(line).get("state")
            s = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
            hits += bool(LOOSE.search(s))
        shape[os.path.relpath(path, snap)] = {
            "sampled": len(res),
            "rows": seen,
            "loose_share": round(hits / len(res), 4),
        }
    pub = {
        p["id"]: p
        for p in jl("/data/dev2/private/panels/goldfree/public231.prompts.jsonl")
    }
    hard = [p for i, p in pub.items() if i.startswith("hard")]
    hs = [
        (
            p["state"]
            if isinstance(p["state"], str)
            else json.dumps(p["state"], ensure_ascii=False)
        )
        for p in hard
    ]
    shape["public231 hard"] = {
        "sampled": len(hs),
        "loose_share": round(sum(bool(LOOSE.search(s)) for s in hs) / len(hs), 4),
    }
    tf = [
        json.loads(x)["state"]
        for x in open("/data/dev2/private/panels/goldfree/typed-final.prompts.jsonl")
    ]
    ts = [s if isinstance(s, str) else json.dumps(s, ensure_ascii=False) for s in tf]
    shape["typed-final"] = {
        "sampled": len(ts),
        "loose_share": round(sum(bool(LOOSE.search(s)) for s in ts) / len(ts), 4),
    }
out["loose_pattern_shape"] = shape
json.dump(out, sys.stdout, indent=1, sort_keys=True)
