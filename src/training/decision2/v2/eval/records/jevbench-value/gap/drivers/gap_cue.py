"""Skill/cue attribution for the gap analysis (node A, stdlib only; aggregates to stdout).

1. "Embedded human conclusion" cue: a hard item whose state contains a quoted
   assertion attributed to a person/role (note, comment, remark, draft,
   recommendation, message ...). Reports per-run accuracy with/without the cue.
2. Noul yes-bias: accuracy at the 0.5 threshold vs shifted thresholds, and how
   many items a pure threshold shift would recover (an offset artefact) vs not.
3. Per-run accuracy on hard items by input-length bin.
The per-item cue flags are saved privately on node A.
"""

import json
import re
import sys

ARGS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
PRIV = "/data/dev2/private/eval/jevbench-value/gap"
PROMPTS = "/data/dev2/private/panels/goldfree/public231.prompts.jsonl"
RUNS_ROOT = "/data/dev2/runs"


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


CUE = re.compile(
    r"(note|comment|remark|draft|recommend\w*|message|e-mail|email|chat|says|said|"
    r"screener|planner|approver|supervisor|reviewer|clerk|analyst|manager|lead)"
    r"[^\"\u201c\n]{0,80}?[:,]\s*[\"\u201c]",
    re.I,
)

mat = {r["id"]: r for r in jl(PRIV + "/per_item_matrix.jsonl")}
prompts = {p["id"]: p for p in jl(PROMPTS)}
ids = list(mat)
runs = list(next(iter(mat.values()))["runs"])

flags = {}
for i in ids:
    st = prompts[i]["state"]
    s = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
    flags[i] = bool(CUE.search(s))
with open(PRIV + "/cue_flags.json", "w") as f:
    json.dump(flags, f)

hard = [i for i in ids if mat[i]["tier"] == "hard"]
out = {
    "cue_items": {
        "hard": sum(flags[i] for i in hard),
        "standard": sum(flags[i] for i in ids if mat[i]["tier"] == "standard"),
        "easy": sum(flags[i] for i in ids if mat[i]["tier"] == "easy"),
        "hard_by_family": {},
    }
}
for i in hard:
    fam = mat[i]["family"]
    d = out["cue_items"]["hard_by_family"].setdefault(fam, [0, 0])
    d[0] += flags[i]
    d[1] += 1

per_run = {}
for r in runs:
    cue = [i for i in hard if flags[i]]
    nocue = [i for i in hard if not flags[i]]
    lb = {}
    for i in hard:
        k = mat[i]["len_bin_full"]
        lb.setdefault(k, [0, 0])
        lb[k][0] += mat[i]["runs"][r]["c"]
        lb[k][1] += 1
    per_run[r] = {
        "hard_cue": [sum(mat[i]["runs"][r]["c"] for i in cue), len(cue)],
        "hard_nocue": [sum(mat[i]["runs"][r]["c"] for i in nocue), len(nocue)],
        "hard_len": lb,
    }
out["per_run"] = per_run

# Noul threshold analysis needs p_yes; read from predictions.
PATHS = ARGS.get("paths", {})
noul = {}
for r, p in PATHS.items():
    base = p if p.startswith("/") else RUNS_ROOT + "/" + p
    pr = {x["id"]: x for x in jl(base + "/output/public231.predictions.jsonl")}
    its = [i for i in ids if mat[i]["type"] == "noul" and mat[i]["tier"] != "easy"]
    py = {i: pr[i]["answers"]["decision"]["noul"] for i in its}
    gold = {i: mat[i]["expected"] == "yes" for i in its}
    res = {}
    for th in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        res[str(th)] = sum((py[i] > th) == gold[i] for i in its)
    wrong_no = [i for i in its if not gold[i] and py[i] > 0.5]
    wrong_yes = [i for i in its if gold[i] and py[i] <= 0.5]
    noul[r] = {
        "n": len(its),
        "acc_by_threshold": res,
        "false_yes": len(wrong_no),
        "false_yes_p_le_0.65": sum(py[i] <= 0.65 for i in wrong_no),
        "false_yes_p_gt_0.8": sum(py[i] > 0.8 for i in wrong_no),
        "false_no": len(wrong_yes),
        "mean_p_on_gold_no": round(
            sum(py[i] for i in its if not gold[i]) / sum(1 for i in its if not gold[i]),
            4,
        ),
        "mean_p_on_gold_yes": round(
            sum(py[i] for i in its if gold[i]) / sum(1 for i in its if gold[i]), 4
        ),
        "false_yes_with_cue": sum(flags[i] for i in wrong_no),
    }
out["noul_threshold"] = noul


def fisher(a, b, c, d):
    import math

    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    hp = lambda x: math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    obs = hp(a)
    return sum(
        hp(x)
        for x in range(max(0, c1 - r2), min(r1, c1) + 1)
        if hp(x) <= obs * (1 + 1e-9)
    )


pc = {}
for a, b in ARGS.get("pairs", []):
    row = {}
    for lab, sel in (
        ("cue", [i for i in hard if flags[i]]),
        ("nocue", [i for i in hard if not flags[i]]),
    ):
        bo = sum(mat[i]["runs"][a]["c"] and not mat[i]["runs"][b]["c"] for i in sel)
        co = sum(mat[i]["runs"][b]["c"] and not mat[i]["runs"][a]["c"] for i in sel)
        row[lab] = [bo, co]
    row["fisher_cue_vs_nocue"] = round(
        fisher(row["cue"][0], row["cue"][1], row["nocue"][0], row["nocue"][1]), 4
    )
    pc[f"{a}__vs__{b}"] = row
out["pair_cue_discordants"] = pc
json.dump(out, sys.stdout, indent=1, sort_keys=True)
