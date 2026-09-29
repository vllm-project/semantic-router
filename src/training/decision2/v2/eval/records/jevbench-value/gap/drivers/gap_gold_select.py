"""Gold-quality audit, step 1: difficulty bins and the audit list (node A, stdlib).

Difficulty p = share of the 66 distinct models (stats helper, item_stats.json
"all_distinct") answering the item correctly. Audit list:
  (A) hard or standard items with p < .15;
  (B) items where >= 3 strong models agree on the same non-gold answer with
      confidence >= .8 (strong = Eikos-27B, AutoJev-27B, JPT-4B, JPT-9B,
      DEV2.0-27B F1, 27B C0, Decider 4B).
Aggregates go to stdout; the audit list is written privately.
With {"show": true}, the private items are printed for manual reading instead.
"""

import collections
import json
import sys

ARGS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
PRIV = "/data/dev2/private/eval/jevbench-value/gap"
STATS = "/data/dev2/private/eval/jevbench-value/stats/item_stats.json"
GOLD = "/data/dev2/private/panels/gold/public231/targets.jsonl"
PROMPTS = "/data/dev2/private/panels/goldfree/public231.prompts.jsonl"
STRONG = ["eikos27b", "autojev27", "jpt4b", "jpt9b", "F1", "C0", "decider4b"]
BINS = [(0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 1.01)]


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


p = {k: v["p"] for k, v in json.load(open(STATS))["all_distinct"].items()}
targets = {t["id"]: t for t in jl(GOLD)}
mat = {r["id"]: r for r in jl(PRIV + "/per_item_matrix.jsonl")}
flags = json.load(open(PRIV + "/cue_flags.json"))

bins = collections.defaultdict(lambda: [0] * len(BINS))
for i, t in targets.items():
    for k, (lo, hi) in enumerate(BINS):
        if lo <= p[i] < hi:
            bins[t["tier"]][k] += 1

audit = {}
for i, t in targets.items():
    reasons = []
    if t["tier"] in ("hard", "standard") and p[i] < 0.15:
        reasons.append("p<.15")
    wrong = collections.Counter(
        mat[i]["runs"][m]["pred"]
        for m in STRONG
        if not mat[i]["runs"][m]["c"] and (mat[i]["runs"][m]["conf"] or 0) >= 0.8
    )
    if wrong and wrong.most_common(1)[0][1] >= 3:
        reasons.append("strong-consensus-nongold")
    if reasons:
        audit[i] = {
            "reasons": reasons,
            "p": p[i],
            "consensus": wrong.most_common(1)[0] if wrong else None,
        }

if ARGS.get("show"):
    prompts = {x["id"]: x for x in jl(PROMPTS)}
    only = ARGS.get("ids")
    for i, a in audit.items():
        if only and i not in only:
            continue
        t, q = targets[i], prompts[i]["questions"]["decision"]
        st = prompts[i]["state"]
        s = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
        strong = {
            m: (mat[i]["runs"][m]["pred"], round(mat[i]["runs"][m]["conf"], 2))
            for m in STRONG
        }
        print("=" * 100)
        print(
            i,
            t["family"],
            t["task_type"],
            "p=%.3f" % a["p"],
            a["reasons"],
            "GOLD:",
            t["expected"],
            "chars",
            len(s),
        )
        print("STRONG:", json.dumps(strong))
        print("Q:", q.get("instructions"))
        print("OPTS:", json.dumps(q.get("criteria"), ensure_ascii=False))
        lim = ARGS.get("max_chars", 6000)
        print(
            "STATE:", s if len(s) <= lim else s[: lim // 2] + " [...] " + s[-lim // 2 :]
        )
    sys.exit(0)

with open(PRIV + "/gold_audit_list.json", "w") as f:
    json.dump(audit, f, indent=1)
out = {
    "n_distinct_models": 66,
    "bins": [f"{lo:.2f}-{min(hi, 1):.2f}" for lo, hi in BINS],
    "difficulty_by_tier": dict(bins),
    "audit_list": {
        "total": len(audit),
        "by_reason": dict(
            collections.Counter("+".join(a["reasons"]) for a in audit.values())
        ),
        "by_tier_family": dict(
            collections.Counter(
                targets[i]["tier"] + ":" + targets[i]["family"] for i in audit
            )
        ),
        "cue_flagged": sum(flags[i] for i in audit),
    },
}
json.dump(out, sys.stdout, indent=1, sort_keys=True)
