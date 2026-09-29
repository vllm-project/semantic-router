"""Private item reader for the gap analysis (node A only; output is read, never committed).

Args JSON: {"pair": "<a>__vs__<b>", "side": "c"|"b", "families": [...]|null, "head": 700}
Prints, for the selected discordant items of a pair, the item kind, the question,
the options, a head of the state and both models' argmax. Used only to write
abstract skill descriptions; nothing from here is copied into the worktree.
"""

import json
import sys

ARGS = json.loads(sys.argv[1])
PRIV = "/data/dev2/private/eval/jevbench-value/gap"
GOLD = "/data/dev2/private/panels/gold/public231/targets.jsonl"
PROMPTS = "/data/dev2/private/panels/goldfree/public231.prompts.jsonl"


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


targets = {t["id"]: t for t in jl(GOLD)}
prompts = {p["id"]: p for p in jl(PROMPTS)}
mat = {r["id"]: r for r in jl(PRIV + "/per_item_matrix.jsonl")}
disc = json.load(open(PRIV + "/pair_discordants.json"))[ARGS["pair"]][
    ARGS.get("side", "c")
]
a, b = ARGS["pair"].split("__vs__")
fams = ARGS.get("families")
head = ARGS.get("head", 700)
tail = ARGS.get("tail", 0)
for i in disc:
    t = targets[i]
    if fams and t["family"] not in fams:
        continue
    p = prompts[i]
    q = p["questions"]["decision"]
    st = (
        p["state"]
        if isinstance(p["state"], str)
        else json.dumps(p["state"], ensure_ascii=False)
    )
    ra, rb = mat[i]["runs"][a], mat[i]["runs"][b]
    print("=" * 80)
    print(
        t["tier"],
        t["family"],
        t["task_type"],
        "chars",
        len(st),
        "gold",
        t["expected"],
        f"| {a}: {ra['pred']} {ra['conf']:.2f} | {b}: {rb['pred']} {rb['conf']:.2f}",
    )
    print("Q:", (q.get("instructions") or "")[:400])
    print("OPTS:", json.dumps(q.get("criteria"), ensure_ascii=False)[:600])
    print("STATE:", st[:head].replace("\n", " "))
    if tail:
        print("...STATE END:", st[-tail:].replace("\n", " "))
