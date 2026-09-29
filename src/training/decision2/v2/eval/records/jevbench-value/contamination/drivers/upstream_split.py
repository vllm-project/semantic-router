"""Upstream JevBench (public MIT repo @1bcc55eb): hard-public vs hard-held-out accuracy per system.

Usage: git -C /tmp/jevbench-upstream show 1bcc55eb:results/v1.2/jevbench-v1.2-per-task.json > pt.json
       python3 upstream_split.py pt.json > upstream_split.json
Held-out hard = by_tier hard total minus the public hard items marked correct (111 public / 109 held out).
Also read: results/v1.4.2/measurement-aggregates.json v14.tiers hard_public / hard_heldout, and
results/v1.4.2/jevbench-v1.4.2-results.json public_accuracy / sealed_accuracy / public_minus_sealed_gap_pp.
"""

import json, math, statistics, sys

d = json.load(open(sys.argv[1]))
tasks = {t["id"]: t for t in d["tasks"]}
out = []
for k, s in d["systems"].items():
    if s.get("partial"):
        pass
    pt = s["public_tasks"]
    bt = s["by_tier"]["hard"]
    ph = [v for i, v in pt.items() if tasks[i]["tier"] == "hard"]
    if len(ph) != 111:
        continue
    pc = sum(1 for v in ph if v[0] == "c")
    tot = bt["c"]
    nh = bt["c"] + bt["w"] + bt["f"] + bt["n"]
    if nh != 220:
        continue
    hc = tot - pc
    hp, hh = pc / 111, hc / 109
    se = math.sqrt(max(hp * (1 - hp), 1e-9) / 111 + max(hh * (1 - hh), 1e-9) / 109)
    out.append(
        dict(
            key=k,
            display=s["display"],
            hard_public=hp,
            hard_heldout=hh,
            diff=hp - hh,
            z=(hp - hh) / se,
            partial=s.get("partial"),
        )
    )
out.sort(key=lambda r: -r["diff"])
m = statistics.mean(r["diff"] for r in out)
sd = statistics.pstdev(r["diff"] for r in out)
print(json.dumps(dict(n=len(out), mean_diff=m, sd_diff=sd, systems=out), indent=0))
