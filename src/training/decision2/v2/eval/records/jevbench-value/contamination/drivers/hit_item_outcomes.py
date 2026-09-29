"""Per-model outcome on the screen-hit public-231 items, keyed by an id hash (CPU, stdlib).

Mode "hash" (node holding the private receipt): prints sha256(id)[:12] of each quarantined
group's public items.  Mode "outcomes" (node A): for the given hashes, prints which models
from the item-signature args answered the item correctly and with what confidence.
Usage: ssh NODE "cd /tmp && python3 - '<json>'" < hit_item_outcomes.py
  {"mode": "hash", "receipts": [PATH, ...]}
  {"mode": "outcomes", "hashes": [...], "models": [{"name", "dir"}]}
"""

import hashlib
import json
import sys
from pathlib import Path


def h(x):
    return hashlib.sha256(x.encode()).hexdigest()[:12]


args = json.loads(sys.argv[1])
if args["mode"] == "hash":
    out = {}
    for path in args["receipts"]:
        priv = json.load(open(path))
        out[Path(path).name] = sorted(
            {
                h(i)
                for rec in priv["groups"].values()
                for ids in rec["protected_ids"].values()
                for i in ids
            }
        )
    print(json.dumps(out))
else:
    res = {}
    for m in args["models"]:
        d = json.load(open(Path(m["dir"]) / "scores" / "public231.score.json"))
        for r in d["per_item"]:
            if h(r["id"]) in args["hashes"]:
                res.setdefault(h(r["id"]), {})[m["name"]] = [
                    bool(r.get("valid", True) and r["correct"]),
                    round(r.get("confidence") or 0, 3),
                ]
    print(json.dumps(res, indent=1))
