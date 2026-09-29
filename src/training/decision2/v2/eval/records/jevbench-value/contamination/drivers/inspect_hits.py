"""Describe the fresh-screen hits without printing text or ids (CPU, stdlib + mirror).

Usage (stdin driver): ssh NODE "cd DIR && PYTHONPATH=$S python3 - '<json>'" < inspect_hits.py
  json: {"out": DIR, "labels": [...], "train": {label: path}, "flagged": PATH or null,
         "score": PATH to any public231.score.json (tier / family per item) or null}
For each quarantined group: methods, counts, hit / group rows, training source / family /
task_type, the public item's tier / family, and whether it is the rescreen-flagged item.
For boilerplate E-leaf units and supplemental exact-leaf items: leaf length in words, how
many public items and training groups share it, and the leaf's field (state / question).
Full details go to DIR/<label>.inspect.private.json (0600).
"""

import collections
import json
import os
import sys
from pathlib import Path

from v2.data.textnorm import normalize, text_leaves

P231 = "/data/dev2/private/panels/goldfree/public231.prompts.jsonl"


def main():
    args = json.loads(sys.argv[1])
    out = Path(args["out"])
    items = {
        json.loads(l)["id"]: json.loads(l)
        for l in open(P231, encoding="utf-8")
        if l.strip()
    }
    meta = {}
    if args.get("score"):
        for r in json.load(open(args["score"]))["per_item"]:
            meta[r["id"]] = {
                "tier": r["tier"],
                "family": r["family"],
                "type": r["type"],
            }
    flagged = set()
    if args.get("flagged"):
        f = json.load(open(args["flagged"]))
        stack = [f]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
            elif isinstance(x, str) and x in items:
                flagged.add(x)
    leaf_field = collections.defaultdict(set)
    leaf_items = collections.defaultdict(set)
    for iid, row in items.items():
        for field in ("state", "questions"):
            for leaf in text_leaves(row[field], decode_json=True):
                n = normalize(leaf)
                leaf_field[n].add(field)
                leaf_items[n].add(iid)
    report = {"flagged_public_items_found": len(flagged)}
    for label in args["labels"]:
        priv = json.load(
            open(out / f"screen-{label}" / f"{label}.overlap.private.json")
        )
        ngram = json.load(open(out / f"screen-{label}" / f"{label}.ngram.private.json"))
        groups = priv["groups"]
        want = set(groups)
        rows = collections.defaultdict(list)
        exact_rows = {r for v in ngram["exact_leaf"].values() for r in v}
        exact_meta = collections.Counter()
        train_leaves = set()
        with open(args["train"][label], encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("group_id") in want:
                    rows[row["group_id"]].append(
                        {
                            k: row.get(k)
                            for k in (
                                "source",
                                "family",
                                "task_type",
                                "language",
                                "split",
                            )
                        }
                    )
                if str(row.get("id")) in exact_rows:
                    exact_meta[(row.get("source"), row.get("task_type"))] += 1
                    for field in ("state", "instructions", "options"):
                        for leaf in text_leaves(row.get(field), decode_json=True):
                            n = normalize(leaf)
                            if n in leaf_items:
                                train_leaves.add(n)
        desc = []
        for gid, rec in groups.items():
            ids = sorted({i for roles in rec["protected_ids"].values() for i in roles})
            desc.append(
                {
                    "methods": rec["methods"],
                    "counts": {k: v for k, v in rec["counts"].items() if v},
                    "rows": rec["rows"],
                    "hit_rows": rec["hit_rows"],
                    "train_rows": dict(
                        collections.Counter(
                            json.dumps(r, sort_keys=True) for r in rows[gid]
                        )
                    ),
                    "public_items": [
                        dict(meta.get(i, {}), flagged=i in flagged) for i in ids
                    ],
                }
            )
        bp = []
        for unit in priv.get("boilerplate_units", []):
            ids = [x[1] for x in unit["example_protected_ids"]]
            bp.append(
                {
                    "kind": unit["kind"],
                    "candidate_groups": unit["candidate_groups"],
                    "protected_rows": unit["protected_rows"],
                    "tiers": dict(
                        collections.Counter(meta.get(i, {}).get("tier") for i in ids)
                    ),
                }
            )
        ex = []
        for iid, rws in ngram["exact_leaf"].items():
            fields = set()
            nwords = []
            for leaf in text_leaves(
                items[iid]["state"], decode_json=True
            ) + text_leaves(items[iid]["questions"], decode_json=True):
                n = normalize(leaf)
                if n in train_leaves:
                    fields |= leaf_field[n]
                    nwords.append(len(n.split()))
            ex.append(
                {
                    "tier": meta.get(iid, {}).get("tier"),
                    "flagged": iid in flagged,
                    "training_rows": len(rws),
                    "fields": sorted(fields),
                    "shortest_shared_leaf_words": min(nwords) if nwords else None,
                }
            )
        report[label] = {
            "quarantine_groups": desc,
            "boilerplate_units": bp,
            "exact_leaf_items": ex,
            "exact_leaf_training_sources": {
                f"{s}|{t}": c for (s, t), c in exact_meta.items()
            },
        }
        p = out / f"{label}.inspect.private.json"
        p.write_text(
            json.dumps(
                {"groups": {g: groups[g] for g in groups}, "exact": ngram["exact_leaf"]}
            )
        )
        os.chmod(p, 0o600)
    print(json.dumps(report, indent=1))


main()
