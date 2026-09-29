"""Full-panel paired item intervals with the unchanged v2.eval.overlap_effects functions.

public 231: items resampled within tier (strata_bootstrap), delta in items.
CSS15 per task: items resampled within the task (task_bootstrap), delta in macro-F1.
5,000 draws, seed 20260927 (PAIRED_REPLICATES / PAIRED_SEED), no items excluded: the
overlap tool's "full" variant. Output: aggregates only (no item ids, text or answers).

    PYTHONPATH=<mirror> python3 - '<json config>' < item_cis.py
"""

import json
import multiprocessing
import sys
from pathlib import Path

from transfer.score import read_jsonl as read_css_jsonl
from v2.eval import overlap_effects as oe
from v2.eval import panels
from v2.eval.gates import verified
from v2.eval.same_panel import PAIRED_REPLICATES, PAIRED_SEED, sha_file, write_json

cfg = json.loads(sys.argv[1])
root = Path(cfg.get("panel_root", str(panels.DEFAULT_ROOT)))
panels.verify(root, ["typed-final", "css15", "public231"])
gold_css = {
    i: row
    for i, row in read_css_jsonl(panels.path(root, "css15", "gold")).items()
    if row["role"] == "evaluation"
}
public_dir = root / panels.FORMAL["public231"]["panel_dir"]

models = {}
for name, run_dir in cfg["models"].items():
    run = Path(run_dir)
    css_path = verified(run, "css15")
    css = oe.css_outcomes(gold_css, read_css_jsonl(css_path))
    public = oe.public_outcomes(run, public_dir)
    report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    tasks = oe.css_task_f1(gold_css, css, set())
    models[name] = {
        "run": str(run),
        "css": css,
        "public": public,
        "tasks": tasks,
        "summary": {
            "run": str(run),
            "predictions_sha256": {
                "css15": sha_file(css_path),
                "public231": sha_file(verified(run, "public231")),
            },
            "public231": oe.public_value(public, set()),
            "checks": {
                "css15_tasks_equal_report": tasks
                == {
                    t: v["macro_f1"]
                    for t, v in report["panels"]["css15"]["tasks"].items()
                },
                "public231_equal_report": oe.public_value(public, set())["correct"]
                == report["panels"]["public231"]["correct"],
            },
        },
    }

jobs = []
pairs = [tuple(pair) for pair in cfg["pairs"]]
for left, right in pairs:
    key = oe.pair_key(left, right)
    lm, rm = models[left], models[right]
    strata = oe.public_strata(lm["public"], rm["public"], set())
    jobs.append(("public", key, "full", (strata, PAIRED_REPLICATES, PAIRED_SEED)))
    for task, (rows, nlabels) in sorted(
        oe.pair_tasks(gold_css, lm["css"], rm["css"], set()).items()
    ):
        if cfg.get("tasks") and task not in cfg["tasks"]:
            continue
        jobs.append(
            ("task", key, task, (rows, nlabels, PAIRED_REPLICATES, PAIRED_SEED))
        )

with multiprocessing.Pool(int(cfg.get("jobs", 8))) as pool:
    results = pool.map(oe._work, jobs)

out = {
    "schema": "dev2-f1-item-cis/1",
    "label": "post-key same-panel",
    "method": "v2.eval.overlap_effects full variant: public_strata + strata_bootstrap (items within tier); "
    "pair_tasks + task_bootstrap (items within task); 5,000 draws, seed 20260927; no exclusions",
    "models": {name: m["summary"] for name, m in models.items()},
    "pairs": {},
}
for left, right in pairs:
    key = oe.pair_key(left, right)
    lv, rv = out["models"][left]["public231"], out["models"][right]["public231"]
    out["pairs"][key] = {
        "public231": {
            "left": lv["correct"],
            "right": rv["correct"],
            "delta": lv["correct"] - rv["correct"],
            "tiers_delta": {
                tier: lv["tiers"][tier]["correct"] - rv["tiers"][tier]["correct"]
                for tier in lv["tiers"]
            },
        },
        "css15_tasks": {},
    }
for kind, key, variant, ci in results:
    left, right = key.split(" - ", 1)
    entry = out["pairs"][key]
    if kind == "public":
        entry["public231"]["ci95"] = ci
    else:
        lt, rt = models[left]["tasks"][variant], models[right]["tasks"][variant]
        entry["css15_tasks"][variant] = {
            "left": lt,
            "right": rt,
            "delta": lt - rt,
            "ci95": ci,
        }

if cfg.get("output"):
    write_json(Path(cfg["output"]), out)
    print("wrote", cfg["output"], sha_file(Path(cfg["output"])))
for key, entry in out["pairs"].items():
    pub = entry["public231"]
    ci = pub.get("ci95", {})
    print(
        f"{key}: public231 {pub['left']} vs {pub['right']} ({pub['delta']:+d} [{ci.get('low')}, {ci.get('high')}]) tiers {pub['tiers_delta']}"
    )
    for task, t in entry["css15_tasks"].items():
        flag = (
            "LOWER"
            if t["ci95"]["high"] < 0
            else ("HIGHER" if t["ci95"]["low"] > 0 else "")
        )
        print(
            f"   {task:24s} {t['left']:.3f} vs {t['right']:.3f} {t['delta']:+.3f} [{t['ci95']['low']:+.3f}, {t['ci95']['high']:+.3f}] {flag}"
        )
for name, m in out["models"].items():
    print(name, m["checks"], m["public231"]["correct"], m["public231"]["tiers"])
