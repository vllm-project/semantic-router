"""Local step: split the aggregate stdout of 01_collect.py / 02_analyze.py into the committed
aggregate files (reliability.json, validity.json, runs-index.json). No per-item data is involved.
Usage: python3 03_write_records.py <collect.json> <analyze.json> <stats dir>
"""

import json, os, sys

collect, an, out = (
    json.load(open(sys.argv[1])),
    json.load(open(sys.argv[2])),
    sys.argv[3],
)

canon = {m["run"]: m["key"] for m in an["models"]}
ALIAS = {  # non-canonical run -> distinct model (same weights; context/renderer/node variants or derived copies)
    "06b/m2/formal/kai1-native-8k": (
        "kai1",
        "context 8K variant (canonical runs at the published 1K limit)",
    ),
    "eval/m1/r2-kai1-repeat": ("kai1", "repeat"),
    "06b/m6/formal/m6-control-released": ("dev2-0.6b", "M6 control re-run"),
    "release/dev2-0p6b-cal698-rescore": ("dev2-0.6b", "release rescore"),
    "release/dev2-0p8b-t1-derived": ("dev2-0.8b", "release T=1 derived"),
    "release/dev2-2b-t1-derived": ("dev2-2b", "release T=1 derived"),
    "release/dev2-4b-t1-derived": ("dev2-4b", "release T=1 derived"),
    "dec/formal/m5/m5-ref-N4XF-soup": ("dev2-4b", "node-B reference re-run"),
    "dec/formal/m5/dryrun/check-n4xf-vs-nox1": ("dev2-4b", "dry-run copy"),
    "release/dev2-8b-t1-derived": ("dev2-8b", "release T=1 derived"),
    "9b/formal-m3/lux1-16k-shared": ("lux1", "same-renderer 16K comparator"),
    "9b/formal/lux1-8k": ("lux1", "8K same-limit control"),
    "eval/m1-adopt/lux1": ("lux1", "node B r4"),
    "eval/m1/d2-lux1-frozen-cache": ("lux1", "frozen cache"),
    "eval/m1/r1-lux1-repeat": ("lux1", "repeat"),
    "eval/m4/nodeB-kernel/lux1": ("lux1", "node-B kernel image"),
    "dec/formal/m2/eos1-8k": ("eos1", "8K same-limit control"),
    "dec/formal/m2/eos1-16k": ("eos1", "16K same-limit control"),
    "dec/formal/m2/nox1-8k": ("nox1", "8K same-limit control"),
    "dec/formal/m3/nox1-16k": ("nox1", "16K same-limit control"),
    "dec/formal/m2/sol1-8k": ("sol1", "8K same-limit control"),
    "dec/formal/m3/sol1-16k": ("sol1", "16K same-limit control"),
    "eval/m1-adopt/autojev27": ("autojev27", "node A (pre-kernel)"),
    "eval/m2/n2-autojev27-nodeB": ("autojev27", "node B (pre-kernel)"),
    "nodeB/formal-peer-autojev27": ("autojev27", "27B-track peer run (pre-kernel)"),
    "nodeB/m2-peer-autojev27-nodeB": ("autojev27", "27B-track node B (pre-kernel)"),
    "nodeB/m2-peer-autojev27-nodeB-kernel": ("autojev27", "27B-track node-B kernel"),
    "eval/m2/q3-eikos27b-nodeB": ("eikos27b", "node B (pre-kernel)"),
    "nodeB/m3-peer-eikos27-nodeB-kernel": ("eikos27b", "27B-track node-B kernel"),
    "eval/m2/q7-jebadiah27b-nodeB": ("jebadiah27b", "node B (pre-kernel)"),
    "nodeB/m3-peer-jebadiah-nodeB-kernel": ("jebadiah27b", "27B-track node-B kernel"),
    "nodeB/m2-F0-formal": ("c27-C0", "C0 at 8K formal"),
}
mods = {m["key"]: m for m in an["models"]}
runs = []
for r in collect["rows"]:
    k = r["run"]
    if k in canon:
        key, role = canon[k], "canonical"
    else:
        key, role = ALIAS[k][0], ALIAS[k][1]
    m = mods[key]
    runs.append(
        {
            "run": (
                k.replace("nodeB/", "node B 27b/").replace("_", "/", 1)
                if k.startswith("nodeB/")
                else k
            ),
            "stored_on": "node B" if k.startswith("nodeB/") else "node A",
            "model": key,
            "model_label": m["label"],
            "role": role,
            "size_tier": m["size"],
            "lineage": m["lineage"],
            "card_model": m["card"],
            "public_total": r["correct"],
            "tiers": r["tiers"],
            "invalid": r["invalid"],
            "strict_valid": r["strict_valid"],
            "point_argmax_disagreements": r["point_argmax_disagreements"],
            "brier_valid": r["brier_valid"],
            "ece_pmax_15": r["ece_pmax_15"],
            "tier_macro_accuracy": r["tier_macro_accuracy"],
            "predictions_sha256": r["predictions_sha256_declared"],
            "predictions_sha_verified": r["predictions_sha_ok"],
            "correctness_vector_sha256": r["vector_sha256"],
            "v3": r["report"].get("v3"),
            "T": r["report"].get("T"),
            "H": r["report"].get("H"),
            "choice": r["report"].get("choice"),
            "noul": r["report"].get("noul"),
            "score": r["report"].get("score"),
            "params_loaded": r["report"].get("params_loaded"),
            "public_long_slice": r["report"].get("public_long"),
            "mlx_type_macro": (r["mlx"] or {}).get("type_macro"),
        }
    )
assert len(runs) == collect["n_runs"]
vg = {}
for r in runs:
    vg.setdefault(r["correctness_vector_sha256"], []).append(r["run"])
pg = {}
for r in runs:
    pg.setdefault(r["predictions_sha256"], []).append(r["run"])
index = {
    "schema": "jevbench-value-runs-index/1",
    "n_runs": len(runs),
    "n_distinct_models": len(mods),
    "all_predictions_sha_verified": all(r["predictions_sha_verified"] for r in runs),
    "identical_correctness_vector_groups": [g for g in vg.values() if len(g) > 1],
    "byte_identical_prediction_groups": [g for g in pg.values() if len(g) > 1],
    "models": [
        {k: v for k, v in m.items() if k not in ("resid",)}
        | {"resid_vs_v3": m.get("resid")}
        for m in an["models"]
    ],
    "runs": runs,
}
meta = {
    "seed": an["seed"],
    "bootstrap_draws": an["bootstrap_draws"],
    "split_half_splits": an["split_half_splits"],
    "n_distinct_models": an["n_distinct_models"],
    "sets": an["sets"],
}
json.dump(index, open(os.path.join(out, "runs-index.json"), "w"), indent=1)
json.dump(
    meta | {"reliability": an["reliability"]},
    open(os.path.join(out, "reliability.json"), "w"),
    indent=1,
)
json.dump(
    meta | {"validity": an["validity"], "beyond_v3": an["beyond_v3"], "c1": an["c1"]},
    open(os.path.join(out, "validity.json"), "w"),
    indent=1,
)
print("ok", len(runs), len(mods))
