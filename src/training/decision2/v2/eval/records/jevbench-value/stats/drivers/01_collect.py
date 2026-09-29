"""Collect public-231 per-item correctness for every scored run (node A + node-B copies).

Writes the per-item matrix to node A only (OUT/matrix.json, chmod 600); prints
aggregate per-run rows (no item ids, no predictions) as JSON on stdout.
Usage: PYTHONPATH=$S python3 - '{"out": "/data/dev2/private/eval/jevbench-value/stats"}' < 01_collect.py
"""

import hashlib, json, os, sys

args = json.loads(sys.argv[1])
OUT = args["out"]
ROOTS = ["/data/dev2/runs", os.path.join(OUT, "nodeB")]
GOLD = "/data/dev2/private/panels/gold/public231/targets.jsonl"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_scores():
    out = []
    for root in ROOTS:
        for dp, dn, fn in os.walk(root):
            if "sealed" in dp.split(os.sep):
                dn[:] = []
                continue
            if dp[len(root) :].count(os.sep) > 6:
                dn[:] = []
            if "public231.score.json" in fn and os.path.basename(dp) == "scores":
                out.append(os.path.dirname(dp))
    return sorted(out)


MLX_ALIAS = {
    "eval/m1-adopt/bosun": "eval/m2/mlx/x-bosun06",
    "eval/m1-adopt/decider2b": "eval/m2/mlx/x-decider2b",
    "eval/m1-adopt/decider4b": "eval/m2/mlx/x-decider4b",
    "eval/m1-adopt/jpt9b": "eval/m2/mlx/x-jpt9b",
    "eval/m1-adopt/kai1": "eval/m2/mlx/x-kai",
    "eval/m1-adopt/lux1": "eval/m2/mlx/x-lux1",
    "eval/m1-adopt/nox1": "eval/m2/mlx/x-nox1",
    "eval/m1-adopt/sol1": "eval/m2/mlx/x-sol1",
    "eval/m1/p1-gliner25": "eval/m2/mlx/x-gliner25",
    "eval/m1/p2-jpt08b": "eval/m2/mlx/x-jpt08b",
    "eval/m1/p3-bosun17b": "eval/m2/mlx/x-bosun17b",
    "eval/m1/p4-jpt4b": "eval/m2/mlx/x-jpt4b",
    "eval/m1/r3-lex": "eval/m2/mlx/x-lex",
    "eval/m1/r4-eos1": "eval/m2/mlx/x-eos1",
    "eval/m1-adopt/autojev27": "eval/m5/mlx-diag-27b/autojev27",
    "eval/m4/nodeB-kernel/eikos27b": "eval/m5/mlx-diag-27b/eikos27b",
    "eval/m4/nodeB-kernel/jebadiah27b": "eval/m5/mlx-diag-27b/jebadiah27b",
    "nodeB/M3-A-soup_formal": "27b/m3-f2/mlx-diag/M3-A-soup",
    "nodeB/M3-S-soup_formal": "27b/m3-f2/mlx-diag/M3-S-soup",
}


def mlx_for(run_key):
    cands = []
    if run_key in MLX_ALIAS:
        cands.append(MLX_ALIAS[run_key])
    cands += [run_key, run_key + "-mlx", run_key.replace("-nodeA", "") + "-mlx"]
    for c in cands:
        p = os.path.join("/data/dev2/runs", c, "mlx-diag.score.json")
        if os.path.isfile(p):
            d = json.load(open(p))
            return {
                "path": c,
                "type_macro": d.get("type_macro_accuracy"),
                "english_type_macro": d.get("english_type_macro_accuracy"),
                "non_english_type_macro": d.get("non_english_type_macro_accuracy"),
                "items": d.get("items"),
                "predictions_sha256": d.get("predictions_sha256"),
            }
    return None


gold = [json.loads(l) for l in open(GOLD)]
gold_ids = sorted(g.get("id") for g in gold)
matrix, rows = {}, []
for d in find_scores():
    key = (
        os.path.relpath(d, "/data/dev2/runs")
        if d.startswith("/data/dev2/runs")
        else "nodeB/" + os.path.relpath(d, ROOTS[1])
    )
    s = json.load(open(os.path.join(d, "scores/public231.score.json")))
    pp = os.path.join(d, "output/public231.predictions.jsonl")
    psha = sha(pp) if os.path.isfile(pp) else None
    per = {it["id"]: it for it in s["per_item"]}
    vec = (
        "".join("1" if per[i]["correct"] else "0" for i in gold_ids)
        if set(per) == set(gold_ids)
        else None
    )
    matrix[key] = {
        "correct": {i: int(bool(per[i]["correct"])) for i in per},
        "valid": {i: int(bool(per[i]["valid"])) for i in per},
        "confidence": {i: per[i].get("confidence") for i in per},
        "predicted": {i: per[i].get("predicted") for i in per},
    }
    tiers = {}
    for it in s["per_item"]:
        t = tiers.setdefault(it["tier"], [0, 0])
        t[0] += int(bool(it["correct"]))
        t[1] += 1
    rep = {}
    rp = os.path.join(d, "REPORT.json")
    if os.path.isfile(rp):
        r = json.load(open(rp))
        tf = r.get("panels", {}).get("typed-final", {})
        bt = tf.get("by_type", {})
        rep = {
            "schema": r.get("schema"),
            "reused": r.get("reused"),
            "model": r.get("model"),
            "params_loaded": (r.get("parameters") or {}).get("loaded"),
            "v3": (r.get("v3") or {}).get("score"),
            "T": (r.get("v3") or {}).get("T"),
            "H": (r.get("v3") or {}).get("H"),
            "choice": (bt.get("choice") or {}).get("accuracy"),
            "noul": (bt.get("noul") or {}).get("accuracy"),
            "score": (bt.get("score") or {}).get("accuracy"),
            "report_public_correct": r.get("panels", {})
            .get("public231", {})
            .get("correct"),
            "report_public_score_sha256": r.get("panels", {})
            .get("public231", {})
            .get("score_sha256"),
            "public_long": (r.get("slices", {}).get("public231") or {}).get("long"),
            "css15_H": r.get("panels", {}).get("css15", {}).get("H"),
        }
    rows.append(
        {
            "run": key,
            "items": s["items"],
            "correct": s["correct"],
            "valid": s["valid"],
            "strict_valid": s["strict_valid"],
            "invalid": s["items"] - s["valid"],
            "renormalized": s.get("renormalized"),
            "point_argmax_disagreements": s.get("point_argmax_disagreements"),
            "brier_valid": s.get("brier_valid"),
            "ece_pmax_15": s.get("ece_pmax_15"),
            "tier_macro_accuracy": s.get("tier_macro_accuracy"),
            "tiers": {k: v[0] for k, v in tiers.items()},
            "tier_n": {k: v[1] for k, v in tiers.items()},
            "targets_sha256": s.get("targets_sha256"),
            "prompts_sha256": s.get("prompts_sha256"),
            "predictions_sha256_declared": s.get("predictions_sha256"),
            "predictions_sha256_file": psha,
            "predictions_sha_ok": psha is not None
            and psha == s.get("predictions_sha256"),
            "vector_sha256": hashlib.sha256(vec.encode()).hexdigest() if vec else None,
            "report": rep,
            "mlx": mlx_for(key),
        }
    )

mp = os.path.join(OUT, "matrix.json")
with open(mp, "w") as f:
    json.dump(
        {
            "item_ids": gold_ids,
            "gold": {
                g["id"]: {
                    "tier": g.get("tier"),
                    "family": g.get("family"),
                    "task_type": g.get("task_type"),
                }
                for g in gold
            },
            "runs": matrix,
        },
        f,
    )
os.chmod(mp, 0o600)
print(json.dumps({"n_runs": len(rows), "gold_items": len(gold_ids), "rows": rows}))
