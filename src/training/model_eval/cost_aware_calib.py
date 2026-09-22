#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cost-Aware Calibration Script
=============================

Cost-aware λ-scan calibration for multi-model routing. Given per-question
records from N models on the same task set, scans a grid of λ values and
reports task-weighted accuracy / cost / per-model share for each. A
recommendation block picks the λ with the best cost reduction while keeping
accuracy within a tolerance of the λ=0 baseline.

Score formula (per bucket b, per model m):
    score_m(b) = acc_m(b) − λ · norm_cost_m(b)
    norm_cost_m(b) = cost_m(b) / max_over_models(cost_m(b))   (in [0, 1])

λ semantics: how many units of normalized cost we are willing to pay for
one unit of accuracy. λ=0 is pure-accuracy routing; larger λ raises the
bar for more expensive models.

Optionally also produces per-category mean token costs (consumed by
result_to_config.py --token-costs-dir), folding the token_cost_summary
functionality into a single calibration pass.

Usage:
    # λ scan + recommendation + token costs side-output (2 models)
    python cost_aware_calib.py \\
        --model "qwen3.5-4b,results/qwen3.5-4b/calib_train/judged.jsonl" \\
        --model "dsv4-flash,results/dsv4-flash/calib_train/judged.jsonl" \\
        --out results/calib_cost_office.json \\
        --token-costs-dir results/token_costs

    # custom λ grid
    python cost_aware_calib.py ... --lambdas 0,0.05,0.1,0.15,0.2,0.3,0.4

    # only produce per-category token costs (no λ scan)
    python cost_aware_calib.py --token-costs-only \\
        --model "qwen3.5-4b,results/qwen3.5-4b/calib_train/judged.jsonl" \\
        --model "dsv4-flash,results/dsv4-flash/calib_train/judged.jsonl" \\
        --token-costs-dir results/token_costs

How to read the output
----------------------

Terminal output (stdout) is the primary readout. Focus on:

  1. Scan table (one per bucket×cost combination). Example:

       === bucket=dataset cost=tokens (B1=qwen3.5-4b=78.42% / B2=dsv4-flash=85.13%) ===
       lambda  acc%     cost/task   dsv4-fla  qwen3.5-
       0.00    85.13    1120.5      100.0%     0.0%
       0.05    84.21     820.3       78.5%    21.5%
       0.10    82.50     180.2       25.0%    75.0%   ← recommended
       0.20    78.42     145.8       10.0%    90.0%
       0.40    78.42     145.8        0.0%   100.0%

       - lambda: the λ value.
       - acc%: task-weighted accuracy at this λ.
       - cost/task: mean per-task cost (tokens or GPU·s).
       - per-model share: fraction of tasks routed to each model.

  2. RECOMMENDATION block at the end of stdout. Example:

       == RECOMMENDATION (dataset_tokens, tolerance=3.0pp) ==
         λ* = 0.1  (acc=82.5%, drop=2.63pp, cost↓83.7%)
         acc drops 2.63pp (within 3.0pp tolerance) while cost drops 83.7%
         Pass this to: result_to_config.py --cost-lambda 0.1

       - λ*: the recommended λ (the one with max cost reduction while
             keeping acc drop ≤ --acc-tolerance pp).
       - drop: accuracy drop vs the λ=0 baseline, in percentage points.
       - cost↓: cost reduction vs the λ=0 baseline, as a percentage.
       - The final line shows the exact command to pass λ to result_to_config.py.

The --out JSON file (when --out is given) contains the full scan tables
plus the recommendation block under the "recommendation.picked" key, for
programmatic consumption. Use it when you need to inspect per-bucket
breakdowns or when downstream scripts need the picked λ.

When --token-costs-dir is set, one JSON file per model is written under
that directory (e.g. {dir}/qwen3.5-4b.json). Each file maps category
names to mean completion_tokens. Pass that same directory to
result_to_config.py --token-costs-dir to enable cost-aware config
generation.
"""
import argparse
import json
import logging
import os
from collections import defaultdict
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)

DEFAULT_LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.4]

# Acceptable acc drop vs λ=0 baseline in percentage points.
ACC_DROP_TOLERANCE = 3.0


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def buckets_of(rows, key):
    g = {}
    for r in rows:
        g.setdefault(r[key], []).append(r)
    return g


def cost_of(r, basis):
    """Return per-record cost. basis='tokens' -> completion_tokens;
    basis='gpus' -> latency_s (caller can scale by TP if needed)."""
    if basis == "tokens":
        return r.get("completion_tokens")
    lat = r.get("latency_s")
    return lat if lat is not None else None


def scan_model_per_bucket(rows, key, basis):
    """Scan one model's records, return per-bucket {name: {n, acc, cost}}.

    Called once per model; the caller combines results across models.
    """
    groups = buckets_of(rows, key)
    out = {}
    for gname, g_rows in groups.items():
        acc = sum(1 for r in g_rows if r["is_correct"]) / len(g_rows)
        c = mean([cost_of(r, basis) for r in g_rows])
        out[gname] = {"n": len(g_rows), "acc": acc, "cost": c}
    return out


def scan(models_data, key, basis, lambdas):
    """Scan λ grid for one (key, basis) combination across all models.

    models_data: {model_name: list_of_rows}
    """
    # Per-bucket per-model stats: {bucket_name: {model_name: {n, acc, cost}}}
    per_bucket = defaultdict(dict)
    for model_name, rows in models_data.items():
        mb = scan_model_per_bucket(rows, key, basis)
        for bname, stats in mb.items():
            per_bucket[bname][model_name] = stats

    # Build aligned bucket list (only buckets present in all models)
    all_models = list(models_data.keys())
    buckets = []
    for bname in sorted(per_bucket):
        models_in_bucket = per_bucket[bname]
        if len(models_in_bucket) != len(all_models):
            continue  # bucket missing some models
        n = min(s["n"] for s in models_in_bucket.values())
        if n == 0:
            continue
        entry = {"name": bname, "n": n, "models": {}}
        for m in all_models:
            s = models_in_bucket[m]
            entry["models"][m] = {"acc": s["acc"], "cost": s["cost"]}
        buckets.append(entry)

    results = []
    for lam in lambdas:
        acc_hits = cost_sum = 0
        n_total = 0
        share = {m: 0 for m in all_models}
        for b in buckets:
            # Compute score per model, pick max
            costs = {m: b["models"][m]["cost"] or 0.0 for m in all_models}
            cmax = max(costs.values()) or 1.0
            scores = {
                m: b["models"][m]["acc"] - lam * (costs[m] / cmax) for m in all_models
            }
            winner = max(scores, key=lambda m: (scores[m], -costs[m]))
            share[winner] += b["n"]
            acc_hits += b["models"][winner]["acc"] * b["n"]
            cost_sum += (b["models"][winner]["cost"] or 0.0) * b["n"]
            n_total += b["n"]
        results.append(
            {
                "lambda": lam,
                "task_weighted_acc": pct(acc_hits, n_total),
                "mean_cost_per_task": round(cost_sum / n_total, 1) if n_total else None,
                "share": {m: pct(share[m], n_total) for m in all_models},
            }
        )
    # Baselines: each model's overall acc
    baselines = {}
    for m, rows in models_data.items():
        baselines[m] = pct(sum(1 for r in rows if r["is_correct"]), len(rows))
    return {"buckets": buckets, "lambda_scan": results, "endpoint_accs": baselines}


def recommend(scan_result, acc_tolerance=ACC_DROP_TOLERANCE):
    """Pick λ with best cost reduction while keeping acc within tolerance."""
    rows = scan_result["lambda_scan"]
    baseline = next((r for r in rows if r["lambda"] == 0.0), rows[0])
    baseline_acc = baseline["task_weighted_acc"]
    threshold = baseline_acc - acc_tolerance

    eligible = [r for r in rows if r["task_weighted_acc"] >= threshold]
    if not eligible:
        return {"lambda": 0.0, "reason": "no λ within tolerance — use λ=0 baseline"}

    min_cost = min(r["mean_cost_per_task"] for r in eligible if r["mean_cost_per_task"])
    candidates = [r for r in eligible if r["mean_cost_per_task"] == min_cost]
    best = max(candidates, key=lambda r: r["lambda"])

    return {
        "lambda": best["lambda"],
        "task_weighted_acc": best["task_weighted_acc"],
        "mean_cost_per_task": best["mean_cost_per_task"],
        "baseline_acc": baseline_acc,
        "acc_drop_pp": round(baseline_acc - best["task_weighted_acc"], 2),
        "cost_reduction_pct": (
            round(
                100.0
                * (baseline["mean_cost_per_task"] - best["mean_cost_per_task"])
                / baseline["mean_cost_per_task"],
                2,
            )
            if baseline["mean_cost_per_task"]
            else None
        ),
        "acc_tolerance_pp": acc_tolerance,
        "reason": (
            f"acc drops {baseline_acc - best['task_weighted_acc']:.2f}pp "
            f"(within {acc_tolerance}pp tolerance) "
            f"while cost drops "
            f"{100.0 * (baseline['mean_cost_per_task'] - best['mean_cost_per_task']) / baseline['mean_cost_per_task']:.1f}%"
        ),
    }


def summarize_token_costs(models_data):
    """Per-category mean completion_tokens per model.

    Returns dict: {model_name: {category: avg_tokens}}.
    """
    out = {}
    for model_name, rows in models_data.items():
        cats = defaultdict(list)
        for r in rows:
            cats[r["category"]].append(r.get("completion_tokens") or 0)
        out[model_name] = {c: sum(v) / len(v) for c, v in cats.items()}
    return out


def _encode_model_id(model_name):
    """Encode a model id into a flat, injective filename stem.

    Uses ``urllib.parse.quote`` with ``safe=''`` so every non-alphanumeric
    character (including ``/``) is percent-encoded. This is injective:
    ``org/model`` -> ``org%2Fmodel`` and the distinct served-model name
    ``org__slash__model`` -> ``org__slash__model`` (unchanged) map to
    different filenames, so two different model ids can never collide.

    Reversible with ``_decode_model_id``. Used by write_token_costs and
    load_token_costs (in result_to_config.py) so a qualified id like
    ``org/model`` round-trips through the token-cost file and is looked
    up under the same key at config-generation time.
    """
    return quote(model_name, safe="")


def _decode_model_id(filename_stem):
    """Reverse of ``_encode_model_id``."""
    return unquote(filename_stem)


def write_token_costs(token_costs, out_dir):
    """Write one JSON per model under out_dir (result_to_config.py format)."""
    os.makedirs(out_dir, exist_ok=True)
    for model, costs in token_costs.items():
        safe = _encode_model_id(model)
        path = os.path.join(out_dir, f"{safe}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(costs, f, indent=2)
        logger.info(f"{model}: {len(costs)} categories -> {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model entry: 'name,path/to/judged.jsonl' (repeatable)",
    )
    ap.add_argument(
        "--out", default="", help="Output JSON path for λ scan + recommendation"
    )
    ap.add_argument(
        "--lambdas",
        type=str,
        default="",
        help=f"Comma-separated λ grid (default: {DEFAULT_LAMBDAS})",
    )
    ap.add_argument(
        "--acc-tolerance",
        type=float,
        default=ACC_DROP_TOLERANCE,
        help=f"Acceptable acc drop vs λ=0 in pp " f"(default: {ACC_DROP_TOLERANCE})",
    )
    ap.add_argument(
        "--token-costs-dir",
        default="",
        help="If set, also write per-model per-category avg tokens "
        "to {dir}/{model}.json (consumed by result_to_config.py)",
    )
    ap.add_argument(
        "--token-costs-only",
        action="store_true",
        help="Only produce token costs JSONs, skip λ scan",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load all models
    models_data = {}
    for entry in args.model:
        if "," not in entry:
            raise ValueError(f"Bad --model entry (need 'name,path'): {entry}")
        name, path = entry.split(",", 1)
        models_data[name] = load(path)
        logger.info(f"loaded {name}: {len(models_data[name])} rows")

    # Always produce token costs (cheap) if requested
    if args.token_costs_dir:
        tc = summarize_token_costs(models_data)
        write_token_costs(tc, args.token_costs_dir)

    if args.token_costs_only:
        return

    lambdas = (
        [float(x) for x in args.lambdas.split(",")] if args.lambdas else DEFAULT_LAMBDAS
    )

    out = {}
    for key in ("dataset", "category"):
        for basis in ("tokens", "gpus"):
            out[f"{key}_{basis}"] = scan(models_data, key, basis, lambdas)

    primary = out["dataset_tokens"]
    out["recommendation"] = {
        "primary_scan": "dataset_tokens",
        "acc_tolerance_pp": args.acc_tolerance,
        "picked": recommend(primary, args.acc_tolerance),
        "all_scans": {
            name: recommend(s, args.acc_tolerance)["lambda"]
            for name, s in out.items()
            if name != "recommendation"
        },
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"λ scan saved -> {args.out}")

    # Print scan tables
    model_names = list(models_data.keys())
    for key in ("dataset", "category"):
        for basis in ("tokens", "gpus"):
            r = out[f"{key}_{basis}"]
            accs = " / ".join(f"{m}={r['endpoint_accs'][m]}%" for m in model_names)
            print(f"\n=== bucket={key} cost={basis} ({accs}) ===")
            share_hdr = " ".join(f"{m[:8]:>8s}" for m in model_names)
            print(f"{'lambda':>6s} {'acc%':>7s} {'cost/task':>10s} {share_hdr}")
            for row in r["lambda_scan"]:
                share_str = " ".join(
                    f"{row['share'].get(m, 0) or 0:>7.1f}%" for m in model_names
                )
                print(
                    f"{row['lambda']:>6.2f} {row['task_weighted_acc']:>7.2f} "
                    f"{row['mean_cost_per_task']:>10.1f} {share_str}"
                )

    rec = out["recommendation"]["picked"]
    print(f"\n== RECOMMENDATION (dataset_tokens, tolerance={args.acc_tolerance}pp) ==")
    print(
        f"  λ* = {rec['lambda']}  (acc={rec.get('task_weighted_acc')}%, "
        f"drop={rec.get('acc_drop_pp')}pp, "
        f"cost↓{rec.get('cost_reduction_pct')}%)"
    )
    print(f"  {rec['reason']}")
    print(f"  Pass this to: result_to_config.py --cost-lambda {rec['lambda']}")


if __name__ == "__main__":
    main()
