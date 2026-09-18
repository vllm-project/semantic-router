"""Scores keep planned denominators and unknown billing visible."""

from __future__ import annotations

import math
import random
from collections import Counter
from datetime import datetime
from statistics import mean

from . import VERSION
from .contracts import BENCHMARK_WEIGHTS

BUCKETS = ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens")


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0, center - half), min(1, center + half)]


def sum_cost(calls):
    if not calls or any(c.get("cost_usd") is None for c in calls):
        return None
    return sum(c["cost_usd"] for c in calls)


def metric(target_id, results, calls, total):
    completed = [r for r in results if r["status"] == "completed"]
    scored = [r for r in completed if isinstance(r.get("correct"), bool)]
    correct = sum(r["correct"] for r in scored)
    subject = [c for c in calls if c["role"] == "subject"]
    overhead = [c for c in calls if c["role"] != "subject"]
    usage_complete = bool(subject) and all(c.get("usage") is not None for c in subject)
    tokens = (
        {k: sum((c.get("usage") or {}).get(k, 0) for c in subject) for k in BUCKETS}
        if usage_complete
        else None
    )
    latencies = [
        r.get("subject_latency_s", r["latency_s"])
        for r in completed
        if r.get("latency_s") is not None
    ]
    known = sum(c.get("cost_usd") or 0 for c in subject)
    return {
        "id": target_id,
        "total": total,
        "completed": len(completed),
        "failed": sum(r["status"] not in {"completed", "running"} for r in results),
        "pending": total - len(results),
        "scored": len(scored),
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_denominator": "all_planned_cases; failures and unanswered count as incorrect",
        "complete": len(completed) == total and len(scored) == total,
        "accuracy_ci95": wilson(correct, total),
        "cost_usd": sum_cost(subject),
        "known_cost_usd": known,
        "cost_complete": bool(subject)
        and all(c.get("cost_usd") is not None for c in subject),
        "evaluation_cost_usd": sum_cost(overhead) if overhead else 0,
        "total_spend_usd": sum_cost(calls),
        "tokens": tokens,
        "call_count": (
            sum(c["inference_call_count"] for c in subject)
            if subject
            and all(c.get("inference_call_count") is not None for c in subject)
            else None
        ),
        "request_count": len(subject),
        "selected_models": dict(
            Counter(
                c.get("selected_model") or c.get("model")
                for c in subject
                if c.get("selected_model") or c.get("model")
            )
        ),
        "decisions": dict(Counter(c["decision"] for c in subject if c.get("decision"))),
        "queue_wait_p50_s": percentile(
            [r["queue_wait_s"] for r in results if r.get("queue_wait_s") is not None],
            0.5,
        ),
        "queue_wait_p95_s": percentile(
            [r["queue_wait_s"] for r in results if r.get("queue_wait_s") is not None],
            0.95,
        ),
        "evaluation_call_count": len(overhead),
        "case_wall_time_p50_s": percentile(
            [r["latency_s"] for r in completed if r.get("latency_s") is not None], 0.5
        ),
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p95_s": percentile(latencies, 0.95),
        "ttft_p50_s": percentile(
            [c["ttft_s"] for c in subject if c.get("ttft_s") is not None], 0.5
        ),
        "request_time_sum_s": sum(c.get("latency_s", 0) for c in subject),
    }


def make_report(store, run_id):
    run = store.get(run_id)
    manifest = run["manifest"]
    results = store.results(run_id)
    calls = store.calls(run_id, summary=True)
    metrics = []
    benchmarks = []
    for target in manifest["targets"]:
        rows = [r for r in results if r["target_id"] == target["id"]]
        tcalls = [c for c in calls if c["target_id"] == target["id"]]
        metrics.append(metric(target["id"], rows, tcalls, len(manifest["cases"])))
        for benchmark in sorted({c["benchmark"] for c in manifest["cases"]}):
            ids = {c["id"] for c in manifest["cases"] if c["benchmark"] == benchmark}
            benchmarks.append(
                {
                    "benchmark": benchmark,
                    "target_id": target["id"],
                    **metric(
                        target["id"],
                        [r for r in rows if r["case_id"] in ids],
                        [c for c in tcalls if c["case_id"] in ids],
                        len(ids),
                    ),
                }
            )
    present = {c["benchmark"] for c in manifest["cases"]}
    full_suite = present == set(BENCHMARK_WEIGHTS)
    weights = manifest["benchmark_weights"]
    custom_subset = bool((manifest.get("dataset") or {}).get("custom_subset", True))
    extension_subset = not present.issubset(BENCHMARK_WEIGHTS)
    for item in metrics:
        rows = [row for row in benchmarks if row["target_id"] == item["id"]]
        weight_sum = sum(weights[row["benchmark"]] for row in rows)
        macro = (
            sum(weights[row["benchmark"]] * row["accuracy"] for row in rows)
            / weight_sum
        )
        item["macro_accuracy"] = macro
        item["sr_bench_score"] = macro if full_suite and item["complete"] else None
        item["score_scope"] = (
            (
                "full sr-bench 1.0"
                if full_suite
                else (
                    "extension diagnostic; normalized adapter weights"
                    if extension_subset
                    else "selected benchmark subset; renormalized fixed weights"
                )
            )
            + f"; profile={manifest['profile']}"
            + ("; custom task subset" if custom_subset else "")
        )
    if manifest["mode"] in {"preview", "replay"}:
        routing_rows = (
            results
            if manifest["mode"] == "preview"
            else store.results(manifest["replay_sources"]["preview_run_id"])
        )
        for item in metrics:
            routing = [
                row.get("details", {}).get("routing", {})
                for row in routing_rows
                if row["target_id"] == item["id"]
            ]
            item["selected_models"] = dict(
                Counter(
                    row["selected_model"]
                    for row in routing
                    if row.get("selected_model")
                )
            )
            item["decisions"] = dict(
                Counter(
                    row["decision_result"]["decision_name"]
                    for row in routing
                    if (row.get("decision_result") or {}).get("decision_name")
                )
            )
    if manifest["mode"] == "preview":
        for item in metrics + benchmarks:
            for key in (
                "accuracy",
                "accuracy_ci95",
                "correct",
                "macro_accuracy",
                "sr_bench_score",
            ):
                if key in item:
                    item[key] = None
    if manifest["mode"] == "replay":
        for item in metrics + benchmarks:
            for key in (
                "accuracy",
                "macro_accuracy",
                "cost_usd",
                "latency_p50_s",
                "latency_p95_s",
                "ttft_p50_s",
            ):
                if key in item:
                    item["estimated_" + key] = item[key]
                    item[key] = None
            item["sr_bench_score"] = None
            item["total_spend_usd"] = 0
            item["cost_complete"] = False
    limitations = [
        "Scores apply only to this frozen sr-bench protocol and selected cases, not an upstream full benchmark score.",
        "Repeated tuning on development cases requires a separate untouched holdout.",
        "Self-hosted token-equivalent cost is not a measured hardware invoice.",
        "Wilson intervals describe case uncertainty; they do not account for source contamination or tuning selection.",
    ]
    if manifest["mode"] == "replay":
        limitations.append(
            "Offline single-selection replay reuses saved answers, token counts and latencies. It makes zero model requests and is not a formal live score or measured routing performance."
        )
    if manifest["mode"] == "preview":
        limitations.append(
            "Preview evaluates routing only; it does not measure answer quality."
        )
    if any(not m["cost_complete"] for m in metrics):
        limitations.append(
            "Some call costs are unknown; cost savings cannot be claimed for those targets."
        )
    if run["status"] != "completed":
        limitations.append(
            "Run is incomplete; planned denominators include missing cases."
        )
    if manifest["cost_policy"] == "capability_only":
        limitations.append(
            "Capability-only protocol: USD budget cannot bound missing or unpriced usage; wall time, output and call caps still apply."
        )
    limitations.append(
        "Cost reservations use conservative serialized-input bounds and frozen inference-call limits; provider billing outside reported token buckets is excluded."
    )
    metadata = manifest.get("limitations", [])
    limitations.extend(metadata)
    if any(c["benchmark"] == "gpqa-diamond" for c in manifest["cases"]):
        limitations.append(
            "GPQA labels were previously seen in this project; this is a retest, not an unseen holdout claim."
        )
    if manifest["cost_policy"] == "require_priced":
        limitations.append(
            "Dispatch reservations use frozen request estimates. Provider or router prompt/output transformations can exceed those estimates; max_cost_usd stops future dispatch from measured spend but is not a universal hard billing cap."
        )
    wall = (
        datetime.fromisoformat(run["updated_at"])
        - datetime.fromisoformat(run["created_at"])
    ).total_seconds()
    return {
        "version": VERSION,
        "run_id": run_id,
        "status": run["status"],
        "mode": manifest["mode"],
        "cost_policy": manifest["cost_policy"],
        "summary": {
            "targets": metrics,
            "wall_time_s": wall,
            "total_spend_usd": 0 if manifest["mode"] == "replay" else sum_cost(calls),
        },
        "benchmarks": benchmarks,
        "limitations": limitations,
        "provenance": {
            "plan_sha256": manifest["plan_sha256"],
            "case_sha256": manifest["case_sha256"],
            "dataset": manifest.get("dataset"),
            "sampling": manifest["sampling"],
            "adapter_versions": manifest.get("adapter_versions", {}),
            "custom_subset": custom_subset,
            "target_profiles": [
                {
                    "id": t["id"],
                    "model": t["model"],
                    "kind": t["kind"],
                    "request_params": t.get("request_params", {}),
                    "config_hash": t.get("config_hash"),
                }
                for t in manifest["targets"]
            ],
            "profile": manifest["profile"],
            "seed": manifest["seed"],
        },
    }


def compare(store, baseline_id, candidate_id):
    baseline = store.get(baseline_id)
    candidate = store.get(candidate_id)
    bm = baseline["manifest"]
    cm = candidate["manifest"]
    for key in (
        "case_sha256",
        "sampling",
        "benchmark_options",
        "auxiliary_targets",
        "benchmark_weights",
        "adapter_versions",
        "mode",
    ):
        if bm.get(key) != cm.get(key):
            raise ValueError(f"Cannot compare different {key}")
    if bm["mode"] != "live":
        raise ValueError("Preview cannot support quality or cost-saving comparisons")
    if baseline["status"] != "completed" or candidate["status"] != "completed":
        raise ValueError("Both runs must complete before a paired comparison")
    br = store.results(baseline_id)
    cr = store.results(candidate_id)
    baseline_profiles = {
        t["model"]: t.get("request_params", {})
        for t in bm["targets"]
        if t["kind"] == "single"
    }
    for target in cm["targets"]:
        if (
            target["kind"] == "single"
            and target["model"] in baseline_profiles
            and target.get("request_params", {}) != baseline_profiles[target["model"]]
        ):
            raise ValueError("Cannot compare changed single-model request parameters")
    singles = [t for t in bm["targets"] if t["kind"] == "single"]
    if not singles:
        raise ValueError("Baseline must contain a single-model target")
    case_ids = [c["id"] for c in bm["cases"]]
    by_target = {
        t["id"]: {r["case_id"]: r for r in br if r["target_id"] == t["id"]}
        for t in singles
    }
    if any(
        len(rows) != len(case_ids)
        or any(not isinstance(r.get("correct"), bool) for r in rows.values())
        for rows in by_target.values()
    ):
        raise ValueError("Baseline quality results are incomplete")
    report_b = make_report(store, baseline_id)
    report_c = make_report(store, candidate_id)
    metric_by_id = {t["id"]: t for t in report_b["summary"]["targets"]}
    best = max(singles, key=lambda t: metric_by_id[t["id"]]["macro_accuracy"])["id"]
    base = by_target[best]
    base_metric = next(t for t in report_b["summary"]["targets"] if t["id"] == best)
    comparisons = []
    for target in cm["targets"]:
        rows = {r["case_id"]: r for r in cr if r["target_id"] == target["id"]}
        if set(rows) != set(case_ids) or any(
            not isinstance(r.get("correct"), bool) for r in rows.values()
        ):
            raise ValueError("Candidate quality results are incomplete")
        diffs = [int(rows[i]["correct"]) - int(base[i]["correct"]) for i in case_ids]
        groups = {
            b: [
                int(rows[c["id"]]["correct"]) - int(base[c["id"]]["correct"])
                for c in bm["cases"]
                if c["benchmark"] == b
            ]
            for b in {c["benchmark"] for c in bm["cases"]}
        }
        weights = {
            b: bm["benchmark_weights"][b]
            / sum(bm["benchmark_weights"][k] for k in groups)
            for b in groups
        }
        macro_delta = sum(weights[b] * mean(values) for b, values in groups.items())
        rng = random.Random(20260918)
        samples = sorted(
            sum(
                weights[b] * mean(rng.choices(values, k=len(values)))
                for b, values in groups.items()
            )
            for _ in range(2000)
        )
        cand_metric = next(
            t for t in report_c["summary"]["targets"] if t["id"] == target["id"]
        )
        bc = base_metric["cost_usd"]
        cc = cand_metric["cost_usd"]
        savings = (
            (1 - cc / bc) * 100
            if bc is not None
            and bc > 0
            and cc is not None
            and bm["cost_policy"] == cm["cost_policy"] == "require_priced"
            else None
        )
        comparisons.append(
            {
                "baseline_target_id": best,
                "candidate_target_id": target["id"],
                "paired_cases": len(diffs),
                "quality_delta": macro_delta,
                "micro_quality_delta": mean(diffs),
                "quality_metric": "fixed-weight benchmark macro accuracy",
                "quality_delta_ci95": [samples[49], samples[1949]],
                "cost_saving_percent": savings,
                "baseline_cost_usd": bc,
                "candidate_cost_usd": cc,
                "wins": sum(d > 0 for d in diffs),
                "losses": sum(d < 0 for d in diffs),
                "ties": sum(d == 0 for d in diffs),
            }
        )
    return {
        "version": VERSION,
        "baseline_run_id": baseline_id,
        "candidate_run_id": candidate_id,
        "baseline_selection": "best observed single model on identical cases; selection uncertainty is not included",
        "comparisons": comparisons,
    }
