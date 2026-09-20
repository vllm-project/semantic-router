"""Scores keep planned denominators and unknown billing visible."""

from __future__ import annotations

import math
import random
from collections import Counter
from datetime import datetime
from fractions import Fraction
from statistics import mean

from . import VERSION
from .accounting import cache_neutral_cost, correction_metadata, effective_calls
from .contracts import BENCHMARK_WEIGHTS, planned_cells
from .failures import first_saved_failure
from .native_output import model_limits
from .target_contracts import effective_auxiliary_targets, target_inventory

BUCKETS = ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens")
COMPARABLE_STATUSES = frozenset({"completed", "failed"})


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


def paired_conservative_interval(delta, groups, weights, alpha=0.05):
    """Hoeffding bound for independent paired differences in [-1, 1]."""
    squared_weights = sum(weights[b] ** 2 / len(values) for b, values in groups.items())
    radius = math.sqrt(2 * math.log(2 / alpha) * squared_weights)
    return [max(-1, delta - radius), min(1, delta + radius)]


def sum_cost(calls):
    if not calls or any(c.get("cost_usd") is None for c in calls):
        return None
    return sum(c["cost_usd"] for c in calls)


def output_diagnostics(results, subject_calls, total):
    """Count saved checks and call endings, without inferring absent output text."""
    details = [row.get("details") or {} for row in results]
    format_checks = [
        row["strict_format"]
        for row in details
        if isinstance(row.get("strict_format"), bool)
    ]
    finish_reasons = Counter(
        call["finish_reason"]
        for call in subject_calls
        if isinstance(call.get("finish_reason"), str) and call["finish_reason"]
    )
    return {
        "planned_cases": total,
        "result_cases": len(results),
        # This is a count of recorded flags, not proof that all other outputs
        # were complete. Missing result/check fields remain unassessed below.
        "output_limit_cases": sum(
            row.get("quality_failure") == "output_limit" for row in details
        ),
        "strict_format": {
            "checked_cases": len(format_checks),
            "failed_cases": format_checks.count(False),
            "unassessed_cases": total - len(format_checks),
        },
        "subject_calls": {
            "total": len(subject_calls),
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "unknown_finish_reason": len(subject_calls) - sum(finish_reasons.values()),
        },
    }


def metric(target_id, results, calls, total):
    completed = [r for r in results if r["status"] == "completed"]
    scored = [r for r in completed if isinstance(r.get("correct"), bool)]
    correct = sum(r["correct"] for r in scored)
    subject = [c for c in calls if c["role"] == "subject"]
    overhead = [c for c in calls if c["role"] != "subject"]
    receipted_cases = {c["case_id"] for c in subject if c.get("case_id")}
    subject_coverage_complete = len(receipted_cases) == total and all(
        r.get("case_id") in receipted_cases for r in results
    )
    usage_complete = (
        bool(subject)
        and subject_coverage_complete
        and all(c.get("usage") is not None for c in subject)
    )
    cost_complete = (
        bool(subject)
        and subject_coverage_complete
        and all(c.get("cost_usd") is not None for c in subject)
    )
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
        "output_diagnostics": output_diagnostics(results, subject, total),
        "accuracy_ci95": wilson(correct, total),
        "cost_usd": sum_cost(subject) if cost_complete else None,
        "cache_neutral_cost_usd": (
            sum(c["cache_neutral_cost_usd"] for c in subject)
            if subject
            and subject_coverage_complete
            and all(c.get("cache_neutral_cost_usd") is not None for c in subject)
            else None
        ),
        "cache_neutral_cost_basis": "Counterfactual token-equivalent subject cost: all prompt tokens at the frozen fresh-input rate plus output; not billed spend.",
        "known_cost_usd": known,
        "cost_complete": cost_complete,
        "evaluation_cost_usd": sum_cost(overhead) if overhead else 0,
        "total_spend_usd": sum_cost(calls) if subject_coverage_complete else None,
        "tokens": tokens,
        "call_count": (
            sum(c["inference_call_count"] for c in subject)
            if subject
            and subject_coverage_complete
            and all(c.get("inference_call_count") is not None for c in subject)
            else None
        ),
        "request_count": len(subject),
        "selected_models": dict(
            Counter(
                c.get("selected_model") or c.get("model")
                for c in subject
                if c.get("selected_model")
                or (c.get("status") == "completed" and c.get("model"))
            )
        ),
        "pending_selection_count": sum(
            c.get("status") in {"sent", "running"} and not c.get("selected_model")
            for c in subject
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
    calls = effective_calls(store, run_id, summary=True)
    calls = [
        {**call, "cache_neutral_cost_usd": cache_neutral_cost(manifest, call)}
        for call in calls
    ]
    metrics = []
    benchmarks = []
    cells = planned_cells(manifest)
    for target in manifest["targets"]:
        selected_ids = {
            cell["case_id"] for cell in cells if cell["target_id"] == target["id"]
        }
        rows = [r for r in results if r["target_id"] == target["id"]]
        tcalls = [c for c in calls if c["target_id"] == target["id"]]
        metrics.append(metric(target["id"], rows, tcalls, len(selected_ids)))
        for benchmark in sorted({c["benchmark"] for c in manifest["cases"]}):
            ids = {
                c["id"]
                for c in manifest["cases"]
                if c["benchmark"] == benchmark and c["id"] in selected_ids
            }
            if not ids:
                continue
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
    selected_ids = {cell["case_id"] for cell in cells}
    present = {c["benchmark"] for c in manifest["cases"] if c["id"] in selected_ids}
    weights = manifest["benchmark_weights"]
    custom_subset = bool(
        "execution_cells" in manifest
        or (manifest.get("dataset") or {}).get("custom_subset", True)
    )
    extension_subset = not present.issubset(BENCHMARK_WEIGHTS)
    for item in metrics:
        rows = [row for row in benchmarks if row["target_id"] == item["id"]]
        target_full_suite = {row["benchmark"] for row in rows} == set(BENCHMARK_WEIGHTS)
        weight_sum = sum(weights[row["benchmark"]] for row in rows)
        macro = (
            sum(weights[row["benchmark"]] * row["accuracy"] for row in rows)
            / weight_sum
        )
        item["macro_accuracy"] = macro
        item["sr_bench_score"] = (
            macro
            if target_full_suite and item["complete"] and not manifest.get("recovery")
            else None
        )
        item["score_scope"] = (
            (
                "full sr-bench 1.0"
                if target_full_suite
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
        for item in metrics + benchmarks:
            case_ids = {
                case["id"]
                for case in manifest["cases"]
                if not item.get("benchmark") or case["benchmark"] == item["benchmark"]
            }
            routing = [
                row.get("details", {}).get("routing", {})
                for row in routing_rows
                if row["target_id"] == item["id"] and row["case_id"] in case_ids
            ]
            for source, output in (
                ("selection_status", "selection_statuses"),
                ("selection_reason", "selection_reasons"),
            ):
                item[output] = dict(
                    Counter(row[source] for row in routing if row.get(source))
                )
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
                "cache_neutral_cost_usd",
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
    if manifest.get("recovery"):
        limitations.append(
            "This is a separate recovery subset attempt. Parent measurements and spend are inherited context only; child completion does not complete the original benchmark."
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
    accounting = correction_metadata(store, run_id)
    if accounting:
        limitations.append(
            "Accounting was reconciled offline from saved stream evidence and frozen prices. Original call/result detail receipts are unchanged; report totals use the versioned correction."
        )
    runner = store.provenance(run_id)
    if runner is None:
        limitations.append(
            "Runner source and dependency provenance was not captured for this saved run; it is unknown."
        )
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
        "failure": store.first_failure(run_id) or first_saved_failure(results),
        "recovery": manifest.get("recovery"),
        "child_attempts": [
            {
                "id": child["id"],
                "status": child["status"],
                "progress": child["progress"],
            }
            for child in store.children(run_id)
        ],
        "summary": {
            "targets": metrics,
            "wall_time_s": wall,
            "total_spend_usd": (
                0
                if manifest["mode"] == "replay"
                else (
                    sum_cost(calls)
                    if all(item["total_spend_usd"] is not None for item in metrics)
                    else None
                )
            ),
        },
        "benchmarks": benchmarks,
        "limitations": limitations,
        "provenance": {
            "accounting_correction": accounting,
            "runner": runner,
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


def _comparison_protocol(baseline, candidate):
    if baseline.get("output_policy", "bounded") != candidate.get(
        "output_policy", "bounded"
    ):
        raise ValueError("Cannot compare different output_policy")
    if baseline.get("output_policy", "bounded") == "native" and model_limits(
        baseline
    ) != model_limits(candidate):
        raise ValueError("Cannot compare changed frozen native model limits")
    for key in (
        "case_sha256",
        "sampling",
        "benchmark_options",
        "benchmark_weights",
        "adapter_versions",
        "mode",
        "limits",
    ):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Cannot compare different {key}")

    if effective_auxiliary_targets(baseline) != effective_auxiliary_targets(candidate):
        raise ValueError("Cannot compare changed effective judge/simulator targets")

    prices, profiles = {}, {}
    for manifest in (baseline, candidate):
        for target in target_inventory(manifest).values():
            for model, price in target.get("prices", {}).items():
                if model in prices and prices[model] != price:
                    raise ValueError("Cannot compare changed frozen model prices")
                prices[model] = price
            if target["kind"] == "single":
                model = target["model"]
                profile = {**manifest["sampling"], **target.get("request_params", {})}
                if model in profiles and profiles[model] != profile:
                    raise ValueError(
                        "Cannot compare changed single-model request parameters"
                    )
                profiles[model] = profile


def _exact_quality(report, target, weights):
    rows = [row for row in report["benchmarks"] if row["target_id"] == target]
    weighted = [
        (
            Fraction(str(weights[row["benchmark"]])),
            Fraction(row["correct"], row["total"]),
        )
        for row in rows
    ]
    return sum(w * score for w, score in weighted) / sum(w for w, _ in weighted)


def _strongest_single(singles, report, weights):
    """Compare frozen weighted counts exactly; never favor a costly quality tie."""
    metrics = {row["id"]: row for row in report["summary"]["targets"]}
    quality = {
        target["id"]: _exact_quality(report, target["id"], weights)
        for target in singles
    }
    maximum = max(quality.values())
    tied = sorted(target for target, score in quality.items() if score == maximum)

    def priced(target):
        item = metrics[target]
        return item["cost_complete"] and item["cost_usd"] is not None

    best = min(
        tied,
        key=lambda target: (
            not priced(target),
            metrics[target]["cost_usd"] if priced(target) else math.inf,
            target,
        ),
    )
    return best, tied, all(priced(target) for target in tied)


class ComparisonValidator:
    """Shared protocol and explicit terminal outcomes for discovery and submit."""

    def __init__(self, store, baseline):
        self.store = store
        self.baseline = baseline
        bm = baseline["manifest"]
        if baseline["status"] not in COMPARABLE_STATUSES or bm["mode"] != "live":
            raise ValueError("Baseline must be a completed or failed live run")
        self.singles = [t for t in bm["targets"] if t["kind"] == "single"]
        if not self.singles:
            raise ValueError("Baseline must contain a single-model target")
        planned = planned_cells(bm)
        selected = {
            target["id"]: {
                c["case_id"] for c in planned if c["target_id"] == target["id"]
            }
            for target in bm["targets"]
        }
        self.expected = selected[self.singles[0]["id"]]
        if not self.expected or any(ids != self.expected for ids in selected.values()):
            raise ValueError(
                "Baseline targets must cover identical nonempty selected cases"
            )
        self.case_ids = [c["id"] for c in bm["cases"] if c["id"] in self.expected]
        results = store.results(baseline["id"])
        self.by_target = self._quality_rows(results, bm["targets"], "Baseline")

    def _quality_rows(self, results, targets, label):
        by_target = {target["id"]: {} for target in targets}
        for result in results:
            rows = by_target.get(result["target_id"])
            if rows is None:
                continue
            case_id = result["case_id"]
            status = result["status"]
            if (
                case_id in rows
                or case_id not in self.expected
                or status not in COMPARABLE_STATUSES
                or (
                    status == "completed"
                    and not isinstance(result.get("correct"), bool)
                )
            ):
                raise ValueError(
                    f"{label} quality results lack unique terminal outcomes"
                )
            # A failed outcome is incorrect under the existing report denominator.
            # This derived score never changes its saved status, grade or receipt.
            rows[case_id] = {
                **result,
                "correct": result["correct"] if status == "completed" else False,
            }
        if any(set(rows) != self.expected for rows in by_target.values()):
            raise ValueError(f"{label} quality results are incomplete")
        return by_target

    def validate(self, candidate):
        if self.baseline["id"] == candidate["id"]:
            raise ValueError("Choose two distinct runs for comparison")
        _comparison_protocol(self.baseline["manifest"], candidate["manifest"])
        if candidate["status"] not in COMPARABLE_STATUSES:
            raise ValueError("Both runs must be completed or failed before comparison")
        planned = planned_cells(candidate["manifest"])
        if any(
            {cell["case_id"] for cell in planned if cell["target_id"] == target["id"]}
            != self.expected
            for target in candidate["manifest"]["targets"]
        ):
            raise ValueError("Candidate targets must cover identical planned cases")
        return self._quality_rows(
            self.store.results(candidate["id"]),
            candidate["manifest"]["targets"],
            "Candidate",
        )


def compare(store, baseline_id, candidate_id):
    baseline = store.get(baseline_id)
    candidate = store.get(candidate_id)
    validator = ComparisonValidator(store, baseline)
    candidate_rows = validator.validate(candidate)
    bm, cm = baseline["manifest"], candidate["manifest"]
    singles, expected = validator.singles, validator.expected
    case_ids, by_target = validator.case_ids, validator.by_target
    report_b = make_report(store, baseline_id)
    report_c = make_report(store, candidate_id)
    metric_by_id = {t["id"]: t for t in report_b["summary"]["targets"]}
    best, tied, tied_costs_complete = _strongest_single(
        singles, report_b, bm["benchmark_weights"]
    )
    base = by_target[best]
    base_metric = metric_by_id[best]
    comparisons = []
    for target in cm["targets"]:
        rows = candidate_rows[target["id"]]
        diffs = [int(rows[i]["correct"]) - int(base[i]["correct"]) for i in case_ids]
        groups = {
            b: [
                int(rows[c["id"]]["correct"]) - int(base[c["id"]]["correct"])
                for c in bm["cases"]
                if c["benchmark"] == b and c["id"] in expected
            ]
            for b in sorted(
                {c["benchmark"] for c in bm["cases"] if c["id"] in expected}
            )
        }
        weights = {
            b: bm["benchmark_weights"][b]
            / sum(bm["benchmark_weights"][k] for k in groups)
            for b in groups
        }
        macro_delta = float(
            _exact_quality(report_c, target["id"], cm["benchmark_weights"])
            - _exact_quality(report_b, best, bm["benchmark_weights"])
        )
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
        neutral_base = base_metric["cache_neutral_cost_usd"]
        neutral_candidate = cand_metric["cache_neutral_cost_usd"]
        savings = (
            (1 - cc / bc) * 100
            if bc is not None
            and bc > 0
            and cc is not None
            and tied_costs_complete
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
                "quality_delta_ci95": paired_conservative_interval(
                    macro_delta, groups, weights
                ),
                "quality_delta_ci95_method": "weighted-paired-hoeffding",
                "quality_delta_ci95_qualification": "Case-independent bounded-difference interval with frozen benchmark weights; excludes strongest-baseline-selection, source contamination and tuning-selection uncertainty.",
                "quality_delta_bootstrap_ci95": [samples[49], samples[1949]],
                "cost_saving_percent": savings,
                "baseline_cost_usd": bc,
                "candidate_cost_usd": cc,
                "cache_neutral_baseline_cost_usd": neutral_base,
                "cache_neutral_candidate_cost_usd": neutral_candidate,
                "cache_neutral_cost_saving_percent": (
                    (1 - neutral_candidate / neutral_base) * 100
                    if neutral_base is not None
                    and neutral_base > 0
                    and neutral_candidate is not None
                    and tied_costs_complete
                    and bm["cost_policy"] == cm["cost_policy"] == "require_priced"
                    else None
                ),
                "cache_neutral_cost_basis": "Counterfactual fresh-input token-equivalent cost against the same selected baseline; excludes cache-read/write discounts and premiums, not measured billing.",
                "wins": sum(d > 0 for d in diffs),
                "losses": sum(d < 0 for d in diffs),
                "ties": sum(d == 0 for d in diffs),
            }
        )
    return {
        "version": VERSION,
        "baseline_run_id": baseline_id,
        "candidate_run_id": candidate_id,
        "baseline_status": baseline["status"],
        "candidate_status": candidate["status"],
        "baseline_quality_complete": all(
            t["complete"] for t in report_b["summary"]["targets"]
        ),
        "candidate_quality_complete": all(
            t["complete"] for t in report_c["summary"]["targets"]
        ),
        "baseline_targets": report_b["summary"]["targets"],
        "candidate_targets": report_c["summary"]["targets"],
        "quality_denominator": "all_planned_cases; explicit failed outcomes count as incorrect",
        "baseline_selection": "best observed single model on identical planned cases; explicit failures count as incorrect; selection uncertainty is not included",
        "baseline_selection_qualification": "This ranks delivered outcomes under the frozen limits, not model capability without execution failures. All single-model targets remain in the ranking, including failed outcomes and unknown costs.",
        "baseline_selected_target_id": best,
        "baseline_tied_best_target_ids": tied,
        "baseline_tie_policy": "Exact frozen weighted quality; ties prefer the lowest complete known subject cost, then stable target ID. Unknown-cost ties rank after known costs and suppress savings claims.",
        "baseline_cost_comparison_eligible": tied_costs_complete,
        "baseline_cost_comparison_reason": (
            None
            if tied_costs_complete
            else "At least one quality-tied best single has incomplete cost; the cheapest strongest baseline cannot be established."
        ),
        "comparisons": comparisons,
    }
