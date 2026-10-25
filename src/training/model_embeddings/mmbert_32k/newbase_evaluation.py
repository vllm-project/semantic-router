"""Fixed-support development metrics and prospective all-gates selection.

This module consumes complete, explicitly named candidate scores. It does not
discover datasets, silently remove failed queries or load a final split.
"""

from __future__ import annotations

import math
from collections import defaultdict


def ndcg(
    scores: dict[str, float],
    relevance: dict[str, float],
    k: int = 10,
    *,
    allow_unretrieved: bool = False,
) -> float:
    if (
        not scores
        or not relevance
        or (not allow_unretrieved and not relevance.keys() <= scores.keys())
        or type(k) is not int
        or k <= 0
    ):
        raise ValueError("nDCG needs complete candidates and matching qrels")
    if any(not math.isfinite(score) for score in scores.values()) or any(
        not math.isfinite(value) or value < 0 for value in relevance.values()
    ):
        raise ValueError("nDCG requires finite scores and nonnegative qrels")
    ideal = sorted(relevance.values(), reverse=True)[:k]
    denominator = sum(
        (2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal)
    )
    if denominator <= 0:
        raise ValueError("A development query has no positive support")
    ranked = sorted(scores, key=lambda key: (-scores[key], key))[:k]
    numerator = sum(
        (2 ** relevance.get(key, 0) - 1) / math.log2(rank + 2)
        for rank, key in enumerate(ranked)
    )
    return numerator / denominator


def pair_accuracy(positive: float, negative: float) -> float:
    if not math.isfinite(positive) or not math.isfinite(negative):
        raise ValueError("Pair scores must be finite")
    return float(positive > negative) + 0.5 * float(positive == negative)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        for position in range(begin, end):
            result[order[position]] = (begin + end - 1) / 2
        begin = end
    return result


def spearman(scores: list[float], labels: list[float]) -> float:
    if (
        not scores
        or len(scores) != len(labels)
        or not all(math.isfinite(value) for value in scores + labels)
    ):
        raise ValueError("Spearman requires matching finite vectors")
    left, right = _ranks(scores), _ranks(labels)
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    left = [value - left_mean for value in left]
    right = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value**2 for value in left) * sum(value**2 for value in right)
    )
    if denominator == 0:
        raise ValueError("Spearman is undefined for a constant vector")
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def summarize_query_scores(rows: list[dict], expected_ids: list[str]) -> dict:
    """Aggregate fixed query support, retaining source/language/length slices."""
    if (
        len(set(expected_ids)) != len(expected_ids)
        or len(rows) != len(expected_ids)
        or {row["id"] for row in rows} != set(expected_ids)
    ):
        raise ValueError("Evaluation must include each frozen query exactly once")
    groups = defaultdict(list)
    for row in rows:
        scores = row["scores"]
        if set(scores) != set(row["candidate_ids"]):
            raise ValueError("Missing or unexpected evaluated candidates")
        value = ndcg(scores, row["relevance"])
        groups["overall"].append(value)
        for field in ("source", "language", "length_bucket"):
            if field in row:
                groups[field + ":" + str(row[field])].append(value)
    return {
        name: {"support": len(values), "ndcg10": sum(values) / len(values)}
        for name, values in groups.items()
    }


def _finite(value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Selection metrics must be finite numbers")
    return float(value)


def _weighted_metric(report: dict, terms: list[dict]) -> float:
    if not terms:
        raise ValueError("A configured metric requires at least one explicit path")
    total = 0.0
    for term in terms:
        path = term["path"]
        if (
            not isinstance(path, list)
            or not path
            or any(not isinstance(key, str) or not key for key in path)
        ):
            raise ValueError("Metric paths must be nonempty string lists")
        value = report
        for key in path:
            value = value[key]
        total += _finite(term["weight"]) * _finite(value)
    return _finite(total)


def check_constraints(candidate: dict, baseline: dict, config: dict) -> dict:
    """Apply explicit numeric constraints to identical development support."""
    if (
        candidate["development_manifest_sha256"]
        != baseline["development_manifest_sha256"]
    ):
        raise ValueError("Candidate and baseline must use identical frozen development")
    if (
        candidate["precision"] != config["precision"]
        or baseline["precision"] != config["precision"]
    ):
        raise ValueError("Metric precision differs from the configured comparison")
    expected = config["expected_exits"]
    if (
        not expected
        or len(set(expected)) != len(expected)
        or set(candidate["exits"]) != set(expected)
        or set(baseline["exits"]) != set(expected)
    ):
        raise ValueError("Reports must contain every configured exit exactly")
    if candidate.get("support") != baseline.get("support"):
        raise ValueError("Development metric support changed")
    checks = {}
    for constraint in config["constraints"]:
        name = constraint["name"]
        if not name or name in checks:
            raise ValueError("Constraint names must be nonempty and unique")
        options = set(constraint) & {"minimum", "maximum", "baseline_delta"}
        if len(options) != 1:
            raise ValueError(
                "A constraint needs one minimum, maximum or baseline delta"
            )
        observed = _weighted_metric(candidate, constraint["terms"])
        option = options.pop()
        reference = _finite(constraint[option])
        if option == "baseline_delta":
            reference += _weighted_metric(baseline, constraint["terms"])
        passed = observed <= reference if option == "maximum" else observed >= reference
        checks[name] = {"observed": observed, "required": reference, "passed": passed}
    return {
        "all_constraints_passed": all(value["passed"] for value in checks.values()),
        "checks": checks,
        "selection_score": _weighted_metric(candidate, config["score_terms"]),
    }


def select_candidate(candidates: list[dict], baseline: dict, config: dict) -> dict:
    """Select a feasible checkpoint using explicit score and deterministic ties."""
    preferred = config.get("preferred_runs", [])
    if len(set(preferred)) != len(preferred):
        raise ValueError("Preferred run order contains duplicates")
    reports, eligible, identities = [], [], set()
    for candidate in candidates:
        run_id, step = candidate["run_id"], candidate["step"]
        identity = (run_id, step)
        if (
            not isinstance(run_id, str)
            or not run_id
            or type(step) is not int
            or step < 0
            or identity in identities
        ):
            raise ValueError("Invalid or duplicate run/checkpoint")
        identities.add(identity)
        result = check_constraints(candidate, baseline, config)
        reports.append({"run_id": run_id, "step": step, **result})
        if result["all_constraints_passed"]:
            priority = (
                preferred.index(run_id) if run_id in preferred else len(preferred)
            )
            eligible.append(
                (-result["selection_score"], step, priority, run_id, candidate)
            )
    selected = min(eligible, key=lambda item: item[:4])[-1] if eligible else None
    return {
        "feasible": selected is not None,
        "selected": selected,
        "candidates": reports,
    }
