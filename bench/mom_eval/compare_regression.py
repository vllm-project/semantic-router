"""Regression comparison for MoM evaluation result bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from bench.mom_eval.common import REPO_ROOT, SCORECARD_INDEX, load_core_suite, write_json


def load_previous_result(recipe_id: str, entrypoint: str, recipe_version: str) -> dict[str, Any] | None:
    index_path = SCORECARD_INDEX
    if not index_path.is_file():
        return None
    index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    scorecards = (index.get("scorecards") or {}).get(recipe_id, {})
    entry_versions = scorecards.get(entrypoint) or {}
    previous_versions = [
        version for version in entry_versions.keys() if version != recipe_version
    ]
    if not previous_versions:
        return None
    previous_version = sorted(previous_versions)[-1]
    rel_path = entry_versions[previous_version]["result_path"]
    path = REPO_ROOT / rel_path
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_regression(current: dict[str, Any]) -> dict[str, Any]:
    identity = current.get("identity") or {}
    previous = load_previous_result(
        str(identity.get("recipe_id")),
        str(identity.get("entrypoint")),
        str(identity.get("recipe_version")),
    )
    core = load_core_suite()
    thresholds: dict[str, float] = {}
    for layer in (core.get("layers") or {}).values():
        for benchmark in layer.get("benchmarks") or []:
            if "regression_threshold" in benchmark:
                thresholds[str(benchmark["id"])] = float(benchmark["regression_threshold"])

    deltas: dict[str, float | None] = {}
    blocking_failures: list[str] = []
    if previous is None:
        return {
            "baseline_version": None,
            "deltas": deltas,
            "blocking_failures": blocking_failures,
            "passed": True,
            "note": "no previous scorecard to compare",
        }

    prev_metrics = previous.get("metrics") or {}
    curr_metrics = current.get("metrics") or {}
    for metric_id, threshold in thresholds.items():
        prev_value = (prev_metrics.get(metric_id) or {}).get("value")
        curr_value = (curr_metrics.get(metric_id) or {}).get("value")
        if prev_value is None or curr_value is None:
            continue
        delta = float(curr_value) - float(prev_value)
        deltas[metric_id] = delta
        if delta < threshold:
            blocking_failures.append(
                f"{metric_id} regressed by {delta:.2f} (threshold {threshold:.2f})"
            )

    passed = not blocking_failures
    publication = current.setdefault("publication", {})
    if not passed:
        publication["publishable"] = False
        publication["classification"] = "blocked"
        reasons = list(publication.get("blocking_reasons") or [])
        reasons.extend(blocking_failures)
        publication["blocking_reasons"] = reasons

    return {
        "baseline_version": (previous.get("identity") or {}).get("recipe_version"),
        "deltas": deltas,
        "blocking_failures": blocking_failures,
        "passed": passed,
    }


def write_regression_report(output_path: Path, report: dict[str, Any]) -> None:
    write_json(output_path, report)
