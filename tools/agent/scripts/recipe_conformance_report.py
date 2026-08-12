"""Deterministic Markdown and JSON report assembly for recipe conformance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_coverage_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Recipe conformance coverage",
        "",
        (
            f"{summary['recipes']} recipes, {summary['entrypoints']} entrypoints, "
            f"{summary['decisions']} decisions, {summary['variants']} probe variants."
        ),
        (
            f"Entrypoints include {summary['auto_entrypoints']} default auto aliases "
            f"and {summary['named_entrypoints']} named recipe aliases."
        ),
        _render_coverage_totals(summary["coverage"]),
        "",
        (
            "| Recipe | Entrypoints | Decisions | Variants | Signals | "
            "Projections | Algorithms | Plugins | Request shapes |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for recipe in payload["recipes"]:
        coverage = recipe["coverage"]
        lines.append(
            f"| {recipe['name']} | {len(recipe['entrypoints'])} | "
            f"{len(recipe['decisions'])} | {recipe['variants']} | "
            f"{coverage['signals']['percent']:.1f}% | "
            f"{coverage['projections']['percent']:.1f}% | "
            f"{coverage['algorithms']['percent']:.1f}% | "
            f"{coverage['plugins']['percent']:.1f}% | "
            f"{', '.join(recipe['request_shapes'])} |"
        )
    lines.extend(
        [
            "",
            "## Ratcheted coverage gaps",
            "",
            ("| Recipe | Signals | Projections | Algorithms | Plugins | Reason |"),
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for recipe in payload["recipes"]:
        coverage = recipe["coverage"]
        dimensions = ("signals", "projections", "algorithms", "plugins")
        gaps = {
            dimension: len(coverage[dimension]["uncovered"]) for dimension in dimensions
        }
        reason = (
            "ratcheted baseline; add explicit probe evidence"
            if any(gaps.values())
            else "fully asserted"
        )
        lines.append(
            f"| {recipe['name']} | {gaps['signals']} | "
            f"{gaps['projections']} | {gaps['algorithms']} | "
            f"{gaps['plugins']} | {reason} |"
        )
    lines.extend(
        [
            "",
            f"Signal families: {', '.join(summary['signal_families'])}",
            "",
            f"Algorithms: {', '.join(summary['algorithms'])}",
            "",
            f"Plugins: {', '.join(summary['plugins']) or 'none'}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_coverage_totals(coverage: dict[str, Any]) -> str:
    dimensions = ("signals", "projections", "algorithms", "plugins")
    return "Coverage: " + ", ".join(
        f"{dimension} {coverage[dimension]['covered']}/"
        f"{coverage[dimension]['total']} "
        f"({coverage[dimension]['percent']:.1f}%)"
        for dimension in dimensions
    )


def render_tag_acceptance_markdown(receipt: dict[str, Any]) -> str:
    lines = [
        "## Robustness tag gates",
        "",
        "| Category | Matched | Total | Pass rate | Minimum |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, result in sorted(receipt.get("categories", {}).items()):
        lines.append(
            f"| {category} | {result['matched']} | {result['total']} | "
            f"{result['pass_rate']:.1f}% | {result['minimum']:.1f}% |"
        )
    if not receipt.get("categories"):
        lines.append("| none configured | 0 | 0 | 100.0% | 0.0% |")
    lines.append("")
    return "\n".join(lines)


def build_consolidated_report(report_root: Path) -> dict[str, Any]:
    inventory_path = report_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(inventory, dict):
        raise TypeError(f"{inventory_path} must contain a JSON object")

    reports: dict[str, dict[str, Any]] = {}
    for report_path in sorted(report_root.glob("*/eval-report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise TypeError(f"{report_path} must contain a JSON object")
        recipe = str(
            _mapping(report.get("inventory")).get("name") or report_path.parent.name
        ).strip()
        reports[recipe] = report

    results = [
        _recipe_result(_mapping(recipe), reports)
        for recipe in _sequence(inventory.get("recipes"))
    ]
    expected = len(results)
    reported = sum(result["status"] != "missing" for result in results)
    passed = sum(result["status"] == "passed" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    missing = expected - reported
    return {
        "schema_version": "v1",
        "inventory": inventory,
        "results": results,
        "summary": {
            "expected_recipes": expected,
            "reported_recipes": reported,
            "passed_recipes": passed,
            "failed_recipes": failed,
            "missing_recipes": missing,
            "matched": sum(result["matched"] for result in results),
            "total": sum(result["total"] for result in results),
            "complete": missing == 0,
            "passed": missing == 0 and failed == 0,
        },
    }


def _recipe_result(
    recipe: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    name = str(recipe.get("name") or "").strip()
    report = reports.get(name)
    evaluation = _mapping(report.get("evaluation")) if report else {}
    is_reported = report is not None
    is_passed = is_reported and bool(evaluation.get("passed"))
    return {
        "recipe": name,
        "status": ("passed" if is_passed else "failed" if is_reported else "missing"),
        "matched": int(evaluation.get("matched") or 0),
        "total": int(evaluation.get("total") or 0),
        "passed": is_passed,
        "report": f"{name}/eval-report.json" if is_reported else None,
        "coverage_acceptance": evaluation.get("coverage_acceptance"),
    }


def render_consolidated_markdown(payload: dict[str, Any]) -> str:
    summary = _mapping(payload.get("summary"))
    lines = [
        render_coverage_markdown(_mapping(payload.get("inventory"))).rstrip(),
        "",
        "## Live CPU results",
        "",
        (
            f"{summary['passed_recipes']} passed, "
            f"{summary['failed_recipes']} failed, "
            f"{summary['missing_recipes']} missing; "
            f"{summary['matched']}/{summary['total']} probes matched."
        ),
        "",
        "| Recipe | Status | Matched | Total |",
        "| --- | --- | ---: | ---: |",
    ]
    for result in _sequence(payload.get("results")):
        lines.append(
            f"| {result['recipe']} | {result['status']} | "
            f"{result['matched']} | {result['total']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
