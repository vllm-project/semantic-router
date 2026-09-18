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
            "| Recipe identity | Version | Entrypoints | Decisions | Variants | "
            "Signals | Projections | Algorithms | Plugins | Request shapes | Devices |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for recipe in payload["recipes"]:
        coverage = recipe["coverage"]
        identity = _mapping(recipe.get("identity"))
        recipe_id = str(identity.get("id") or recipe["name"])
        display_name = str(identity.get("name") or recipe_id)
        lines.append(
            f"| {display_name} (`{recipe_id}`) | {identity.get('version', '')} | "
            f"{len(recipe['entrypoints'])} | "
            f"{len(recipe['decisions'])} | {recipe['variants']} | "
            f"{coverage['signals']['percent']:.1f}% | "
            f"{coverage['projections']['percent']:.1f}% | "
            f"{coverage['algorithms']['percent']:.1f}% | "
            f"{coverage['plugins']['percent']:.1f}% | "
            f"{', '.join(recipe['request_shapes'])} | "
            f"{', '.join(recipe.get('required_devices', [])) or 'CPU compatible'} |"
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
    reported = sum(result["status"] in {"passed", "failed"} for result in results)
    passed = sum(result["status"] == "passed" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    missing = sum(result["status"] == "missing" for result in results)
    requires_hardware = sum(
        result["status"] == "requires_hardware" for result in results
    )
    cpu_results = [result for result in results if not result["required_devices"]]
    return {
        "schema_version": "v1",
        "inventory": inventory,
        "results": results,
        "summary": {
            "evaluation_scopes": sorted(
                {result["evaluation_scope"] for result in results if result["report"]}
            ),
            "deployment_passed": reported == expected
            and all(result["deployment_passed"] for result in results),
            "expected_recipes": expected,
            "reported_recipes": reported,
            "passed_recipes": passed,
            "failed_recipes": failed,
            "missing_recipes": missing,
            "requires_hardware_recipes": requires_hardware,
            "matched": sum(result["matched"] for result in results),
            "total": sum(result["total"] for result in results),
            "complete": reported == expected,
            "passed": reported == expected and failed == 0,
            "cpu_compatible_complete": bool(cpu_results)
            and all(result["status"] != "missing" for result in cpu_results),
            "cpu_compatible_passed": bool(cpu_results)
            and all(result["status"] == "passed" for result in cpu_results),
        },
    }


def _recipe_result(
    recipe: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    name = str(recipe.get("name") or "").strip()
    report = reports.get(name)
    evaluation = _mapping(report.get("evaluation")) if report else {}
    is_reported = report is not None
    total = int(evaluation.get("total") or 0)
    expected = int(recipe.get("variants") or total)
    execution = _mapping(evaluation.get("execution"))
    complete = total > 0 and total == expected
    if "variants" in recipe:
        complete = complete and bool(execution.get("complete"))
    is_passed = is_reported and bool(evaluation.get("passed")) and complete
    scope = str(evaluation.get("evaluation_scope") or "deployment")
    devices = recipe.get("required_devices", [])
    absent_status = "requires_hardware" if devices else "missing"
    return {
        "recipe": name,
        "evaluation_scope": scope,
        "deployment_passed": (
            is_passed
            if scope == "deployment"
            else bool(
                _mapping(_mapping(evaluation.get("scopes")).get("deployment")).get(
                    "passed"
                )
            )
        ),
        "scopes": evaluation.get("scopes"),
        "selection_status_counts": evaluation.get("selection_status_counts"),
        "selection_reasons": evaluation.get("selection_reasons"),
        "status": (
            "passed" if is_passed else "failed" if is_reported else absent_status
        ),
        "matched": int(evaluation.get("matched") or 0),
        "total": total,
        "expected_probes": expected,
        "execution": execution,
        "passed": is_passed,
        "required_devices": devices,
        "report": f"{name}/eval-report.json" if is_reported else None,
        "coverage_acceptance": evaluation.get("coverage_acceptance"),
    }


def render_consolidated_markdown(payload: dict[str, Any]) -> str:
    summary = _mapping(payload.get("summary"))
    lines = [
        render_coverage_markdown(_mapping(payload.get("inventory"))).rstrip(),
        "",
        "## Live results",
        "",
        (
            f"{summary['passed_recipes']} passed, "
            f"{summary['failed_recipes']} failed, "
            f"{summary['missing_recipes']} missing, "
            f"{summary.get('requires_hardware_recipes', 0)} require hardware; "
            f"{summary['matched']}/{summary['total']} probes matched."
        ),
        "",
        f"Evaluation scopes: {', '.join(summary.get('evaluation_scopes') or ['deployment'])}. Deployment passed: `{summary.get('deployment_passed', False)}`.",
        "",
        "| Recipe | Scope | Status | Matched | Total | Required devices |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for result in _sequence(payload.get("results")):
        lines.append(
            f"| {result['recipe']} | {result.get('evaluation_scope', 'deployment')} | {result['status']} | "
            f"{result['matched']} | {result['total']} | "
            f"{', '.join(result.get('required_devices', []))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
