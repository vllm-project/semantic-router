#!/usr/bin/env python3
"""Harness scorecard generation for ongoing agent-contract governance."""

from __future__ import annotations

from agent_models import HarnessScorecard
from agent_support import REPO_ROOT, load_manifests
from agent_validation import collect_validation_errors


def build_harness_scorecard() -> HarnessScorecard:
    repo_manifest, _, _, _, skill_registry = load_manifests()
    validation_errors = collect_validation_errors()
    skills = skill_registry.get("skills", {})
    return HarnessScorecard(
        validation_status="pass" if not validation_errors else "fail",
        validation_error_count=len(validation_errors),
        indexed_harness_doc_count=len(repo_manifest.get("docs", [])),
        governed_doc_count=len(
            repo_manifest.get("doc_governance", {}).get("canonical_docs", [])
        ),
        open_technical_debt_items=collect_open_tech_debt_items(),
        local_rule_count=len(repo_manifest.get("local_agent_rules", [])),
        subsystem_count=len(repo_manifest.get("subsystems", [])),
        surface_count=len(skill_registry.get("surfaces", {})),
        primary_skill_count=len(skills.get("primary", [])),
        fragment_skill_count=len(skills.get("fragments", [])),
        support_skill_count=len(skills.get("support", [])),
        legacy_reference_skill_count=len(skills.get("legacy_reference", [])),
        open_execution_plan_tasks=collect_open_execution_plan_tasks(),
        validation_errors=validation_errors,
    )


def collect_open_execution_plan_tasks() -> list[str]:
    tasks: list[str] = []
    plan_dir = REPO_ROOT / "docs" / "agent" / "plans"
    if not plan_dir.exists():
        return tasks

    for plan_path in sorted(plan_dir.glob("*.md")):
        if plan_path.name == "README.md":
            continue
        for line in plan_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- [ ]"):
                tasks.append(f"{plan_path.name}: {stripped[6:].strip()}")
    return tasks


def collect_open_tech_debt_items() -> list[str]:
    debt_path = REPO_ROOT / "docs" / "agent" / "tech-debt-register.md"
    if not debt_path.exists():
        return []

    items: list[str] = []
    current_heading: str | None = None
    for line in debt_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("### TD"):
            current_heading = stripped.removeprefix("### ").strip()
            continue
        if current_heading and stripped.lower() == "- status: open":
            items.append(current_heading)
            current_heading = None
    return items
