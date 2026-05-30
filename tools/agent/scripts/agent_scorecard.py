#!/usr/bin/env python3
"""Harness scorecard generation for ongoing agent-contract governance."""

from __future__ import annotations

import re

from agent_models import HarnessScorecard
from agent_support import REPO_ROOT, load_manifests
from agent_tech_debt_support import collect_open_tech_debt_items
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
        support_skill_count=len(skills.get("support", [])),
        current_execution_plans=collect_current_execution_plan_paths(),
        open_execution_plan_tasks=collect_open_execution_plan_tasks(),
        validation_errors=validation_errors,
    )


PLAN_LINK_PATTERN = re.compile(r"\((pl-\d{4}-[a-z0-9-]+\.md)\)")


def collect_current_execution_plan_paths() -> list[str]:
    plan_readme = REPO_ROOT / "docs" / "agent" / "plans" / "README.md"
    text = plan_readme.read_text(encoding="utf-8")
    lines = read_section_lines(text, "## Current Execution Plans")
    plans: list[str] = []
    for line in lines:
        match = PLAN_LINK_PATTERN.search(line)
        if match:
            plans.append(match.group(1))
    return plans


def collect_open_execution_plan_tasks() -> list[str]:
    tasks: list[str] = []
    plan_dir = REPO_ROOT / "docs" / "agent" / "plans"
    if not plan_dir.exists():
        return tasks

    for plan_name in collect_current_execution_plan_paths():
        plan_path = plan_dir / plan_name
        if not plan_path.exists():
            continue
        for task in collect_open_checklist_items(plan_path.read_text(encoding="utf-8")):
            tasks.append(f"{plan_path.name}: {task}")
    return tasks


def collect_open_checklist_items(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [ ]"):
            if current:
                items.append(" ".join(current))
            current = [stripped[6:].strip()]
            continue
        if stripped.startswith("- ["):
            if current:
                items.append(" ".join(current))
            current = None
            continue
        if current is not None:
            if line.startswith((" ", "\t")) and stripped:
                current.append(stripped)
                continue
            if stripped:
                items.append(" ".join(current))
                current = None
    if current:
        items.append(" ".join(current))
    return items


def read_section_lines(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    collected: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == heading:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section:
            collected.append(line)
    return collected
