#!/usr/bin/env python3
"""Helpers for indexed governance docs such as ADRs and execution plans."""

from __future__ import annotations

import re
from pathlib import Path

from agent_support import REPO_ROOT

ADR_REQUIRED_SECTIONS = [
    "## Status",
    "## Context",
    "## Decision",
    "## Consequences",
]
PLAN_REQUIRED_SECTIONS = [
    "## Goal",
    "## Scope",
    "## Exit Criteria",
    "## Task List",
    "## Current Loop",
    "## Decision Log",
    "## Follow-up Debt / ADR Links",
]
ADR_FILENAME_PATTERN = re.compile(r"^\d{4}-[a-z0-9-]+\.md$")


def validate_adr_inventory_and_template(repo_manifest: dict, errors: list[str]) -> None:
    adr_dir = REPO_ROOT / "docs" / "agent" / "adr"
    if not adr_dir.exists():
        errors.append("Missing docs/agent/adr directory")
        return

    adr_readme = adr_dir / "README.md"
    if not adr_readme.exists():
        errors.append("Missing docs/agent/adr/README.md")
        return

    adr_readme_text = adr_readme.read_text(encoding="utf-8")
    manifest_docs = set(repo_manifest.get("docs", []))
    adr_docs = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in adr_dir.glob("*.md")
        if path.name != "README.md"
    )

    for adr_doc in adr_docs:
        validate_single_adr_doc(adr_doc, adr_readme_text, manifest_docs, errors)


def validate_single_adr_doc(
    adr_doc: str, adr_readme_text: str, manifest_docs: set[str], errors: list[str]
) -> None:
    if adr_doc not in manifest_docs:
        errors.append(f"repo-manifest docs is missing ADR '{adr_doc}'")

    adr_name = Path(adr_doc).name
    if not ADR_FILENAME_PATTERN.match(adr_name):
        errors.append(
            f"ADR '{adr_doc}' must use a zero-padded numeric slug like '0001-example.md'"
        )

    if f"({adr_name})" not in adr_readme_text:
        errors.append(f"docs/agent/adr/README.md must link to ADR '{adr_doc}'")

    adr_text = (REPO_ROOT / adr_doc).read_text(encoding="utf-8")
    if not adr_text.startswith("# ADR "):
        errors.append(f"ADR '{adr_doc}' must start with '# ADR '")
    for section in ADR_REQUIRED_SECTIONS:
        if section not in adr_text:
            errors.append(f"ADR '{adr_doc}' is missing required section '{section}'")


def validate_plan_inventory_and_template(
    repo_manifest: dict, errors: list[str]
) -> None:
    plan_dir = REPO_ROOT / "docs" / "agent" / "plans"
    if not plan_dir.exists():
        errors.append("Missing docs/agent/plans directory")
        return

    plan_readme = plan_dir / "README.md"
    if not plan_readme.exists():
        errors.append("Missing docs/agent/plans/README.md")
        return

    plan_readme_text = plan_readme.read_text(encoding="utf-8")
    manifest_docs = set(repo_manifest.get("docs", []))
    plan_docs = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in plan_dir.glob("*.md")
        if path.name != "README.md"
    )

    for plan_doc in plan_docs:
        validate_single_plan_doc(plan_doc, plan_readme_text, manifest_docs, errors)


def validate_single_plan_doc(
    plan_doc: str, plan_readme_text: str, manifest_docs: set[str], errors: list[str]
) -> None:
    if plan_doc not in manifest_docs:
        errors.append(f"repo-manifest docs is missing execution plan '{plan_doc}'")

    plan_name = Path(plan_doc).name
    if f"({plan_name})" not in plan_readme_text:
        errors.append(
            f"docs/agent/plans/README.md must link to execution plan '{plan_doc}'"
        )

    plan_text = (REPO_ROOT / plan_doc).read_text(encoding="utf-8")
    for section in PLAN_REQUIRED_SECTIONS:
        if section not in plan_text:
            errors.append(
                f"Execution plan '{plan_doc}' is missing required section '{section}'"
            )
