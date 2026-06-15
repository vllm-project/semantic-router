#!/usr/bin/env python3
"""Helpers for indexed governance docs such as execution plans."""

from __future__ import annotations

import re
from pathlib import Path

from agent_support import REPO_ROOT

PLAN_REQUIRED_SECTIONS = [
    "## Goal",
    "## Scope",
    "## Exit Criteria",
    "## Task List",
    "## Next Action",
    "## Operating Rules",
    "## Related Docs",
]
PLAN_FILENAME_PATTERN = re.compile(r"^pl-(\d{4})-[a-z0-9-]+\.md$")


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

    plan_indices: list[str] = []
    for plan_doc in plan_docs:
        plan_index = validate_single_plan_doc(
            plan_doc, plan_readme_text, manifest_docs, errors
        )
        if plan_index:
            plan_indices.append(plan_index)

    duplicate_plan_indices = sorted(
        {index for index in plan_indices if plan_indices.count(index) > 1}
    )
    if duplicate_plan_indices:
        errors.append(
            "docs/agent/plans has duplicate execution plan indices: "
            + ", ".join(duplicate_plan_indices)
        )


def validate_single_plan_doc(
    plan_doc: str, plan_readme_text: str, manifest_docs: set[str], errors: list[str]
) -> str:
    if plan_doc not in manifest_docs:
        errors.append(f"repo-manifest docs is missing execution plan '{plan_doc}'")

    plan_name = Path(plan_doc).name
    filename_match = PLAN_FILENAME_PATTERN.match(plan_name)
    if not filename_match:
        errors.append(
            f"Execution plan '{plan_doc}' must use a slug like 'pl-0001-example.md'"
        )
    filename_index = filename_match.group(1) if filename_match else ""
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
    return filename_index
