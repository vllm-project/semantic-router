#!/usr/bin/env python3
"""Doc and governance validation for the shared agent harness."""

from __future__ import annotations

import re
from pathlib import Path

from agent_support import (
    ABSOLUTE_MARKDOWN_LINK_PATTERN,
    AGENT_GOVERNANCE_DOC,
    AGENT_INDEX_DOC,
    AGENTS_ENTRY_DOC,
    REPO_ROOT,
)

MAX_AGENTS_ENTRY_LINES = 90
TEMPORARY_WORKING_NOTES = {"TASKS.md", "CI_LOOP.md"}
REQUIRED_DOC_SECTIONS = {
    "docs/agent/README.md": [
        "## Start Here",
        "## Governance and Structure",
        "## Task-Specific Guidance",
        "## Executable Contract",
        "## Contributor Interface",
    ],
    "docs/agent/governance.md": [
        "## Rule Layers",
        "## Source of Truth Policy",
        "## What Does Not Belong in the Canonical Harness",
        "## Maintenance Rules",
        "## Standard Validation Entry",
    ],
    "docs/agent/tech-debt-register.md": [
        "## Why This Exists",
        "## Policy",
        "## Open Debt Items",
        "## How to Retire Debt",
    ],
    "docs/agent/plans/README.md": [
        "## When to Use an Execution Plan",
        "## What Belongs in an Execution Plan",
        "## What Does Not Belong in an Execution Plan",
        "## Execution Plan Versus Other Governance Files",
        "## Execution Plan Template",
        "## Current Execution Plans",
    ],
    "docs/agent/local-rules.md": [
        "## Indexed Local `AGENTS.md` Files",
        "## Policy",
    ],
    "docs/agent/skill-catalog.md": [
        "## Primary Skills",
        "## Fragment Skills",
        "## Support Skills",
        "## Source of Truth",
    ],
    "docs/agent/adr/README.md": [
        "## When to Create or Update an ADR",
        "## What Belongs in an ADR",
        "## What Does Not Belong in an ADR",
        "## ADR Versus Other Governance Files",
        "## ADR Template",
        "## Current ADRs",
    ],
}
LOCAL_AGENT_REQUIRED_SECTIONS = [
    "## Scope",
    "## Responsibilities",
    "## Change Rules",
]
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
TECH_DEBT_ITEM_PATTERN = re.compile(r"^### (TD\d{3})\b", re.MULTILINE)
ADR_FILENAME_PATTERN = re.compile(r"^\d{4}-[a-z0-9-]+\.md$")


def validate_support_files(repo_manifest: dict, errors: list[str]) -> None:
    for doc_path in repo_manifest["docs"]:
        absolute_doc_path = REPO_ROOT / doc_path
        if not absolute_doc_path.exists():
            errors.append(f"Missing agent doc: {doc_path}")
            continue
        validate_portable_markdown_links(absolute_doc_path, errors)
    for skill_path in repo_manifest["skills"]:
        absolute_skill_path = REPO_ROOT / skill_path
        if not absolute_skill_path.exists():
            errors.append(f"Missing agent skill: {skill_path}")
            continue
        validate_portable_markdown_links(absolute_skill_path, errors)


def validate_portable_markdown_links(path: Path, errors: list[str]) -> None:
    if path.suffix.lower() != ".md":
        return

    text = path.read_text(encoding="utf-8")
    for match in ABSOLUTE_MARKDOWN_LINK_PATTERN.finditer(text):
        target = match.group(1)
        errors.append(
            f"Portable docs only: {path.relative_to(REPO_ROOT)} uses absolute markdown link '{target}'"
        )


def validate_agent_harness_layers(
    repo_manifest: dict, task_matrix: dict, skill_registry: dict, errors: list[str]
) -> None:
    validate_agent_entry_docs(repo_manifest, errors)
    validate_agent_contract_subsystem(repo_manifest, errors)
    validate_task_matrix_contract_paths(task_matrix, errors)
    validate_no_temporary_notes_in_agents(errors)
    validate_agent_doc_inventory(repo_manifest, errors)
    validate_doc_index_coverage(repo_manifest, errors)
    validate_local_agent_rule_inventory(repo_manifest, errors)
    validate_local_agent_rule_routing(
        repo_manifest, task_matrix, skill_registry, errors
    )
    validate_doc_governance(repo_manifest, errors)
    validate_required_doc_sections(errors)
    validate_adr_inventory_and_template(repo_manifest, errors)
    validate_plan_inventory_and_template(repo_manifest, errors)
    validate_tech_debt_register(errors)


def validate_agent_entry_docs(repo_manifest: dict, errors: list[str]) -> None:
    if not AGENT_INDEX_DOC.exists():
        errors.append("Missing docs/agent/README.md")
    if not AGENT_GOVERNANCE_DOC.exists():
        errors.append("Missing docs/agent/governance.md")

    agents_text = AGENTS_ENTRY_DOC.read_text(encoding="utf-8")
    agent_line_count = len(agents_text.splitlines())
    if agent_line_count > MAX_AGENTS_ENTRY_LINES:
        errors.append(
            f"AGENTS.md is {agent_line_count} lines; keep the agent entry under {MAX_AGENTS_ENTRY_LINES} lines"
        )
    if "docs/agent/README.md" not in agents_text:
        errors.append("AGENTS.md must link to docs/agent/README.md")

    repo_docs = set(repo_manifest.get("docs", []))
    for required_doc in ("docs/agent/README.md", "docs/agent/governance.md"):
        if required_doc not in repo_docs:
            errors.append(
                f"repo-manifest docs must include canonical agent doc '{required_doc}'"
            )


def validate_agent_contract_subsystem(repo_manifest: dict, errors: list[str]) -> None:
    agent_contract = next(
        (
            subsystem
            for subsystem in repo_manifest.get("subsystems", [])
            if subsystem.get("name") == "agent-contract"
        ),
        None,
    )
    if agent_contract is None:
        errors.append("repo-manifest is missing the 'agent-contract' subsystem")
        return

    if "docs/agent/README.md" not in agent_contract.get("entrypoints", []):
        errors.append("agent-contract entrypoints must include docs/agent/README.md")
    validate_no_temporary_canonical_refs(
        "repo-manifest agent-contract paths",
        agent_contract.get("paths", []),
        errors,
    )
    validate_no_temporary_canonical_refs(
        "repo-manifest agent-contract entrypoints",
        agent_contract.get("entrypoints", []),
        errors,
    )


def validate_task_matrix_contract_paths(task_matrix: dict, errors: list[str]) -> None:
    for rule in task_matrix.get("rules", []):
        if rule.get("name") not in {"agent_text", "agent_exec"}:
            continue
        validate_no_temporary_canonical_refs(
            f"task-matrix rule '{rule['name']}' paths",
            rule.get("paths", []),
            errors,
        )


def validate_no_temporary_notes_in_agents(errors: list[str]) -> None:
    agents_text = AGENTS_ENTRY_DOC.read_text(encoding="utf-8")
    for note_name in TEMPORARY_WORKING_NOTES:
        if note_name in agents_text:
            errors.append(
                f"AGENTS.md must not reference temporary working note '{note_name}'"
            )


def validate_no_temporary_canonical_refs(
    label: str, values: list[str], errors: list[str]
) -> None:
    for value in values:
        if value in TEMPORARY_WORKING_NOTES:
            errors.append(f"{label} must not include temporary working note '{value}'")


def validate_agent_doc_inventory(repo_manifest: dict, errors: list[str]) -> None:
    manifest_docs = set(repo_manifest.get("docs", []))
    actual_docs = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted((REPO_ROOT / "docs" / "agent").rglob("*.md"))
    }
    missing_from_manifest = actual_docs - manifest_docs
    if missing_from_manifest:
        errors.append(
            "repo-manifest docs is missing canonical agent docs: "
            + ", ".join(sorted(missing_from_manifest))
        )


def validate_doc_index_coverage(repo_manifest: dict, errors: list[str]) -> None:
    readme_text = AGENT_INDEX_DOC.read_text(encoding="utf-8")
    for doc_path in repo_manifest.get("docs", []):
        if not doc_path.startswith("docs/agent/") or doc_path == "docs/agent/README.md":
            continue
        if (
            doc_path.startswith("docs/agent/adr/")
            and doc_path != "docs/agent/adr/README.md"
        ):
            continue
        if (
            doc_path.startswith("docs/agent/plans/")
            and doc_path != "docs/agent/plans/README.md"
        ):
            continue
        relative_target = Path(doc_path).relative_to("docs/agent").as_posix()
        if f"({relative_target})" not in readme_text:
            errors.append(
                f"docs/agent/README.md must link to canonical agent doc '{doc_path}'"
            )


def validate_local_agent_rule_inventory(repo_manifest: dict, errors: list[str]) -> None:
    local_rules = repo_manifest.get("local_agent_rules", [])
    if not local_rules:
        errors.append("repo-manifest must define local_agent_rules")
        return

    local_rules_text = (REPO_ROOT / "docs" / "agent" / "local-rules.md").read_text(
        encoding="utf-8"
    )
    for entry in local_rules:
        path = entry["path"]
        absolute_path = REPO_ROOT / path
        if not absolute_path.exists():
            errors.append(f"Missing local agent rule: {path}")
            continue
        for field in ("steward", "freshness"):
            if not entry.get(field):
                errors.append(f"Local agent rule '{path}' is missing {field}")
        relative_link_target = Path(path).as_posix()
        if (
            f"({Path('..', '..', relative_link_target).as_posix()})"
            not in local_rules_text
        ):
            errors.append(f"docs/agent/local-rules.md must link to '{path}'")
        validate_local_agent_rule_template(path, absolute_path, errors)


def validate_local_agent_rule_routing(
    repo_manifest: dict, task_matrix: dict, skill_registry: dict, errors: list[str]
) -> None:
    local_paths = {
        entry["path"] for entry in repo_manifest.get("local_agent_rules", [])
    }
    if not local_paths:
        return

    agent_text_rule = next(
        (
            rule
            for rule in task_matrix.get("rules", [])
            if rule.get("name") == "agent_text"
        ),
        None,
    )
    if agent_text_rule is None:
        errors.append("task-matrix is missing the 'agent_text' rule")
        return

    missing_from_agent_text = local_paths - set(agent_text_rule.get("paths", []))
    if missing_from_agent_text:
        errors.append(
            "task-matrix agent_text paths are missing local AGENTS rules: "
            + ", ".join(sorted(missing_from_agent_text))
        )

    harness_docs = skill_registry.get("surfaces", {}).get("harness_docs", {})
    missing_from_harness_docs = local_paths - set(harness_docs.get("paths", []))
    if missing_from_harness_docs:
        errors.append(
            "skill-registry surface 'harness_docs' is missing local AGENTS rules: "
            + ", ".join(sorted(missing_from_harness_docs))
        )

    harness_skill = next(
        (
            skill
            for skill in skill_registry.get("skills", {}).get("primary", [])
            if skill.get("name") == "harness-contract-change"
        ),
        None,
    )
    if harness_skill is None:
        errors.append(
            "skill-registry is missing primary skill 'harness-contract-change'"
        )
        return

    missing_from_selector_paths = local_paths - set(
        harness_skill.get("selector_paths", [])
    )
    if missing_from_selector_paths:
        errors.append(
            "primary skill 'harness-contract-change' is missing local AGENTS selector paths: "
            + ", ".join(sorted(missing_from_selector_paths))
        )


def validate_local_agent_rule_template(
    relative_path: str, absolute_path: Path, errors: list[str]
) -> None:
    text = absolute_path.read_text(encoding="utf-8")
    for section in LOCAL_AGENT_REQUIRED_SECTIONS:
        if section not in text:
            errors.append(
                f"{relative_path} is missing required section '{section}' for local agent rules"
            )


def validate_doc_governance(repo_manifest: dict, errors: list[str]) -> None:
    governance = repo_manifest.get("doc_governance")
    if not governance:
        errors.append("repo-manifest must define doc_governance")
        return

    owner_source = governance.get("owner_source")
    if not owner_source:
        errors.append("doc_governance is missing owner_source")
    elif not (REPO_ROOT / owner_source).exists():
        errors.append(
            f"doc_governance references missing owner_source '{owner_source}'"
        )

    governed_paths = {
        entry["path"]
        for entry in governance.get("canonical_docs", [])
        if entry.get("path")
    }
    expected_paths = {
        path
        for path in repo_manifest.get("docs", [])
        if path == "AGENTS.md" or path.startswith("docs/agent/")
    }
    missing_paths = expected_paths - governed_paths
    if missing_paths:
        errors.append(
            "doc_governance is missing canonical doc entries for: "
            + ", ".join(sorted(missing_paths))
        )

    for entry in governance.get("canonical_docs", []):
        path = entry.get("path")
        if not path:
            errors.append("doc_governance has an entry without path")
            continue
        if not (REPO_ROOT / path).exists():
            errors.append(f"doc_governance references missing doc '{path}'")
        for field in ("steward", "freshness"):
            if not entry.get(field):
                errors.append(f"doc_governance entry '{path}' is missing {field}")


def validate_required_doc_sections(errors: list[str]) -> None:
    for doc_path, sections in REQUIRED_DOC_SECTIONS.items():
        text = (REPO_ROOT / doc_path).read_text(encoding="utf-8")
        for section in sections:
            if section not in text:
                errors.append(f"{doc_path} is missing required section '{section}'")


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
                errors.append(
                    f"ADR '{adr_doc}' is missing required section '{section}'"
                )


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


def validate_tech_debt_register(errors: list[str]) -> None:
    debt_path = REPO_ROOT / "docs" / "agent" / "tech-debt-register.md"
    if not debt_path.exists():
        return

    text = debt_path.read_text(encoding="utf-8")
    item_ids = TECH_DEBT_ITEM_PATTERN.findall(text)
    if not item_ids:
        errors.append(
            "docs/agent/tech-debt-register.md must contain at least one TD item"
        )
        return

    duplicates = sorted(
        {item_id for item_id in item_ids if item_ids.count(item_id) > 1}
    )
    if duplicates:
        errors.append(
            "docs/agent/tech-debt-register.md has duplicate debt IDs: "
            + ", ".join(duplicates)
        )

    headings = re.findall(r"^### (TD\d{3}\b.*)$", text, flags=re.MULTILINE)
    chunks = re.split(r"^### TD\d{3}\b.*$", text, flags=re.MULTILINE)[1:]
    for heading, chunk in zip(headings, chunks, strict=False):
        for required_marker in (
            "- Status:",
            "- Scope:",
            "- Evidence:",
            "- Exit criteria:",
        ):
            if required_marker not in chunk:
                errors.append(
                    f"docs/agent/tech-debt-register.md item '{heading}' is missing '{required_marker}'"
                )
