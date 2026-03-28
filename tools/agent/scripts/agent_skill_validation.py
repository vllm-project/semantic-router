#!/usr/bin/env python3
"""Skill and surface validation for the shared agent harness."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import yaml
from agent_support import (
    CHANGE_SURFACES_DOC,
    REPO_ROOT,
    build_skill_lookup,
    collect_task_rule_names,
    iter_registry_skills,
)

SKILL_REQUIRED_SECTIONS = {
    "primary": [
        "## Trigger",
        "## Workflow",
        "## Must Read",
        "## Standard Commands",
        "## Gotchas",
        "## Acceptance",
    ],
    "fragments": [
        "## Trigger",
        "## Workflow",
        "## Must Read",
        "## Standard Commands",
        "## Acceptance",
    ],
    "support": [
        "## Trigger",
        "## Workflow",
        "## Must Read",
        "## Standard Commands",
        "## Gotchas",
        "## Acceptance",
    ],
    "legacy_reference": [
        "## Trigger",
        "## Workflow",
        "## Must Read",
        "## Standard Commands",
        "## Acceptance",
    ],
}
SKILL_FRONTMATTER_PATTERN = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
SKILL_CATEGORY_FRONTMATTER = {
    "primary": "primary",
    "fragments": "fragment",
    "support": "support",
    "legacy_reference": "legacy_reference",
}
MAX_MUST_READ_LINKS_BY_CATEGORY = {
    "primary": 3,
    "fragments": 3,
    "support": 4,
    "legacy_reference": 4,
}

DEFERRED_MUST_READ_PATHS = {
    "docs/agent/feature-complete-checklist.md": (
        "close-out checklist docs belong in workflow or acceptance, not Must Read"
    ),
}
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def validate_skill_registry(
    repo_manifest: dict, task_matrix: dict, skill_registry: dict, errors: list[str]
) -> None:
    task_rule_names = collect_task_rule_names(task_matrix)
    skill_lookup = build_skill_lookup(skill_registry)
    validate_surface_catalog(task_rule_names, skill_registry, errors)
    validate_skill_entries(repo_manifest, skill_lookup, skill_registry, errors)
    validate_skill_catalog(skill_registry, errors)


def validate_surface_catalog(
    task_rule_names: set[str], skill_registry: dict, errors: list[str]
) -> None:
    change_surfaces_text = CHANGE_SURFACES_DOC.read_text(encoding="utf-8")
    for surface_name, surface in skill_registry["surfaces"].items():
        if not surface.get("description"):
            errors.append(f"Surface '{surface_name}' is missing description")
        if not surface.get("paths"):
            errors.append(f"Surface '{surface_name}' has no paths")
        if not surface.get("task_rules"):
            errors.append(f"Surface '{surface_name}' has no task_rules")
        for task_rule in surface.get("task_rules", []):
            if task_rule not in task_rule_names:
                errors.append(
                    f"Surface '{surface_name}' references unknown task rule '{task_rule}'"
                )
        if f"`{surface_name}`" not in change_surfaces_text:
            errors.append(
                f"Surface '{surface_name}' is missing from docs/agent/change-surfaces.md"
            )


def validate_skill_entries(
    repo_manifest: dict,
    skill_lookup: dict[str, dict],
    skill_registry: dict,
    errors: list[str],
) -> None:
    manifest_skill_paths = set(repo_manifest["skills"])
    registry_skill_paths: set[str] = set()

    for skill in iter_registry_skills(skill_registry):
        skill_path = validate_skill_file_reference(skill, manifest_skill_paths, errors)
        if skill_path:
            registry_skill_paths.add(skill_path)
        validate_skill_definition(skill, skill_lookup, skill_registry, errors)

    missing_from_registry = manifest_skill_paths - registry_skill_paths
    if missing_from_registry:
        errors.append(
            "repo-manifest lists skills missing from skill-registry: "
            + ", ".join(sorted(missing_from_registry))
        )


def validate_skill_file_reference(
    skill: dict, manifest_skill_paths: set[str], errors: list[str]
) -> str | None:
    skill_name = skill["name"]
    skill_path = skill.get("path")
    if not skill_path:
        errors.append(f"Skill '{skill_name}' is missing path")
        return None
    if skill_path not in manifest_skill_paths:
        errors.append(
            f"Skill '{skill_name}' path '{skill_path}' is missing from repo-manifest skills"
        )
    if not (REPO_ROOT / skill_path).exists():
        errors.append(f"Skill '{skill_name}' references missing file '{skill_path}'")
    return skill_path


def validate_skill_definition(
    skill: dict, skill_lookup: dict[str, dict], skill_registry: dict, errors: list[str]
) -> None:
    skill_name = skill["name"]
    category = skill["category"]
    validate_skill_frontmatter(skill, errors)
    if not skill.get("description"):
        errors.append(f"Skill '{skill_name}' is missing description")

    if category == "primary":
        validate_primary_skill(skill, skill_lookup, skill_registry, errors)
    elif category == "fragments":
        validate_fragment_skill(skill, skill_registry, errors)
    elif category == "support" and not skill.get("acceptance_criteria"):
        errors.append(f"Support skill '{skill_name}' is missing acceptance_criteria")


def validate_skill_catalog(skill_registry: dict, errors: list[str]) -> None:
    catalog_text = (REPO_ROOT / "docs" / "agent" / "skill-catalog.md").read_text(
        encoding="utf-8"
    )
    for skill in iter_registry_skills(skill_registry):
        if f"`{skill['name']}`" not in catalog_text:
            errors.append(
                f"docs/agent/skill-catalog.md must list skill '{skill['name']}'"
            )
        validate_skill_template(skill, errors)


def validate_skill_template(skill: dict, errors: list[str]) -> None:
    required_sections = SKILL_REQUIRED_SECTIONS.get(skill["category"], [])
    skill_text = (REPO_ROOT / skill["path"]).read_text(encoding="utf-8")
    for section in required_sections:
        if section not in skill_text:
            errors.append(
                f"{skill['path']} is missing required section '{section}' for category '{skill['category']}'"
            )
    validate_must_read_budget(skill, skill_text, errors)


def validate_must_read_budget(skill: dict, skill_text: str, errors: list[str]) -> None:
    max_links = MAX_MUST_READ_LINKS_BY_CATEGORY.get(skill["category"])
    if max_links is None:
        return
    must_read_links = collect_must_read_links(skill_text)
    link_count = len(must_read_links)
    validate_deferred_must_read_links(skill, must_read_links, errors)
    if link_count > max_links:
        errors.append(
            f"{skill['path']} exceeds the Must Read budget for category "
            f"'{skill['category']}' ({link_count} > {max_links})"
        )


def validate_deferred_must_read_links(
    skill: dict, must_read_links: list[str], errors: list[str]
) -> None:
    skill_path = REPO_ROOT / skill["path"]
    for link_target in must_read_links:
        if "://" in link_target:
            continue
        normalized = normalize_skill_link(skill_path, link_target)
        if normalized is None:
            continue
        reason = DEFERRED_MUST_READ_PATHS.get(normalized)
        if reason:
            errors.append(
                f"{skill['path']} must not include '{normalized}' in Must Read: {reason}"
            )


def collect_must_read_links(skill_text: str) -> list[str]:
    in_section = False
    links: list[str] = []
    for line in skill_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if stripped == "## Must Read":
                in_section = True
                continue
            if in_section:
                break
        if in_section:
            links.extend(MARKDOWN_LINK_PATTERN.findall(line))
    return links


def count_must_read_links(skill_text: str) -> int:
    return len(collect_must_read_links(skill_text))


def normalize_skill_link(skill_path, link_target: str) -> str | None:
    try:
        resolved = (skill_path.parent / link_target).resolve()
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        normalized = PurePosixPath(link_target).as_posix()
        return normalized if not normalized.startswith("../") else None


def validate_skill_frontmatter(skill: dict, errors: list[str]) -> None:
    skill_text = (REPO_ROOT / skill["path"]).read_text(encoding="utf-8")
    match = SKILL_FRONTMATTER_PATTERN.match(skill_text)
    if match is None:
        errors.append(f"{skill['path']} is missing YAML frontmatter")
        return

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        errors.append(f"{skill['path']} has invalid YAML frontmatter: {exc}")
        return

    if not isinstance(frontmatter, dict):
        errors.append(f"{skill['path']} frontmatter must decode to a mapping")
        return

    if frontmatter.get("name") != skill["name"]:
        errors.append(
            f"{skill['path']} frontmatter name must match registry name '{skill['name']}'"
        )

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"{skill['path']} frontmatter is missing a non-empty description")
    elif "Use when" not in description:
        errors.append(
            f"{skill['path']} frontmatter description must include trigger guidance with 'Use when'"
        )

    expected_category = SKILL_CATEGORY_FRONTMATTER.get(
        skill["category"], skill["category"]
    )
    if frontmatter.get("category") != expected_category:
        errors.append(
            f"{skill['path']} frontmatter category must be '{expected_category}'"
        )


def validate_primary_skill(
    skill: dict, skill_lookup: dict[str, dict], skill_registry: dict, errors: list[str]
) -> None:
    skill_name = skill["name"]
    priority = skill.get("priority")
    if not isinstance(priority, int):
        errors.append(f"Primary skill '{skill_name}' is missing integer priority")
    auto_surface_fragments = skill.get("auto_surface_fragments")
    if auto_surface_fragments not in (None, True, False):
        errors.append(
            f"Primary skill '{skill_name}' has non-boolean auto_surface_fragments"
        )
    if not skill.get("fragments") and not auto_surface_fragments:
        errors.append(
            f"Primary skill '{skill_name}' has no fragment refs or auto-surface fragment resolution"
        )
    if not skill.get("required_surfaces"):
        errors.append(f"Primary skill '{skill_name}' has no required_surfaces")
    if not skill.get("stop_conditions"):
        errors.append(f"Primary skill '{skill_name}' has no stop_conditions")
    if not skill.get("acceptance_criteria"):
        errors.append(f"Primary skill '{skill_name}' has no acceptance_criteria")
    if not skill.get("selector_paths") and skill_name != "cross-stack-bugfix":
        errors.append(f"Primary skill '{skill_name}' has no selector_paths")
    for fragment in skill.get("fragments", []):
        fragment_skill = skill_lookup.get(fragment)
        if fragment_skill is None:
            errors.append(
                f"Primary skill '{skill_name}' references unknown fragment '{fragment}'"
            )
            continue
        if fragment_skill["category"] != "fragments":
            errors.append(
                f"Primary skill '{skill_name}' references non-fragment skill '{fragment}'"
            )
    validate_surface_refs(skill_name, skill, skill_registry, errors)


def validate_fragment_skill(
    skill: dict, skill_registry: dict, errors: list[str]
) -> None:
    skill_name = skill["name"]
    if not skill.get("owned_surfaces"):
        errors.append(f"Fragment skill '{skill_name}' has no owned_surfaces")
    if not skill.get("stop_conditions"):
        errors.append(f"Fragment skill '{skill_name}' has no stop_conditions")
    if not skill.get("acceptance_criteria"):
        errors.append(f"Fragment skill '{skill_name}' has no acceptance_criteria")
    validate_surface_refs(skill_name, skill, skill_registry, errors)


def validate_surface_refs(
    skill_name: str, skill: dict, skill_registry: dict, errors: list[str]
) -> None:
    known_surfaces = set(skill_registry["surfaces"])
    for field in (
        "required_surfaces",
        "conditional_surfaces",
        "optional_surfaces",
        "owned_surfaces",
    ):
        for surface in skill.get(field, []):
            if surface not in known_surfaces:
                errors.append(
                    f"Skill '{skill_name}' references unknown surface '{surface}' in {field}"
                )
