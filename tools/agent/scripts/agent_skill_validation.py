#!/usr/bin/env python3
"""Skill and surface validation for the shared agent harness."""

from __future__ import annotations

from agent_support import (
    CHANGE_SURFACES_DOC,
    REPO_ROOT,
    build_skill_lookup,
    collect_task_rule_names,
    iter_registry_skills,
)

SKILL_REQUIRED_SECTIONS = {
    "primary": ["## Trigger", "## Must Read", "## Standard Commands", "## Acceptance"],
    "fragments": [
        "## Trigger",
        "## Must Read",
        "## Standard Commands",
        "## Acceptance",
    ],
    "support": ["## Trigger", "## Must Read", "## Standard Commands", "## Acceptance"],
    "legacy_reference": [
        "## Trigger",
        "## Must Read",
        "## Standard Commands",
        "## Acceptance",
    ],
}


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


def validate_primary_skill(
    skill: dict, skill_lookup: dict[str, dict], skill_registry: dict, errors: list[str]
) -> None:
    skill_name = skill["name"]
    priority = skill.get("priority")
    if not isinstance(priority, int):
        errors.append(f"Primary skill '{skill_name}' is missing integer priority")
    if not skill.get("fragments"):
        errors.append(f"Primary skill '{skill_name}' has no fragment refs")
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
