#!/usr/bin/env python3
"""Validation for the executable harness context map."""

from __future__ import annotations

from agent_support import REPO_ROOT

REQUIRED_CONTEXT_MAP_DEFAULT_SECTIONS = {"start_here", "resume_refs"}
REQUIRED_CONTEXT_MAP_PATHS = {
    "defaults.start_here": {"AGENTS.md", "docs/agent/README.md"},
}


def validate_context_map(
    repo_manifest: dict,
    task_matrix: dict,
    skill_registry: dict,
    context_map: dict,
    errors: list[str],
) -> None:
    validate_context_map_version(context_map, errors)
    validate_context_map_defaults(context_map, errors)
    validate_context_map_known_rules(task_matrix, context_map, errors)
    validate_context_map_known_surfaces(skill_registry, context_map, errors)
    validate_context_map_known_skills(skill_registry, context_map, errors)
    validate_context_map_known_envs(repo_manifest, context_map, errors)
    validate_context_map_ref_groups(context_map, errors)


def validate_context_map_version(context_map: dict, errors: list[str]) -> None:
    if context_map.get("version") != 1:
        errors.append("tools/agent/context-map.yaml must declare version: 1")


def validate_context_map_defaults(context_map: dict, errors: list[str]) -> None:
    defaults = context_map.get("defaults", {})
    for section in REQUIRED_CONTEXT_MAP_DEFAULT_SECTIONS:
        if not defaults.get(section):
            errors.append(
                f"tools/agent/context-map.yaml defaults is missing '{section}' references"
            )
    for section, required_paths in REQUIRED_CONTEXT_MAP_PATHS.items():
        validate_required_default_paths(defaults, section, required_paths, errors)


def validate_required_default_paths(
    defaults: dict, section: str, required_paths: set[str], errors: list[str]
) -> None:
    refs = collect_context_map_paths(defaults.get(section.split(".", 1)[1], []))
    missing = required_paths - refs
    if missing:
        errors.append(
            f"tools/agent/context-map.yaml {section} is missing required paths: "
            + ", ".join(sorted(missing))
        )


def validate_context_map_known_rules(
    task_matrix: dict, context_map: dict, errors: list[str]
) -> None:
    known_rule_names = {rule["name"] for rule in task_matrix.get("rules", [])}
    validate_known_mapping_keys(
        context_map.get("rules", {}),
        known_rule_names,
        "task rule",
        errors,
    )


def validate_context_map_known_surfaces(
    skill_registry: dict, context_map: dict, errors: list[str]
) -> None:
    known_surface_names = set(skill_registry.get("surfaces", {}))
    mapped_surfaces = set(context_map.get("surfaces", {}))
    missing_surfaces = known_surface_names - mapped_surfaces
    if missing_surfaces:
        errors.append(
            "tools/agent/context-map.yaml is missing surface mappings for: "
            + ", ".join(sorted(missing_surfaces))
        )
    validate_known_mapping_keys(
        context_map.get("surfaces", {}),
        known_surface_names,
        "surface",
        errors,
    )


def validate_context_map_known_skills(
    skill_registry: dict, context_map: dict, errors: list[str]
) -> None:
    known_skill_names = {
        skill["name"]
        for category in skill_registry.get("skills", {}).values()
        for skill in category
    }
    validate_known_mapping_keys(
        context_map.get("skills", {}),
        known_skill_names,
        "skill",
        errors,
    )


def validate_context_map_known_envs(
    repo_manifest: dict, context_map: dict, errors: list[str]
) -> None:
    known_env_names = set(repo_manifest.get("supported_envs", {}))
    validate_known_mapping_keys(
        context_map.get("envs", {}),
        known_env_names,
        "env",
        errors,
    )


def validate_known_mapping_keys(
    mapping: dict,
    known_names: set[str],
    label: str,
    errors: list[str],
) -> None:
    for name in mapping:
        if name not in known_names:
            errors.append(
                f"tools/agent/context-map.yaml references unknown {label} '{name}'"
            )


def validate_context_map_ref_groups(context_map: dict, errors: list[str]) -> None:
    validate_context_map_ref_list(
        context_map.get("defaults", {}).get("start_here", []),
        "defaults.start_here",
        errors,
    )
    validate_context_map_ref_list(
        context_map.get("defaults", {}).get("resume_refs", []),
        "defaults.resume_refs",
        errors,
    )
    validate_named_ref_group(context_map.get("rules", {}), "rules", errors)
    validate_named_ref_group(context_map.get("surfaces", {}), "surfaces", errors)
    validate_named_ref_group(context_map.get("envs", {}), "envs", errors)
    validate_skill_resume_refs(context_map.get("skills", {}), errors)


def validate_named_ref_group(group: dict, label: str, errors: list[str]) -> None:
    for entry_name, refs in group.items():
        validate_context_map_ref_list(refs, f"{label}.{entry_name}", errors)


def validate_skill_resume_refs(skills: dict, errors: list[str]) -> None:
    for skill_name, data in skills.items():
        validate_context_map_ref_list(
            data.get("resume_refs", []),
            f"skills.{skill_name}.resume_refs",
            errors,
        )


def collect_context_map_paths(refs: list[dict]) -> set[str]:
    return {ref["path"] for ref in refs if "path" in ref}


def validate_context_map_ref_list(
    refs: list[dict], label: str, errors: list[str]
) -> None:
    if not isinstance(refs, list):
        errors.append(f"tools/agent/context-map.yaml {label} must be a list")
        return
    for ref in refs:
        if not isinstance(ref, dict):
            errors.append(f"tools/agent/context-map.yaml {label} entries must be maps")
            continue
        path = ref.get("path")
        reason = ref.get("reason")
        if not path:
            errors.append(
                f"tools/agent/context-map.yaml {label} has an entry without path"
            )
            continue
        if not reason:
            errors.append(
                f"tools/agent/context-map.yaml {label} entry '{path}' is missing reason"
            )
        if not (REPO_ROOT / path).exists():
            errors.append(
                f"tools/agent/context-map.yaml {label} references missing path '{path}'"
            )
