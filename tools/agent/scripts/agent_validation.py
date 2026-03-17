#!/usr/bin/env python3
"""Harness contract validation for agent docs, manifests, and skills."""

from __future__ import annotations

from agent_ci_validation import validate_ci_changes_filters
from agent_context_validation import validate_context_map
from agent_doc_validation import (
    validate_agent_harness_layers,
    validate_support_files,
)
from agent_resolution import resolve_primary_skill
from agent_skill_validation import validate_skill_registry
from agent_support import (
    AGENT_DIR,
    REPO_ROOT,
    append_missing_make_target,
    build_skill_lookup,
    collect_make_targets,
    collect_manifest_globs,
    load_context_map,
    load_manifests,
    load_yaml,
    validate_glob,
)


def validate_supported_envs(
    repo_manifest: dict, skill_registry: dict, make_targets: set[str], errors: list[str]
) -> None:
    skill_lookup = build_skill_lookup(skill_registry)
    for env_name, env_data in repo_manifest["supported_envs"].items():
        for key in ("build_target", "serve_command"):
            append_missing_make_target(
                errors, f"{env_name}.{key}", env_data[key], make_targets
            )
        if not env_data.get("local_env", False):
            continue
        validate_local_env(env_name, env_data, skill_lookup, errors)


def validate_local_env(
    env_name: str,
    env_data: dict,
    skill_lookup: dict[str, dict],
    errors: list[str],
) -> None:
    smoke_config = env_data.get("smoke_config")
    if not smoke_config:
        errors.append(f"{env_name} is missing smoke_config")
    elif not (REPO_ROOT / smoke_config).exists():
        errors.append(f"{env_name} references missing smoke_config '{smoke_config}'")

    fragment = env_data.get("local_dev_fragment")
    if not fragment:
        errors.append(f"{env_name} is missing local_dev_fragment")
    elif fragment not in skill_lookup:
        errors.append(f"{env_name} references unknown fragment skill '{fragment}'")

    for reference_key in ("deployment_reference", "example_config"):
        reference_path = env_data.get(reference_key)
        if reference_path and not (REPO_ROOT / reference_path).exists():
            errors.append(
                f"{env_name} references missing {reference_key} '{reference_path}'"
            )


def validate_subsystems(repo_manifest: dict, errors: list[str]) -> None:
    for subsystem in repo_manifest["subsystems"]:
        for entrypoint in subsystem["entrypoints"]:
            if not (REPO_ROOT / entrypoint).exists():
                errors.append(f"Missing subsystem entrypoint: {entrypoint}")


def validate_e2e_map(e2e_map: dict, errors: list[str]) -> None:
    for profile, data in e2e_map["profile_rules"].items():
        if not data["paths"]:
            errors.append(f"E2E profile '{profile}' has no path patterns")


def validate_task_matrix(
    task_matrix: dict, make_targets: set[str], errors: list[str]
) -> None:
    for rule in task_matrix["rules"]:
        commands = [*rule.get("fast_tests", []), *rule.get("feature_tests", [])]
        for command in commands:
            append_missing_make_target(
                errors, f"Task rule '{rule['name']}'", command, make_targets
            )


def validate_manifest_globs(
    repo_manifest: dict,
    task_matrix: dict,
    e2e_map: dict,
    structure_rules: dict,
    skill_registry: dict,
    errors: list[str],
) -> None:
    patterns = dict.fromkeys(
        collect_manifest_globs(
            repo_manifest, task_matrix, e2e_map, structure_rules, skill_registry
        )
    )
    for pattern in patterns:
        if not validate_glob(pattern):
            errors.append(f"Pattern has no matches: {pattern}")


def validate_routing_fixtures(errors: list[str]) -> None:
    fixtures_path = AGENT_DIR / "routing-fixtures.yaml"
    if not fixtures_path.exists():
        errors.append(
            "Missing routing fixtures file: tools/agent/routing-fixtures.yaml"
        )
        return

    fixtures = load_yaml(fixtures_path)
    for case in fixtures.get("cases", []):
        case_name = case.get("name", "<unnamed>")
        changed_files = case.get("changed_files", [])
        expected_primary_skill = case.get("expected_primary_skill")

        if not changed_files:
            errors.append(f"Routing fixture '{case_name}' has no changed_files")
            continue
        if not expected_primary_skill:
            errors.append(
                f"Routing fixture '{case_name}' is missing expected_primary_skill"
            )
            continue

        missing_files = [
            path for path in changed_files if not (REPO_ROOT / path).exists()
        ]
        if missing_files:
            errors.append(
                f"Routing fixture '{case_name}' references missing files: "
                + ", ".join(missing_files)
            )
            continue

        resolved_primary_skill = resolve_primary_skill(changed_files)["name"]
        if resolved_primary_skill != expected_primary_skill:
            errors.append(
                f"Routing fixture '{case_name}' resolved '{resolved_primary_skill}' "
                f"instead of '{expected_primary_skill}'"
            )


def validate_discovery_bridge(errors: list[str]) -> None:
    bridge_path = REPO_ROOT / ".agents" / "skills" / "harness" / "SKILL.md"
    if not bridge_path.exists():
        errors.append(
            "Missing native-discovery bridge: .agents/skills/harness/SKILL.md"
        )
        return

    bridge_text = bridge_path.read_text(encoding="utf-8")
    required_refs = [
        "AGENTS.md",
        "docs/agent/README.md",
        "make agent-report",
        "tools/agent/skills/",
    ]
    for ref in required_refs:
        if ref not in bridge_text:
            errors.append(
                f"Discovery bridge skill must reference '{ref}' in {bridge_path.relative_to(REPO_ROOT)}"
            )


def collect_validation_errors() -> list[str]:
    repo_manifest, task_matrix, e2e_map, structure_rules, skill_registry = (
        load_manifests()
    )
    context_map = load_context_map()
    make_targets = collect_make_targets()
    errors: list[str] = []

    validate_supported_envs(repo_manifest, skill_registry, make_targets, errors)
    validate_subsystems(repo_manifest, errors)
    validate_e2e_map(e2e_map, errors)
    validate_task_matrix(task_matrix, make_targets, errors)
    validate_manifest_globs(
        repo_manifest, task_matrix, e2e_map, structure_rules, skill_registry, errors
    )
    validate_ci_changes_filters(repo_manifest, task_matrix, e2e_map, errors)
    validate_support_files(repo_manifest, errors)
    validate_agent_harness_layers(repo_manifest, task_matrix, skill_registry, errors)
    validate_context_map(
        repo_manifest, task_matrix, skill_registry, context_map, errors
    )
    validate_skill_registry(repo_manifest, task_matrix, skill_registry, errors)
    validate_routing_fixtures(errors)
    validate_discovery_bridge(errors)
    return errors


def validate_manifests() -> int:
    errors = collect_validation_errors()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("Agent manifests validated successfully.")
    return 0
