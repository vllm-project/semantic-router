#!/usr/bin/env python3
"""CI/workflow consistency validation for the shared agent harness."""

from __future__ import annotations

import yaml
from agent_support import REPO_ROOT


def validate_ci_changes_filters(
    repo_manifest: dict, task_matrix: dict, e2e_map: dict, errors: list[str]
) -> None:
    filters = load_ci_changes_filters(errors)
    if filters is None:
        return

    validate_ci_filter_against_task_rule("agent_text", task_matrix, filters, errors)
    validate_ci_filter_against_task_rule("agent_exec", task_matrix, filters, errors)
    validate_ci_profile_filters(e2e_map, filters, errors)
    validate_e2e_profile_lists(repo_manifest, e2e_map, errors)


def load_ci_changes_filters(errors: list[str]) -> dict | None:
    ci_changes_path = REPO_ROOT / ".github" / "workflows" / "ci-changes.yml"
    workflow = yaml.safe_load(ci_changes_path.read_text(encoding="utf-8"))
    steps = workflow.get("jobs", {}).get("filter", {}).get("steps", [])
    changes_step = next((step for step in steps if step.get("id") == "changes"), None)
    if changes_step is None:
        errors.append(".github/workflows/ci-changes.yml is missing the 'changes' step")
        return None

    filters_text = changes_step.get("with", {}).get("filters")
    if not filters_text:
        errors.append(".github/workflows/ci-changes.yml is missing with.filters")
        return None
    return yaml.safe_load(filters_text)


def validate_ci_filter_against_task_rule(
    rule_name: str, task_matrix: dict, filters: dict, errors: list[str]
) -> None:
    rule = next(
        (
            candidate
            for candidate in task_matrix["rules"]
            if candidate["name"] == rule_name
        ),
        None,
    )
    if rule is None:
        errors.append(f"task-matrix is missing rule '{rule_name}'")
        return

    workflow_paths = set(filters.get(rule_name, []))
    task_paths = set(rule.get("paths", []))
    if workflow_paths != task_paths:
        errors.append(
            f"ci-changes filter '{rule_name}' does not match task-matrix rule '{rule_name}'"
        )


def validate_ci_profile_filters(
    e2e_map: dict, filters: dict, errors: list[str]
) -> None:
    for profile, data in e2e_map.get("profile_rules", {}).items():
        filter_name = f"e2e_{profile.replace('-', '_')}"
        workflow_paths = set(filters.get(filter_name, []))
        profile_paths = set(data.get("paths", []))
        if workflow_paths != profile_paths:
            errors.append(
                f"ci-changes filter '{filter_name}' does not match e2e-profile-map rule '{profile}'"
            )


def validate_e2e_profile_lists(
    repo_manifest: dict, e2e_map: dict, errors: list[str]
) -> None:
    manifest_profiles = set(
        repo_manifest.get("validation", {}).get("e2e", {}).get("full_ci_profiles", [])
    )
    map_profiles = set(e2e_map.get("full_ci_profiles", []))
    if manifest_profiles != map_profiles:
        errors.append(
            "repo-manifest validation.e2e.full_ci_profiles does not match tools/agent/e2e-profile-map.yaml"
        )

    default_local_profile = (
        repo_manifest.get("validation", {}).get("e2e", {}).get("default_local_profile")
    )
    default_local_profiles = set(e2e_map.get("default_local_profiles", []))
    if default_local_profile not in default_local_profiles:
        errors.append(
            "repo-manifest validation.e2e.default_local_profile is missing from tools/agent/e2e-profile-map.yaml default_local_profiles"
        )
