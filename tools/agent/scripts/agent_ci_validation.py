#!/usr/bin/env python3
"""CI/workflow consistency validation for the shared agent harness."""

from __future__ import annotations

import re

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
    validate_e2e_inventory(e2e_map, errors)
    validate_workflow_suite_rules(e2e_map, errors)


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
    standard_profiles = set(e2e_map.get("profile_rules", {}))
    manual_profiles = set(e2e_map.get("manual_profile_rules", {}))
    if standard_profiles & manual_profiles:
        errors.append(
            "tools/agent/e2e-profile-map.yaml reuses profile names across profile_rules and manual_profile_rules"
        )

    manifest_profiles = set(
        repo_manifest.get("validation", {}).get("e2e", {}).get("full_ci_profiles", [])
    )
    map_profiles = set(e2e_map.get("full_ci_profiles", []))
    if manifest_profiles != map_profiles:
        errors.append(
            "repo-manifest validation.e2e.full_ci_profiles does not match tools/agent/e2e-profile-map.yaml"
        )
    if not map_profiles.issubset(standard_profiles):
        errors.append(
            "tools/agent/e2e-profile-map.yaml full_ci_profiles must be a subset of profile_rules"
        )

    default_local_profile = (
        repo_manifest.get("validation", {}).get("e2e", {}).get("default_local_profile")
    )
    default_local_profiles = set(e2e_map.get("default_local_profiles", []))
    if default_local_profile not in default_local_profiles:
        errors.append(
            "repo-manifest validation.e2e.default_local_profile is missing from tools/agent/e2e-profile-map.yaml default_local_profiles"
        )
    if not default_local_profiles.issubset(standard_profiles):
        errors.append(
            "tools/agent/e2e-profile-map.yaml default_local_profiles must be a subset of profile_rules"
        )

    for profile_name, data in combined_profile_rules(e2e_map).items():
        for field in ("owner", "coverage_role", "selection_mode", "summary"):
            if not data.get(field):
                errors.append(
                    f"tools/agent/e2e-profile-map.yaml profile '{profile_name}' is missing '{field}'"
                )


def combined_profile_rules(e2e_map: dict) -> dict[str, dict]:
    combined = dict(e2e_map.get("profile_rules", {}))
    combined.update(e2e_map.get("manual_profile_rules", {}))
    return combined


def validate_e2e_inventory(e2e_map: dict, errors: list[str]) -> None:
    mapped_profiles = set(combined_profile_rules(e2e_map))
    runnable_profiles = parse_runnable_profiles(errors)
    readme_profiles = parse_readme_profiles(errors)

    if runnable_profiles is not None and runnable_profiles != mapped_profiles:
        errors.append(
            "e2e/profiles/all/imports.go profile inventory does not match tools/agent/e2e-profile-map.yaml"
            + format_inventory_diff(mapped_profiles, runnable_profiles)
        )
    if readme_profiles is not None and readme_profiles != mapped_profiles:
        errors.append(
            "e2e/README.md supported profile inventory does not match tools/agent/e2e-profile-map.yaml"
            + format_inventory_diff(mapped_profiles, readme_profiles)
        )


def parse_runnable_profiles(errors: list[str]) -> set[str] | None:
    imports_path = REPO_ROOT / "e2e" / "profiles" / "all" / "imports.go"
    if not imports_path.exists():
        errors.append("Missing e2e/profiles/all/imports.go")
        return None
    text = imports_path.read_text(encoding="utf-8")
    return set(
        re.findall(
            r'github\.com/vllm-project/semantic-router/e2e/profiles/([a-z0-9-]+)"',
            text,
        )
    )


def parse_readme_profiles(errors: list[str]) -> set[str] | None:
    readme_path = REPO_ROOT / "e2e" / "README.md"
    if not readme_path.exists():
        errors.append("Missing e2e/README.md")
        return None
    text = readme_path.read_text(encoding="utf-8")
    match = re.search(
        r"### Supported Profiles\s+(.*?)\s+### Coverage Ownership Matrix",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        errors.append(
            "e2e/README.md is missing the Supported Profiles / Coverage Ownership Matrix sections"
        )
        return None
    return set(
        re.findall(r"^- \*\*([a-z0-9-]+)\*\*:", match.group(1), flags=re.MULTILINE)
    )


def format_inventory_diff(expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    parts: list[str] = []
    if missing:
        parts.append(" missing: " + ", ".join(missing))
    if extra:
        parts.append(" extra: " + ", ".join(extra))
    if not parts:
        return ""
    return " (" + ";".join(parts) + ")"


def validate_workflow_suite_rules(e2e_map: dict, errors: list[str]) -> None:
    for suite_name, data in e2e_map.get("workflow_suite_rules", {}).items():
        for field in ("owner", "kind", "summary", "workflow", "local_command"):
            if not data.get(field):
                errors.append(
                    f"tools/agent/e2e-profile-map.yaml workflow suite '{suite_name}' is missing '{field}'"
                )
        workflow_path = data.get("workflow")
        if not workflow_path:
            continue
        workflow_paths = load_workflow_trigger_paths(workflow_path, errors)
        if workflow_paths is None:
            continue
        suite_paths = set(data.get("paths", []))
        if workflow_paths != suite_paths:
            errors.append(
                f"workflow suite '{suite_name}' paths do not match {workflow_path}"
            )


def load_workflow_trigger_paths(
    workflow_path: str, errors: list[str]
) -> set[str] | None:
    path = REPO_ROOT / workflow_path
    if not path.exists():
        errors.append(f"Missing workflow '{workflow_path}'")
        return None
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    workflow_on = workflow.get("on", workflow.get(True, {}))
    pull_request = workflow_on.get("pull_request", {})
    if not isinstance(pull_request, dict):
        errors.append(f"Workflow '{workflow_path}' is missing pull_request.paths")
        return None
    paths = pull_request.get("paths")
    if not paths:
        errors.append(f"Workflow '{workflow_path}' is missing pull_request.paths")
        return None
    return set(paths)
