#!/usr/bin/env python3
"""CI/workflow consistency validation for the shared agent harness."""

from __future__ import annotations

import re
import sys

import yaml
from agent_support import REPO_ROOT

TOOLS_CI_DIR = REPO_ROOT / "tools" / "ci"
if str(TOOLS_CI_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_CI_DIR))

from test_domain_registry import (  # noqa: E402
    local_full_ci_paths,
    profile_records,
    registry_schema_errors,
    suite_domains,
)


def validate_ci_changes_filters(
    repo_manifest: dict, task_matrix: dict, e2e_map: dict, errors: list[str]
) -> None:
    validate_test_domain_registry(task_matrix, e2e_map, errors)
    validate_e2e_profile_lists(repo_manifest, e2e_map, errors)
    validate_e2e_inventory(e2e_map, errors)
    validate_workflow_suite_rules(e2e_map, errors)


def validate_test_domain_registry(
    task_matrix: dict, e2e_map: dict, errors: list[str]
) -> None:
    errors.extend(registry_schema_errors())
    registry_profiles = profile_records()
    registry_full = [
        name for name, data in registry_profiles.items() if data.get("full_ci")
    ]
    registry_default = [
        name for name, data in registry_profiles.items() if data.get("default_local")
    ]
    if registry_full != e2e_map.get("full_ci_profiles", []):
        errors.append(
            "test-domain registry full CI profiles do not match e2e-profile-map"
        )
    if registry_default != e2e_map.get("default_local_profiles", []):
        errors.append(
            "test-domain registry default local profiles do not match e2e-profile-map"
        )
    if tuple(e2e_map.get("full_ci_triggers", [])) != local_full_ci_paths():
        errors.append(
            "test-domain registry local full-CI paths do not match e2e-profile-map"
        )
    sections = {"pr": "profile_rules", "manual": "manual_profile_rules"}
    for selection, section in sections.items():
        expected = {
            name: tuple(data.get("paths", []))
            for name, data in registry_profiles.items()
            if data.get("selection") == selection
        }
        actual = {
            name: tuple(data.get("paths", []))
            for name, data in e2e_map.get(section, {}).items()
        }
        if expected != actual:
            errors.append(
                "tools/agent/test-domain-registry.yaml profile paths do not "
                f"match tools/agent/e2e-profile-map.yaml {section}"
            )
    validate_registry_workflow_suites(e2e_map, errors)
    validate_registry_task_rules(task_matrix, errors)


def validate_registry_task_rules(task_matrix: dict, errors: list[str]) -> None:
    rules = {rule["name"]: rule for rule in task_matrix.get("rules", [])}
    for domain_name, domain in suite_domains().values():
        task_rule = domain.get("task_rule")
        if not task_rule:
            continue
        rule = rules.get(task_rule)
        if rule is None or rule.get("domain") != domain_name:
            errors.append(
                f"task-matrix rule {task_rule!r} must reference registry domain "
                f"{domain_name!r}"
            )
            continue
        if rule.get("paths") or rule.get("fast_tests") or rule.get("feature_tests"):
            errors.append(
                f"task-matrix rule {task_rule!r} duplicates registry paths or commands"
            )


def validate_registry_workflow_suites(e2e_map: dict, errors: list[str]) -> None:
    registry_suites = suite_domains()
    map_suites = e2e_map.get("workflow_suite_rules", {})
    if set(registry_suites) != set(map_suites):
        errors.append(
            "test-domain registry workflow suites do not match "
            "e2e-profile-map workflow_suite_rules"
        )
        return
    for suite_name, (_, domain) in registry_suites.items():
        mapped = map_suites[suite_name]
        expected = {
            "workflow": domain["workflow"],
            "change_filter": domain["selector"],
            "local_command": domain["local"]["feature"][0],
            "paths": frozenset(domain.get("paths", [])),
        }
        actual = {
            "workflow": mapped.get("workflow"),
            "change_filter": mapped.get("change_filter"),
            "local_command": mapped.get("local_command"),
            "paths": frozenset(mapped.get("paths", [])),
        }
        if expected != actual:
            errors.append(
                f"test-domain registry suite {suite_name!r} does not match "
                "e2e-profile-map workflow suite metadata"
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
    return set(re.findall(r'register\(\s*"([a-z0-9-]+)"', text))


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
        for field in (
            "owner",
            "kind",
            "summary",
            "workflow",
            "change_filter",
            "local_command",
        ):
            if not data.get(field):
                errors.append(
                    f"tools/agent/e2e-profile-map.yaml workflow suite '{suite_name}' is missing '{field}'"
                )
        workflow_path = data.get("workflow")
        if not workflow_path:
            continue
        validate_reusable_workflow(workflow_path, errors)


def validate_reusable_workflow(workflow_path: str, errors: list[str]) -> bool:
    path = REPO_ROOT / workflow_path
    if not path.exists():
        errors.append(f"Missing workflow '{workflow_path}'")
        return False
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    workflow_on = workflow.get("on", workflow.get(True, {}))
    if "workflow_call" not in workflow_on:
        errors.append(f"Workflow '{workflow_path}' is missing workflow_call")
        return False
    return True
