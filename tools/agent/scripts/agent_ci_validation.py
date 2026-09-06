#!/usr/bin/env python3
"""CI and E2E registry consistency validation for the agent harness."""

from __future__ import annotations

import re
import sys

import yaml
from agent_support import REPO_ROOT

TOOLS_CI_DIR = REPO_ROOT / "tools" / "ci"
if str(TOOLS_CI_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_CI_DIR))

from test_domain_registry import (  # noqa: E402
    profile_records,
    registry_schema_errors,
    suite_domains,
)


def validate_ci_changes_filters(
    repo_manifest: dict,
    task_matrix: dict,
    test_domain_registry: dict,
    errors: list[str],
) -> None:
    errors.extend(registry_schema_errors(test_domain_registry))
    validate_registry_task_rules(task_matrix, test_domain_registry, errors)
    validate_e2e_profile_lists(repo_manifest, test_domain_registry, errors)
    validate_e2e_inventory(test_domain_registry, errors)
    validate_workflow_suites(test_domain_registry, errors)


def validate_registry_task_rules(
    task_matrix: dict, test_domain_registry: dict, errors: list[str]
) -> None:
    rules = {rule["name"]: rule for rule in task_matrix.get("rules", [])}
    for domain_name, domain in suite_domains(test_domain_registry).values():
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


def validate_e2e_profile_lists(
    repo_manifest: dict, test_domain_registry: dict, errors: list[str]
) -> None:
    profiles = profile_records(test_domain_registry)
    standard_profiles = {
        name for name, data in profiles.items() if data.get("selection") == "pr"
    }
    full_ci_profiles = {name for name, data in profiles.items() if data.get("full_ci")}
    default_local_profiles = {
        name for name, data in profiles.items() if data.get("default_local")
    }
    manifest_e2e = repo_manifest.get("validation", {}).get("e2e", {})
    manifest_full_ci = set(manifest_e2e.get("full_ci_profiles", []))
    if manifest_full_ci != full_ci_profiles:
        errors.append(
            "repo-manifest validation.e2e.full_ci_profiles does not match "
            "tools/agent/test-domain-registry.yaml"
        )
    if not full_ci_profiles.issubset(standard_profiles):
        errors.append("full-CI E2E profiles must use selection: pr")

    default_local_profile = manifest_e2e.get("default_local_profile")
    if default_local_profile not in default_local_profiles:
        errors.append(
            "repo-manifest validation.e2e.default_local_profile is not marked "
            "default_local in tools/agent/test-domain-registry.yaml"
        )
    if not default_local_profiles.issubset(standard_profiles):
        errors.append("default-local E2E profiles must use selection: pr")


def validate_e2e_inventory(test_domain_registry: dict, errors: list[str]) -> None:
    mapped_profiles = set(profile_records(test_domain_registry))
    runnable_profiles = parse_runnable_profiles(errors)
    readme_profiles = parse_readme_profiles(errors)

    if runnable_profiles is not None and runnable_profiles != mapped_profiles:
        errors.append(
            "e2e/profiles/all/imports.go profile inventory does not match "
            "tools/agent/test-domain-registry.yaml"
            + format_inventory_diff(mapped_profiles, runnable_profiles)
        )
    if readme_profiles is not None and readme_profiles != mapped_profiles:
        errors.append(
            "e2e/README.md supported profile inventory does not match "
            "tools/agent/test-domain-registry.yaml"
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
            "e2e/README.md is missing the Supported Profiles / Coverage Ownership "
            "Matrix sections"
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
    return " (" + ";".join(parts) + ")" if parts else ""


def validate_workflow_suites(test_domain_registry: dict, errors: list[str]) -> None:
    for suite_name, (_, domain) in suite_domains(test_domain_registry).items():
        feature_commands = domain.get("local", {}).get("feature", [])
        if len(feature_commands) != 1:
            errors.append(
                f"workflow suite {suite_name!r} must declare one local feature command"
            )
        workflow_path = domain.get("workflow")
        if workflow_path:
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
