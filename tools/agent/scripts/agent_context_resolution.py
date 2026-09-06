#!/usr/bin/env python3
"""Task-context, rule, and surface resolution for the agent harness."""

from __future__ import annotations

import fnmatch
import sys
from collections.abc import Iterable

from agent_models import ResolvedContext
from agent_support import REPO_ROOT, load_manifests

TOOLS_CI_DIR = REPO_ROOT / "tools" / "ci"
if str(TOOLS_CI_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_CI_DIR))

from classify_pr_changes import classify as classify_pr_changes  # noqa: E402
from test_domain_registry import (  # noqa: E402
    domain_records,
    profile_records,
    suite_domains,
)


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def local_agent_rule_paths(repo_manifest: dict) -> set[str]:
    return {entry["path"] for entry in repo_manifest.get("local_agent_rules", [])}


def matches_with_local_rule_policy(
    path: str,
    patterns: Iterable[str],
    local_rule_paths: set[str],
    *,
    allow_local_rules: bool = False,
) -> bool:
    if path in local_rule_paths and not allow_local_rules:
        return False
    return matches_any(path, patterns)


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


DEFAULT_LOOP_MODE = "completion"
DEFAULT_EXECUTION_PLAN_POLICY = "long_horizon"
LOOP_MODE_PRIORITY = {
    "lightweight": 0,
    "completion": 1,
}
EXECUTION_PLAN_POLICY_PRIORITY = {
    "none": 0,
    "long_horizon": 1,
    "always": 2,
}


def match_task_matrix_rules(
    task_matrix: dict, changed_files: list[str], local_rule_path_set: set[str]
) -> tuple[list[dict], list[str], list[str], bool]:
    matched_rules: list[dict] = []
    fast_tests: list[str] = []
    feature_tests: list[str] = []
    requires_local_smoke = False

    for rule in task_matrix["rules"]:
        domain = domain_records().get(str(rule.get("domain") or ""))
        paths = domain.get("paths", []) if domain else rule["paths"]
        local = domain.get("local", {}) if domain else {}
        rule_fast_tests = local.get("fast", rule.get("fast_tests", []))
        rule_feature_tests = local.get("feature", rule.get("feature_tests", []))
        allow_local_rules = rule["name"] == "agent_text"
        if any(
            matches_with_local_rule_policy(
                path,
                paths,
                local_rule_path_set,
                allow_local_rules=allow_local_rules,
            )
            for path in changed_files
        ):
            matched_rules.append(rule)
            fast_tests.extend(rule_fast_tests)
            feature_tests.extend(rule_feature_tests)
            requires_local_smoke = requires_local_smoke or rule.get(
                "requires_local_smoke", False
            )

    return matched_rules, fast_tests, feature_tests, requires_local_smoke


def targeted_profiles_for_rules(
    profile_rules: dict, changed_files: list[str], local_rule_path_set: set[str]
) -> list[str]:
    return [
        profile
        for profile, data in profile_rules.items()
        if any(
            matches_with_local_rule_policy(path, data["paths"], local_rule_path_set)
            for path in changed_files
        )
    ]


def resolve_e2e_profiles(
    changed_files: list[str],
    test_domain_registry: dict,
    local_rule_path_set: set[str],
) -> tuple[list[str], list[str], list[str], str]:
    profiles = profile_records(test_domain_registry)
    standard_profile_rules = {
        name: data for name, data in profiles.items() if data.get("selection") == "pr"
    }
    manual_profile_rules = {
        name: data
        for name, data in profiles.items()
        if data.get("selection") == "manual"
    }
    workflow_suite_rules = {
        suite_name: domain
        for suite_name, (_, domain) in suite_domains(test_domain_registry).items()
    }
    targeted_profiles = targeted_profiles_for_rules(
        standard_profile_rules, changed_files, local_rule_path_set
    )
    targeted_manual_profiles = targeted_profiles_for_rules(
        manual_profile_rules, changed_files, local_rule_path_set
    )
    targeted_workflow_suites = targeted_profiles_for_rules(
        workflow_suite_rules, changed_files, local_rule_path_set
    )
    hosted_profiles = list(classify_pr_changes(changed_files).profiles)
    if targeted_profiles or targeted_manual_profiles or hosted_profiles:
        local_profiles = [
            *(targeted_profiles or hosted_profiles),
            *targeted_manual_profiles,
        ]
        ci_profiles = hosted_profiles
        ci_mode = "targeted" if hosted_profiles else "none"
    else:
        local_profiles = []
        ci_profiles = []
        ci_mode = "none"

    return (
        local_profiles,
        ci_profiles,
        targeted_workflow_suites,
        ci_mode,
    )


def is_doc_only_rule_set(matched_rules: list[dict]) -> bool:
    return bool(matched_rules) and all(
        rule.get("doc_only", False) for rule in matched_rules
    )


def resolve_loop_mode(matched_rules: list[dict]) -> str:
    if not matched_rules:
        return DEFAULT_LOOP_MODE
    return max(
        (rule.get("loop_mode", DEFAULT_LOOP_MODE) for rule in matched_rules),
        key=lambda mode: LOOP_MODE_PRIORITY.get(mode, -1),
    )


def resolve_execution_plan_policy(matched_rules: list[dict]) -> str:
    if not matched_rules:
        return DEFAULT_EXECUTION_PLAN_POLICY
    return max(
        (
            rule.get("execution_plan_policy", DEFAULT_EXECUTION_PLAN_POLICY)
            for rule in matched_rules
        ),
        key=lambda policy: EXECUTION_PLAN_POLICY_PRIORITY.get(policy, -1),
    )


def resolve_context(changed_files: list[str]) -> ResolvedContext:
    repo_manifest, task_matrix, test_domain_registry, _, _ = load_manifests()
    local_rule_path_set = local_agent_rule_paths(repo_manifest)
    matched_rules, fast_tests, feature_tests, requires_local_smoke = (
        match_task_matrix_rules(task_matrix, changed_files, local_rule_path_set)
    )
    local_profiles, ci_profiles, targeted_workflow_suites, ci_mode = (
        resolve_e2e_profiles(
            changed_files,
            test_domain_registry,
            local_rule_path_set,
        )
    )
    doc_only = is_doc_only_rule_set(matched_rules)
    loop_mode = resolve_loop_mode(matched_rules)
    execution_plan_policy = resolve_execution_plan_policy(matched_rules)
    classification = classify_pr_changes(changed_files)
    selected_jobs = list(classification.selected_jobs)
    expected_artifacts = unique_preserve_order(
        str(artifact)
        for domain in domain_records().values()
        if domain.get("pr_job") in selected_jobs
        for artifact in [domain.get("reporting", {}).get("artifact")]
        if artifact
    )
    if doc_only:
        local_profiles = []
    return ResolvedContext(
        changed_files=changed_files,
        matched_rules=[rule["name"] for rule in matched_rules],
        fast_tests=unique_preserve_order(fast_tests),
        feature_tests=unique_preserve_order(feature_tests),
        requires_local_smoke=requires_local_smoke and not doc_only,
        local_e2e_profiles=unique_preserve_order(local_profiles),
        ci_e2e_profiles=unique_preserve_order(ci_profiles),
        workflow_integration_suites=unique_preserve_order(targeted_workflow_suites),
        ci_test_domains=selected_jobs,
        expected_artifacts=expected_artifacts,
        ci_e2e_mode=ci_mode,
        doc_only=doc_only,
        loop_mode=loop_mode,
        execution_plan_policy=execution_plan_policy,
    )


def resolve_impacted_surfaces(changed_files: list[str]) -> list[str]:
    repo_manifest, _, _, _, skill_registry = load_manifests()
    local_rule_path_set = local_agent_rule_paths(repo_manifest)
    impacted = [
        surface_name
        for surface_name, surface in skill_registry["surfaces"].items()
        if (
            surface_name == "harness_docs"
            and any(path in local_rule_path_set for path in changed_files)
        )
        or any(
            matches_with_local_rule_policy(
                path,
                surface["paths"],
                local_rule_path_set,
                allow_local_rules=surface_name == "harness_docs",
            )
            for path in changed_files
        )
    ]
    return unique_preserve_order(impacted)
