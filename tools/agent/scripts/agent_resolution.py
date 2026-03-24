#!/usr/bin/env python3
"""Change matching, skill resolution, and report assembly for the agent gate."""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
from collections.abc import Iterable

from agent_context_pack import build_context_pack
from agent_models import (
    AgentReport,
    EnvironmentResolution,
    ResolvedContext,
    SkillResolution,
)
from agent_support import (
    REPO_ROOT,
    build_skill_lookup,
    load_manifests,
    resolve_env_data,
)


def split_changed_files(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[\n,]+", raw)
    cleaned = [normalize_changed_path(part) for part in parts if part.strip()]
    return sorted(dict.fromkeys(cleaned))


def normalize_changed_path(raw_path: str) -> str:
    path = raw_path.strip()
    while path.startswith("./"):
        path = path[2:]
    return path


def git_changed_files(base_ref: str | None) -> list[str]:
    if base_ref is None:
        base_ref = os.getenv("AGENT_BASE_REF", "origin/main")

    def run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    merge_base = None
    if run_git("rev-parse", "--verify", base_ref).returncode == 0:
        result = run_git("merge-base", "HEAD", base_ref)
        if result.returncode == 0:
            merge_base = result.stdout.strip()

    if not merge_base:
        result = run_git("rev-parse", "--verify", "HEAD^")
        if result.returncode == 0:
            merge_base = "HEAD^"

    if not merge_base:
        return []

    result = run_git("diff", "--name-only", f"{merge_base}...HEAD")
    if result.returncode != 0:
        return []

    return split_changed_files(result.stdout)


def get_changed_files(explicit: str | None, base_ref: str | None) -> list[str]:
    files = split_changed_files(explicit)
    if files:
        return files
    return git_changed_files(base_ref)


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
        allow_local_rules = rule["name"] == "agent_text"
        if any(
            matches_with_local_rule_policy(
                path,
                rule["paths"],
                local_rule_path_set,
                allow_local_rules=allow_local_rules,
            )
            for path in changed_files
        ):
            matched_rules.append(rule)
            fast_tests.extend(rule.get("fast_tests", []))
            feature_tests.extend(rule.get("feature_tests", []))
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
    changed_files: list[str], e2e_map: dict, local_rule_path_set: set[str]
) -> tuple[list[str], list[str], list[str], str]:
    standard_profile_rules = e2e_map.get("profile_rules", {})
    manual_profile_rules = e2e_map.get("manual_profile_rules", {})
    workflow_suite_rules = e2e_map.get("workflow_suite_rules", {})
    full_ci = any(
        matches_with_local_rule_policy(
            path, e2e_map["full_ci_triggers"], local_rule_path_set
        )
        for path in changed_files
    )
    targeted_profiles = targeted_profiles_for_rules(
        standard_profile_rules, changed_files, local_rule_path_set
    )
    targeted_manual_profiles = targeted_profiles_for_rules(
        manual_profile_rules, changed_files, local_rule_path_set
    )
    targeted_workflow_suites = targeted_profiles_for_rules(
        workflow_suite_rules, changed_files, local_rule_path_set
    )

    if full_ci:
        local_profiles = targeted_profiles or e2e_map["default_local_profiles"]
        local_profiles.extend(targeted_manual_profiles)
        ci_profiles = e2e_map["full_ci_profiles"]
        ci_mode = "all"
    elif targeted_profiles or targeted_manual_profiles:
        local_profiles = [*targeted_profiles, *targeted_manual_profiles]
        ci_profiles = targeted_profiles
        ci_mode = "targeted" if targeted_profiles else "none"
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
    repo_manifest, task_matrix, e2e_map, _, _ = load_manifests()
    local_rule_path_set = local_agent_rule_paths(repo_manifest)
    matched_rules, fast_tests, feature_tests, requires_local_smoke = (
        match_task_matrix_rules(task_matrix, changed_files, local_rule_path_set)
    )
    local_profiles, ci_profiles, targeted_workflow_suites, ci_mode = (
        resolve_e2e_profiles(changed_files, e2e_map, local_rule_path_set)
    )
    doc_only = is_doc_only_rule_set(matched_rules)
    loop_mode = resolve_loop_mode(matched_rules)
    execution_plan_policy = resolve_execution_plan_policy(matched_rules)
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
        ci_e2e_mode=ci_mode,
        doc_only=doc_only,
        loop_mode=loop_mode,
        execution_plan_policy=execution_plan_policy,
    )


def resolve_environment(env_name: str) -> EnvironmentResolution:
    repo_manifest, _, _, _, _ = load_manifests()
    manifest_env, env_data = resolve_env_data(repo_manifest, env_name)
    return EnvironmentResolution(
        requested_env=env_name,
        manifest_env=manifest_env,
        build_target=env_data["build_target"],
        serve_command=env_data["serve_command"],
        smoke_config=env_data.get("smoke_config"),
        local_dev_fragment=env_data.get("local_dev_fragment"),
        local_env=env_data.get("local_env", False),
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


def selector_match_count(
    skill: dict, changed_files: list[str], local_rule_path_set: set[str]
) -> int:
    selector_paths = skill.get("selector_paths", [])
    allow_local_rules = skill["name"] == "harness-contract-change"
    return sum(
        1
        for path in changed_files
        if (
            (allow_local_rules and path in local_rule_path_set)
            or matches_with_local_rule_policy(
                path,
                selector_paths,
                local_rule_path_set,
                allow_local_rules=allow_local_rules,
            )
        )
    )


def anchor_match_count(
    skill: dict, changed_files: list[str], local_rule_path_set: set[str]
) -> int:
    anchor_paths = skill.get("anchor_paths", [])
    if not anchor_paths:
        return 0
    return sum(
        1
        for path in changed_files
        if matches_with_local_rule_policy(
            path,
            anchor_paths,
            local_rule_path_set,
        )
    )


def resolve_primary_skill(changed_files: list[str]) -> dict:
    repo_manifest, _, _, _, skill_registry = load_manifests()
    local_rule_path_set = local_agent_rule_paths(repo_manifest)
    primary_skills = skill_registry["skills"]["primary"]
    fallback = None
    best_match = None
    best_score = None

    for skill in primary_skills:
        if skill["name"] == "cross-stack-bugfix":
            fallback = skill
            continue

        match_count = selector_match_count(skill, changed_files, local_rule_path_set)
        if match_count == 0:
            continue

        anchor_count = anchor_match_count(skill, changed_files, local_rule_path_set)
        score = (anchor_count, match_count, skill.get("priority", 0))
        if best_score is None or score > best_score:
            best_match = skill
            best_score = score

    if best_match is not None:
        return best_match
    if fallback is not None:
        return fallback
    raise KeyError("Primary skill 'cross-stack-bugfix' is missing from registry")


def resolve_skill(changed_files: list[str], env_name: str | None) -> SkillResolution:
    _, _, _, _, skill_registry = load_manifests()
    skill_lookup = build_skill_lookup(skill_registry)
    primary = resolve_primary_skill(changed_files)
    context = resolve_context(changed_files)
    impacted_surfaces = resolve_impacted_surfaces(changed_files)
    fragment_names = list(primary.get("fragments", []))

    if env_name:
        env = resolve_environment(env_name)
        should_add_local_fragment = env.local_env and (
            context.requires_local_smoke or "local_smoke" in impacted_surfaces
        )
        if (
            should_add_local_fragment
            and env.local_dev_fragment
            and env.local_dev_fragment not in fragment_names
        ):
            fragment_names.append(env.local_dev_fragment)

    fragment_paths = [
        skill_lookup[name]["path"] for name in fragment_names if name in skill_lookup
    ]
    conditional_hit = [
        surface
        for surface in primary.get("conditional_surfaces", [])
        if surface in impacted_surfaces
    ]
    optional_hit = [
        surface
        for surface in primary.get("optional_surfaces", [])
        if surface in impacted_surfaces
    ]
    return SkillResolution(
        primary_skill=primary["name"],
        primary_skill_path=primary["path"],
        fragment_skills=unique_preserve_order(fragment_names),
        fragment_skill_paths=unique_preserve_order(fragment_paths),
        impacted_surfaces=impacted_surfaces,
        required_surfaces=primary.get("required_surfaces", []),
        conditional_surfaces_hit=conditional_hit,
        optional_surfaces_hit=optional_hit,
        stop_conditions=primary.get("stop_conditions", []),
        acceptance_criteria=primary.get("acceptance_criteria", []),
    )


def build_validation_commands(
    env: EnvironmentResolution, context: ResolvedContext
) -> list[str]:
    _, _, e2e_map, _, _ = load_manifests()
    commands = [*context.fast_tests, *context.feature_tests]
    if context.requires_local_smoke and env.local_env:
        commands.extend(
            [
                f"make agent-dev ENV={env.requested_env}",
                f"make agent-serve-local ENV={env.requested_env}",
                "make agent-smoke-local",
            ]
        )
    for profile in context.local_e2e_profiles:
        commands.append(f"make e2e-test E2E_PROFILE={profile} E2E_VERBOSE=true")
    for suite_name in context.workflow_integration_suites:
        suite = e2e_map.get("workflow_suite_rules", {}).get(suite_name, {})
        local_command = suite.get("local_command")
        if local_command:
            commands.append(local_command)
    return unique_preserve_order(commands)


def build_report(changed_files: list[str], env_name: str) -> AgentReport:
    env = resolve_environment(env_name)
    skill = resolve_skill(changed_files, env_name)
    context = resolve_context(changed_files)
    context_pack = build_context_pack(changed_files, env, skill, context)
    commands = build_validation_commands(env, context)
    return AgentReport(
        env=env,
        skill=skill,
        context=context,
        context_pack=context_pack,
        validation_commands=commands,
    )


def run_local_e2e(changed_files: list[str]) -> int:
    context = resolve_context(changed_files)
    if not context.local_e2e_profiles:
        print("No affected local E2E profiles.")
        return 0
    for profile in context.local_e2e_profiles:
        subprocess.run(
            f"make e2e-test E2E_PROFILE={shlex.quote(profile)} E2E_VERBOSE=true",
            cwd=REPO_ROOT,
            shell=True,
            check=True,
        )
    return 0
