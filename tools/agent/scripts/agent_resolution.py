#!/usr/bin/env python3
"""Skill resolution and report assembly for the agent gate."""

from __future__ import annotations

import shlex
import subprocess

from agent_changed_files import (
    get_changed_files,
    git_changed_files,
    split_changed_files,
)
from agent_context_pack import build_context_pack
from agent_context_resolution import (
    local_agent_rule_paths,
    matches_with_local_rule_policy,
    resolve_context,
    resolve_impacted_surfaces,
    unique_preserve_order,
)
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

__all__ = [
    "build_report",
    "get_changed_files",
    "git_changed_files",
    "resolve_context",
    "resolve_environment",
    "resolve_primary_skill",
    "resolve_skill",
    "run_local_e2e",
    "split_changed_files",
    "unique_preserve_order",
]


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
    fragment_names = resolve_active_fragments(
        primary,
        impacted_surfaces,
        skill_lookup,
    )

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


def resolve_active_fragments(
    primary: dict,
    impacted_surfaces: list[str],
    skill_lookup: dict[str, dict],
) -> list[str]:
    active_fragments: list[str] = []
    impacted_surface_set = set(impacted_surfaces)
    if primary.get("auto_surface_fragments"):
        for fragment_name, fragment in skill_lookup.items():
            if fragment.get("category") != "fragments":
                continue
            owned_surfaces = set(fragment.get("owned_surfaces", []))
            if not owned_surfaces or owned_surfaces.intersection(impacted_surface_set):
                active_fragments.append(fragment_name)
        return unique_preserve_order(active_fragments)

    for fragment_name in primary.get("fragments", []):
        fragment = skill_lookup.get(fragment_name)
        if fragment is None:
            continue
        owned_surfaces = set(fragment.get("owned_surfaces", []))
        if not owned_surfaces or owned_surfaces.intersection(impacted_surface_set):
            active_fragments.append(fragment_name)
    return unique_preserve_order(active_fragments)


def build_validation_commands(
    env: EnvironmentResolution, context: ResolvedContext
) -> list[str]:
    commands = [*context.fast_tests, *context.feature_tests]
    if context.requires_local_smoke and env.local_env:
        commands.extend(
            [
                f"make agent-dev ENV={env.requested_env}",
                f"make agent-serve-local ENV={env.requested_env}",
                "make agent-smoke-local",
            ]
        )
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
