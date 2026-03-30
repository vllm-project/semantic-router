#!/usr/bin/env python3
"""Task-first context-pack assembly for the agent harness."""

from __future__ import annotations

import fnmatch
import re

from agent_models import ContextPack, ContextReference
from agent_support import (
    REPO_ROOT,
    build_skill_lookup,
    load_context_map,
    load_manifests,
)

MUST_READ_HEADING = "## Must Read"
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

CONTEXT_PACK_SECTIONS = (
    "start_here",
    "must_read",
    "read_if_applicable",
    "local_rules",
    "resume_refs",
)

RESUME_REF_EXECUTION_PLAN_POLICIES = {"always", "long_horizon"}


class ContextPackBuilder:
    """Deduplicate context references while preserving a task-first section order."""

    def __init__(self) -> None:
        self._sections: dict[str, list[ContextReference]] = {
            section: [] for section in CONTEXT_PACK_SECTIONS
        }
        self._index: dict[str, tuple[str, ContextReference]] = {}

    def add(self, section: str, path: str, reason: str, source: str) -> None:
        normalized = normalize_context_path(path)
        existing = self._index.get(normalized)
        if existing is not None:
            _, reference = existing
            if reason not in reference.reason:
                reference.reason = f"{reference.reason} Also: {reason}"
            if source not in reference.sources:
                reference.sources.append(source)
            return

        reference = ContextReference(
            path=normalized,
            reason=reason,
            sources=[source],
        )
        self._sections[section].append(reference)
        self._index[normalized] = (section, reference)

    def build(self) -> ContextPack:
        return ContextPack(
            start_here=self._sections["start_here"],
            must_read=self._sections["must_read"],
            read_if_applicable=self._sections["read_if_applicable"],
            local_rules=self._sections["local_rules"],
            resume_refs=self._sections["resume_refs"],
        )


def normalize_context_path(raw_path: str) -> str:
    path = raw_path.strip()
    while path.startswith("./"):
        path = path[2:]
    return path


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def build_context_pack(
    changed_files: list[str],
    env,
    skill,
    context,
) -> ContextPack:
    repo_manifest, _, _, _, skill_registry = load_manifests()
    context_map = load_context_map()
    skill_lookup = build_skill_lookup(skill_registry)
    builder = ContextPackBuilder()

    add_context_map_refs(
        builder,
        "start_here",
        context_map.get("defaults", {}).get("start_here", []),
        "context-map:defaults.start_here",
    )
    add_skill_refs(builder, skill.primary_skill, skill.primary_skill_path, skill_lookup)
    for fragment_name, fragment_path in zip(
        skill.fragment_skills, skill.fragment_skill_paths, strict=False
    ):
        add_skill_refs(builder, fragment_name, fragment_path, skill_lookup)

    for rule_name in context.matched_rules:
        add_context_map_refs(
            builder,
            "read_if_applicable",
            context_map.get("rules", {}).get(rule_name, []),
            f"context-map:rule:{rule_name}",
        )

    for surface_name in skill.impacted_surfaces:
        add_context_map_refs(
            builder,
            "read_if_applicable",
            context_map.get("surfaces", {}).get(surface_name, []),
            f"context-map:surface:{surface_name}",
        )

    add_context_map_refs(
        builder,
        "read_if_applicable",
        context_map.get("envs", {}).get(env.manifest_env, []),
        f"context-map:env:{env.manifest_env}",
    )
    add_matching_local_rules(builder, changed_files, repo_manifest)
    if should_include_resume_refs(context):
        add_context_map_refs(
            builder,
            "resume_refs",
            context_map.get("defaults", {}).get("resume_refs", []),
            "context-map:defaults.resume_refs",
        )
        add_context_map_refs(
            builder,
            "resume_refs",
            context_map.get("skills", {})
            .get(skill.primary_skill, {})
            .get("resume_refs", []),
            f"context-map:skill:{skill.primary_skill}.resume_refs",
        )
    return builder.build()


def should_include_resume_refs(context) -> bool:
    return context.execution_plan_policy in RESUME_REF_EXECUTION_PLAN_POLICIES


def add_matching_local_rules(
    builder: ContextPackBuilder, changed_files: list[str], repo_manifest: dict
) -> None:
    for rule in repo_manifest.get("local_agent_rules", []):
        scope = rule.get("scope")
        path = rule["path"]
        if path in changed_files or (
            scope and any(matches_any(changed, [scope]) for changed in changed_files)
        ):
            builder.add(
                "local_rules",
                path,
                "Nearest hotspot supplement for the changed subtree.",
                f"local-rule:{path}",
            )


def add_skill_refs(
    builder: ContextPackBuilder,
    skill_name: str,
    skill_path: str,
    skill_lookup: dict[str, dict],
) -> None:
    category = skill_lookup[skill_name]["category"]
    builder.add(
        "must_read",
        skill_path,
        f"{category.rstrip('s').capitalize()} skill contract for this task.",
        f"skill:{skill_name}",
    )
    for referenced_path in extract_must_read_paths(skill_path):
        builder.add(
            "must_read",
            referenced_path,
            f"Referenced by the '{skill_name}' skill Must Read section.",
            f"skill:{skill_name}:must-read",
        )


def add_context_map_refs(
    builder: ContextPackBuilder,
    section: str,
    refs: list[dict],
    source: str,
) -> None:
    for ref in refs:
        builder.add(section, ref["path"], ref["reason"], source)


def extract_must_read_paths(skill_path: str) -> list[str]:
    absolute_path = REPO_ROOT / skill_path
    lines = absolute_path.read_text(encoding="utf-8").splitlines()
    in_section = False
    resolved_paths: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if stripped == MUST_READ_HEADING:
                in_section = True
                continue
            if in_section:
                break
        if not in_section:
            continue
        for match in MARKDOWN_LINK_PATTERN.finditer(line):
            resolved = resolve_markdown_target(skill_path, match.group(1))
            if resolved and resolved not in resolved_paths:
                resolved_paths.append(resolved)
    return resolved_paths


def resolve_markdown_target(source_path: str, target: str) -> str | None:
    if target.startswith(("http://", "https://", "mailto:")):
        return None

    target_path = target.split("#", 1)[0]
    if not target_path:
        return None

    source_dir = (REPO_ROOT / source_path).parent
    resolved = (source_dir / target_path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None
