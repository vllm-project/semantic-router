#!/usr/bin/env python3
"""Shared data models for agent gate reporting."""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field


@dataclass
class ResolvedContext:
    changed_files: list[str]
    matched_rules: list[str] = field(default_factory=list)
    fast_tests: list[str] = field(default_factory=list)
    feature_tests: list[str] = field(default_factory=list)
    requires_local_smoke: bool = False
    local_e2e_profiles: list[str] = field(default_factory=list)
    ci_e2e_profiles: list[str] = field(default_factory=list)
    workflow_integration_suites: list[str] = field(default_factory=list)
    ci_e2e_mode: str = "none"
    doc_only: bool = False

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    def to_env(self) -> str:
        values = {
            "AGENT_CHANGED_FILES": ",".join(self.changed_files),
            "AGENT_MATCHED_RULES": ",".join(self.matched_rules),
            "AGENT_FAST_TESTS": "|||".join(self.fast_tests),
            "AGENT_FEATURE_TESTS": "|||".join(self.feature_tests),
            "AGENT_REQUIRES_LOCAL_SMOKE": str(self.requires_local_smoke).lower(),
            "AGENT_LOCAL_E2E_PROFILES": ",".join(self.local_e2e_profiles),
            "AGENT_CI_E2E_PROFILES": ",".join(self.ci_e2e_profiles),
            "AGENT_WORKFLOW_INTEGRATION_SUITES": ",".join(
                self.workflow_integration_suites
            ),
            "AGENT_CI_E2E_MODE": self.ci_e2e_mode,
            "AGENT_DOC_ONLY": str(self.doc_only).lower(),
        }
        return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())

    def to_summary(self) -> str:
        lines = [
            "Agent Context",
            f"  Changed files: {len(self.changed_files)}",
            f"  Matched rules: {', '.join(self.matched_rules) or 'none'}",
            f"  Fast tests: {', '.join(self.fast_tests) or 'none'}",
            f"  Feature tests: {', '.join(self.feature_tests) or 'none'}",
            f"  Local smoke: {'required' if self.requires_local_smoke else 'not required'}",
            f"  Local E2E profiles: {', '.join(self.local_e2e_profiles) or 'none'}",
            "  Workflow integration suites: "
            + (", ".join(self.workflow_integration_suites) or "none"),
            f"  CI E2E mode: {self.ci_e2e_mode}",
        ]
        if self.ci_e2e_profiles:
            lines.append(f"  CI E2E profiles: {', '.join(self.ci_e2e_profiles)}")
        if self.doc_only:
            lines.append("  Classification: documentation-only")
        return "\n".join(lines)


@dataclass
class EnvironmentResolution:
    requested_env: str
    manifest_env: str
    build_target: str
    serve_command: str
    smoke_config: str | None
    local_dev_fragment: str | None
    local_env: bool

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    def to_env(self) -> str:
        values = {
            "AGENT_REQUESTED_ENV": self.requested_env,
            "AGENT_MANIFEST_ENV": self.manifest_env,
            "AGENT_BUILD_TARGET": self.build_target,
            "AGENT_SERVE_COMMAND": self.serve_command,
            "AGENT_SMOKE_CONFIG": self.smoke_config or "",
            "AGENT_LOCAL_DEV_FRAGMENT": self.local_dev_fragment or "",
            "AGENT_LOCAL_ENV": str(self.local_env).lower(),
        }
        return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())

    def to_summary(self) -> str:
        lines = [
            "Agent Environment",
            f"  Requested env: {self.requested_env}",
            f"  Manifest env: {self.manifest_env}",
            f"  Build target: {self.build_target}",
            f"  Serve command: {self.serve_command}",
            f"  Local env: {'yes' if self.local_env else 'no'}",
        ]
        if self.smoke_config:
            lines.append(f"  Smoke config: {self.smoke_config}")
        if self.local_dev_fragment:
            lines.append(f"  Local dev fragment: {self.local_dev_fragment}")
        return "\n".join(lines)


@dataclass
class SkillResolution:
    primary_skill: str
    primary_skill_path: str
    fragment_skills: list[str]
    fragment_skill_paths: list[str]
    impacted_surfaces: list[str]
    required_surfaces: list[str]
    conditional_surfaces_hit: list[str]
    optional_surfaces_hit: list[str]
    stop_conditions: list[str]
    acceptance_criteria: list[str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    def to_env(self) -> str:
        values = {
            "AGENT_PRIMARY_SKILL": self.primary_skill,
            "AGENT_PRIMARY_SKILL_PATH": self.primary_skill_path,
            "AGENT_FRAGMENT_SKILLS": ",".join(self.fragment_skills),
            "AGENT_IMPACTED_SURFACES": ",".join(self.impacted_surfaces),
            "AGENT_REQUIRED_SURFACES": ",".join(self.required_surfaces),
            "AGENT_CONDITIONAL_SURFACES_HIT": ",".join(self.conditional_surfaces_hit),
            "AGENT_OPTIONAL_SURFACES_HIT": ",".join(self.optional_surfaces_hit),
        }
        return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())

    def to_summary(self) -> str:
        lines = [
            "Agent Skill",
            f"  Primary skill: {self.primary_skill}",
            f"  Fragment skills: {', '.join(self.fragment_skills) or 'none'}",
            f"  Impacted surfaces: {', '.join(self.impacted_surfaces) or 'none'}",
            f"  Required surfaces: {', '.join(self.required_surfaces) or 'none'}",
            "  Conditional surfaces hit: "
            + (", ".join(self.conditional_surfaces_hit) or "none"),
        ]
        if self.optional_surfaces_hit:
            lines.append(
                f"  Optional surfaces hit: {', '.join(self.optional_surfaces_hit)}"
            )
        return "\n".join(lines)


@dataclass
class ContextReference:
    path: str
    reason: str
    sources: list[str] = field(default_factory=list)


@dataclass
class ContextPack:
    start_here: list[ContextReference] = field(default_factory=list)
    must_read: list[ContextReference] = field(default_factory=list)
    read_if_applicable: list[ContextReference] = field(default_factory=list)
    local_rules: list[ContextReference] = field(default_factory=list)
    resume_refs: list[ContextReference] = field(default_factory=list)

    def to_summary(self) -> str:
        lines = ["Context Pack"]
        append_context_refs(lines, "Start here", self.start_here)
        append_context_refs(lines, "Must read", self.must_read)
        append_context_refs(lines, "Read if applicable", self.read_if_applicable)
        append_context_refs(lines, "Local rules", self.local_rules)
        append_context_refs(lines, "Resume refs", self.resume_refs)
        return "\n".join(lines)


def append_context_refs(
    lines: list[str], label: str, refs: list[ContextReference]
) -> None:
    if not refs:
        return
    lines.append(f"  {label}:")
    for ref in refs:
        lines.append(f"    - {ref.path} :: {ref.reason}")


@dataclass
class AgentReport:
    env: EnvironmentResolution
    skill: SkillResolution
    context: ResolvedContext
    context_pack: ContextPack
    validation_commands: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def to_summary(self) -> str:
        lines = [
            self.env.to_summary(),
            "",
            self.skill.to_summary(),
            "",
            self.context.to_summary(),
            "",
            self.context_pack.to_summary(),
            "",
            "Validation Commands",
        ]
        if self.validation_commands:
            lines.extend(f"  - {command}" for command in self.validation_commands)
        else:
            lines.append("  - none")
        return "\n".join(lines)


@dataclass
class HarnessScorecard:
    validation_status: str
    validation_error_count: int
    indexed_harness_doc_count: int
    governed_doc_count: int
    open_technical_debt_items: list[str]
    local_rule_count: int
    subsystem_count: int
    surface_count: int
    primary_skill_count: int
    fragment_skill_count: int
    support_skill_count: int
    legacy_reference_skill_count: int
    open_execution_plan_tasks: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    def to_summary(self) -> str:
        lines = [
            "Harness Scorecard",
            f"  Validation status: {self.validation_status}",
            f"  Validation errors: {self.validation_error_count}",
            f"  Indexed harness docs: {self.indexed_harness_doc_count}",
            f"  Governed docs: {self.governed_doc_count}",
            f"  Open technical debt items: {len(self.open_technical_debt_items)}",
            f"  Local rules: {self.local_rule_count}",
            f"  Subsystems: {self.subsystem_count}",
            f"  Surfaces: {self.surface_count}",
            f"  Primary skills: {self.primary_skill_count}",
            f"  Fragment skills: {self.fragment_skill_count}",
            f"  Support skills: {self.support_skill_count}",
            f"  Legacy reference skills: {self.legacy_reference_skill_count}",
            f"  Open execution plan tasks: {len(self.open_execution_plan_tasks)}",
        ]
        if self.open_technical_debt_items:
            lines.extend(f"  - {item}" for item in self.open_technical_debt_items)
        if self.open_execution_plan_tasks:
            lines.extend(f"  - {task}" for task in self.open_execution_plan_tasks)
        if self.validation_errors:
            lines.append("  Validation error details:")
            lines.extend(f"  - {error}" for error in self.validation_errors)
        return "\n".join(lines)
