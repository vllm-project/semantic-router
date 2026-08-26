#!/usr/bin/env python3
"""Shared helpers for agent-aware scripts."""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml
from go_complexity_manifest import MANIFEST_RELATIVE_PATH
from go_lint_gate import go_lint_tool_required, run_go_lint_gate
from module_file_groups import group_files_by_module

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_DIR = REPO_ROOT / "tools" / "agent"
AGENT_DOCS_DIR = REPO_ROOT / "tools" / "agent" / "docs"
CHANGE_SURFACES_DOC = AGENT_DOCS_DIR / "change-surfaces.md"
AGENT_INDEX_DOC = AGENT_DOCS_DIR / "README.md"
AGENT_GOVERNANCE_DOC = AGENT_DOCS_DIR / "governance.md"
AGENTS_ENTRY_DOC = REPO_ROOT / "AGENTS.md"
MAKEFILES = [
    REPO_ROOT / "Makefile",
    *sorted((REPO_ROOT / "tools" / "make").glob("*.mk")),
]
GO_AGENT_CONFIG = REPO_ROOT / "tools" / "linter" / "go" / ".golangci.agent.yml"
GO_COMPLEXITY_DEBT_MANIFEST = REPO_ROOT / MANIFEST_RELATIVE_PATH
GO_MODULE_CONFIG_OVERRIDES = {
    REPO_ROOT / "dashboard" / "backend": REPO_ROOT
    / "tools"
    / "linter"
    / "go"
    / ".golangci.yml",
}
RUFF_CONFIG = REPO_ROOT / "tools" / "linter" / "python" / ".ruff.toml"
ABSOLUTE_MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\((/[^)]+)\)")
REFERENCE_CONFIG_PATTERNS = (
    "config/**",
    "src/semantic-router/pkg/config/**",
)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_context_map() -> dict:
    return load_yaml(AGENT_DIR / "context-map.yaml")


def load_manifests() -> tuple[dict, dict, dict, dict, dict]:
    return (
        load_yaml(AGENT_DIR / "repo-manifest.yaml"),
        load_yaml(AGENT_DIR / "task-matrix.yaml"),
        load_yaml(AGENT_DIR / "test-domain-registry.yaml"),
        load_yaml(AGENT_DIR / "structure-rules.yaml"),
        load_yaml(AGENT_DIR / "skill-registry.yaml"),
    )


def collect_make_targets() -> set[str]:
    pattern = re.compile(r"^([A-Za-z0-9_.-]+):(?:\s|$)")
    targets: set[str] = set()
    for path in MAKEFILES:
        targets.update(collect_make_targets_from_file(path, pattern))
    return targets


def collect_make_targets_from_file(path: Path, pattern: re.Pattern[str]) -> set[str]:
    targets: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("\t", "#", ".")):
                continue
            match = pattern.match(line)
            if not match:
                continue
            target = match.group(1)
            if "%" not in target and "$" not in target:
                targets.add(target)
    return targets


def validate_glob(pattern: str) -> bool:
    return any(REPO_ROOT.glob(pattern))


def append_missing_make_target(
    errors: list[str], label: str, command: str, make_targets: set[str]
) -> None:
    if not command.startswith("make "):
        return
    target = command.split()[1]
    if target not in make_targets:
        errors.append(f"{label} references missing make target '{target}'")


def collect_manifest_globs(
    repo_manifest: dict,
    task_matrix: dict,
    test_domain_registry: dict,
    structure_rules: dict,
    skill_registry: dict,
) -> list[str]:
    manifest_globs: list[str] = []
    for subsystem in repo_manifest["subsystems"]:
        manifest_globs.extend(subsystem["paths"])

    for section in ("domains", "profiles"):
        for data in test_domain_registry[section].values():
            manifest_globs.extend(data.get("paths", []))

    for rule in task_matrix["rules"]:
        manifest_globs.extend(rule["paths"])

    for language in structure_rules["languages"].values():
        manifest_globs.extend(language["globs"])
    for dep_rule in structure_rules["dependency_rules"]:
        manifest_globs.extend(dep_rule["applies_to"])

    for surface in skill_registry["surfaces"].values():
        manifest_globs.extend(surface["paths"])
    for skill in iter_registry_skills(skill_registry):
        manifest_globs.extend(skill.get("selector_paths", []))
        manifest_globs.extend(skill.get("anchor_paths", []))

    return manifest_globs


def iter_registry_skills(skill_registry: dict) -> list[dict]:
    skills: list[dict] = []
    for category in ("primary", "support"):
        for skill in skill_registry["skills"].get(category, []):
            enriched = dict(skill)
            enriched["category"] = category
            skills.append(enriched)
    return skills


def build_skill_lookup(skill_registry: dict) -> dict[str, dict]:
    return {skill["name"]: skill for skill in iter_registry_skills(skill_registry)}


def collect_task_rule_names(task_matrix: dict) -> set[str]:
    return {rule["name"] for rule in task_matrix["rules"]}


def resolve_env_data(repo_manifest: dict, env_name: str) -> tuple[str, dict]:
    requested = env_name.strip()
    for manifest_name, data in repo_manifest["supported_envs"].items():
        aliases = set(data.get("aliases", []))
        aliases.add(manifest_name)
        if requested in aliases:
            return manifest_name, data
    supported = ", ".join(sorted(repo_manifest["supported_envs"]))
    raise KeyError(f"Unsupported env '{env_name}'. Expected one of: {supported}")


def run_command(command: str) -> None:
    print(f"+ {command}")
    subprocess.run(command, cwd=REPO_ROOT, shell=True, check=True)


def run_test_commands(commands: list[str], label: str) -> int:
    if not commands:
        print(f"No {label} commands matched.")
        return 0
    print(f"Running {label} commands:")
    for command in commands:
        run_command(command)
    return 0


def run_reference_config_lint(changed_files: list[str]) -> int:
    if not any(
        fnmatch.fnmatch(path, pattern)
        for path in changed_files
        for pattern in REFERENCE_CONFIG_PATTERNS
    ):
        print("No reference config contract files changed.")
        return 0
    module_root = REPO_ROOT / "src" / "semantic-router"
    command = [
        "go",
        "test",
        "./pkg/config/...",
        "-run",
        "TestReferenceConfig",
        "-count=1",
    ]
    print(f"+ {' '.join(command)} (cwd={module_root})")
    result = subprocess.run(command, cwd=module_root, check=False)
    return result.returncode


def run_go_lint(changed_files: list[str], base_ref: str | None = None) -> int:
    return run_go_lint_gate(
        changed_files=changed_files,
        base_ref=base_ref,
        repo_root=REPO_ROOT,
        default_config=GO_AGENT_CONFIG,
        debt_manifest=GO_COMPLEXITY_DEBT_MANIFEST,
        module_overrides=GO_MODULE_CONFIG_OVERRIDES,
    )


def needs_go_lint_tool(changed_files: list[str]) -> bool:
    return go_lint_tool_required(changed_files, REPO_ROOT)


def rust_clippy_base_flags() -> list[str]:
    return [
        "--all-targets",
        "--",
        "-D",
        "warnings",
        "-W",
        "clippy::cognitive_complexity",
        "-W",
        "clippy::type_complexity",
        "-W",
        "clippy::too_many_arguments",
    ]


def resolve_rust_span_path(crate_root: Path, file_name: str) -> Path:
    span_path = Path(file_name)
    if span_path.is_absolute():
        return span_path.resolve()
    return (crate_root / span_path).resolve()


def collect_changed_rust_messages(
    crate_root: Path, changed_paths: set[Path], stdout: str
) -> tuple[list[str], list[str], bool]:
    relevant_messages: list[str] = []
    parse_failures: list[str] = []
    saw_compiler_message = False

    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            parse_failures.append(line)
            continue

        if payload.get("reason") != "compiler-message":
            continue
        saw_compiler_message = True
        message = payload.get("message", {})
        spans = message.get("spans", [])
        if not spans:
            continue

        span_paths = {
            resolve_rust_span_path(crate_root, file_name)
            for span in spans
            if (file_name := span.get("file_name"))
        }
        if span_paths.isdisjoint(changed_paths):
            continue

        rendered = message.get("rendered")
        if rendered:
            relevant_messages.append(rendered.rstrip())
            continue
        relevant_messages.append(json.dumps(message, sort_keys=True))

    return relevant_messages, parse_failures, saw_compiler_message


def build_rust_clippy_command(crate_root: Path) -> list[str]:
    command = ["cargo", "clippy", "--message-format=json"]
    if crate_root.name == "candle-binding":
        command.append("--no-default-features")
    command.extend(rust_clippy_base_flags())
    return command


def run_rust_clippy_for_crate(crate_root: Path, changed_paths: set[Path]) -> int:
    command = build_rust_clippy_command(crate_root)
    print(f"+ {' '.join(command)} (cwd={crate_root})")
    result = subprocess.run(
        command,
        cwd=crate_root,
        capture_output=True,
        text=True,
        check=False,
    )
    relevant_messages, parse_failures, saw_compiler_message = (
        collect_changed_rust_messages(crate_root, changed_paths, result.stdout)
    )
    if relevant_messages:
        sys.stderr.write("\n".join(relevant_messages) + "\n")
        return 1

    if result.returncode != 0:
        if result.stderr:
            sys.stderr.write(result.stderr)
        if parse_failures and not saw_compiler_message:
            sys.stderr.write("\n".join(parse_failures) + "\n")
            return result.returncode
        print(
            "Ignoring crate-wide Rust clippy findings outside the changed-file set "
            f"for {crate_root.relative_to(REPO_ROOT)}."
        )
        return 0

    if result.stderr:
        sys.stderr.write(result.stderr)
    return 0


def run_rust_lint(changed_files: list[str]) -> int:
    grouped = group_files_by_module(REPO_ROOT, changed_files, "Cargo.toml", {".rs"})
    if not grouped:
        print("No changed Rust files detected.")
        return 0

    for crate_root in sorted(grouped):
        changed_paths = {path.resolve() for path in grouped[crate_root]}
        if run_rust_clippy_for_crate(crate_root, changed_paths) != 0:
            return 1
    return 0


def run_python_lint(changed_files: list[str]) -> int:
    files = [
        str(REPO_ROOT / changed)
        for changed in changed_files
        if changed.endswith(".py") and (REPO_ROOT / changed).exists()
    ]
    if not files:
        print("No changed Python files detected.")
        return 0
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--config",
        str(RUFF_CONFIG),
        *files,
    ]
    print(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0
