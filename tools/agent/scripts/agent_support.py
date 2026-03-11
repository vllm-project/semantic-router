#!/usr/bin/env python3
"""Shared helpers for agent-aware scripts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml
from go_lint_support import (
    filter_go_issues,
    load_golangci_payload,
    print_go_issues,
    resolve_golangci_lint,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_DIR = REPO_ROOT / "tools" / "agent"
CHANGE_SURFACES_DOC = REPO_ROOT / "docs" / "agent" / "change-surfaces.md"
AGENT_INDEX_DOC = REPO_ROOT / "docs" / "agent" / "README.md"
AGENT_GOVERNANCE_DOC = REPO_ROOT / "docs" / "agent" / "governance.md"
AGENTS_ENTRY_DOC = REPO_ROOT / "AGENTS.md"
MAKEFILES = [
    REPO_ROOT / "Makefile",
    *sorted((REPO_ROOT / "tools" / "make").glob("*.mk")),
]
GO_AGENT_CONFIG = REPO_ROOT / "tools" / "linter" / "go" / ".golangci.agent.yml"
GO_MODULE_CONFIG_OVERRIDES = {
    REPO_ROOT
    / "dashboard"
    / "backend": REPO_ROOT
    / "tools"
    / "linter"
    / "go"
    / ".golangci.yml",
}
RUFF_CONFIG = REPO_ROOT / "tools" / "linter" / "python" / ".ruff.toml"
ABSOLUTE_MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\((/[^)]+)\)")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_context_map() -> dict:
    return load_yaml(AGENT_DIR / "context-map.yaml")


def load_manifests() -> tuple[dict, dict, dict, dict, dict]:
    return (
        load_yaml(AGENT_DIR / "repo-manifest.yaml"),
        load_yaml(AGENT_DIR / "task-matrix.yaml"),
        load_yaml(AGENT_DIR / "e2e-profile-map.yaml"),
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
    e2e_map: dict,
    structure_rules: dict,
    skill_registry: dict,
) -> list[str]:
    manifest_globs: list[str] = []
    for subsystem in repo_manifest["subsystems"]:
        manifest_globs.extend(subsystem["paths"])

    manifest_globs.extend(e2e_map["full_ci_triggers"])
    for data in e2e_map["profile_rules"].values():
        manifest_globs.extend(data["paths"])

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

    return manifest_globs


def iter_registry_skills(skill_registry: dict) -> list[dict]:
    skills: list[dict] = []
    for category in ("primary", "fragments", "support", "legacy_reference"):
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


def group_files_by_module(
    changed_files: list[str], manifest_name: str, extensions: set[str]
) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for changed in changed_files:
        path = REPO_ROOT / changed
        if path.suffix not in extensions or not path.exists():
            continue
        current = path.parent
        while current != REPO_ROOT.parent:
            manifest = current / manifest_name
            if manifest.exists():
                grouped.setdefault(current, []).append(path)
                break
            if current == REPO_ROOT:
                break
            current = current.parent
    return grouped


def run_go_lint(changed_files: list[str], base_ref: str | None = None) -> int:
    grouped = group_files_by_module(changed_files, "go.mod", {".go"})
    if not grouped:
        print("No changed Go files detected.")
        return 0

    golangci_lint = resolve_golangci_lint(REPO_ROOT)
    for module_root, files in grouped.items():
        config_path = GO_MODULE_CONFIG_OVERRIDES.get(module_root, GO_AGENT_CONFIG)
        changed_paths = {file.relative_to(REPO_ROOT).as_posix() for file in files}
        package_dirs = sorted(
            {
                (
                    "."
                    if file.parent == module_root
                    else f"./{file.parent.relative_to(module_root).as_posix()}"
                )
                for file in files
            }
        )
        command = [
            golangci_lint,
            "run",
            "--config",
            str(config_path),
        ]
        if base_ref:
            command.extend(["--new-from-rev", base_ref])
        command.extend(
            [
                "--issues-exit-code",
                "0",
                "--output.json.path",
                "stdout",
                "--path-mode",
                "abs",
                *package_dirs,
            ]
        )
        print(f"+ {' '.join(command)} (cwd={module_root})")
        result = subprocess.run(
            command,
            cwd=module_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            return result.returncode
        if result.stderr:
            sys.stderr.write(result.stderr)
        payload = load_golangci_payload(result.stdout)
        issues = filter_go_issues(
            REPO_ROOT, module_root, payload.get("Issues", []), changed_paths
        )
        if issues:
            print_go_issues(issues)
            print(f"{len(issues)} changed-file Go lint issue(s) found.")
            return 1
    return 0


def run_rust_lint(changed_files: list[str]) -> int:
    grouped = group_files_by_module(changed_files, "Cargo.toml", {".rs"})
    if not grouped:
        print("No changed Rust files detected.")
        return 0

    base_flags = [
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
    for crate_root in sorted(grouped):
        command = ["cargo", "clippy"]
        if crate_root.name == "candle-binding":
            command.append("--no-default-features")
        command.extend(base_flags)
        print(f"+ {' '.join(command)} (cwd={crate_root})")
        subprocess.run(command, cwd=crate_root, check=True)
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
    command = ["python3", "-m", "ruff", "check", "--config", str(RUFF_CONFIG), *files]
    print(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0
