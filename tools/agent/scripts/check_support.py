#!/usr/bin/env python3
"""Small, reusable helpers for changed-file checks."""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from go_lint_support import (
    filter_go_issues,
    load_golangci_payload,
    print_go_issues,
    resolve_golangci_lint,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MAKEFILES = [
    REPO_ROOT / "Makefile",
    *sorted((REPO_ROOT / "tools" / "make").glob("*.mk")),
]
GO_LINT_CONFIG = REPO_ROOT / "tools" / "linter" / "go" / ".golangci.yml"
RUFF_CONFIG = REPO_ROOT / "tools" / "linter" / "python" / ".ruff.toml"
REFERENCE_CONFIG_PATTERNS = (
    "config/**",
    "src/semantic-router/pkg/config/**",
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


def append_missing_make_target(
    errors: list[str], label: str, command: str, make_targets: set[str]
) -> None:
    if not command.startswith("make "):
        return
    target = command.split()[1]
    if target not in make_targets:
        errors.append(f"{label} references missing make target '{target}'")


def run_command(command: str) -> None:
    print(f"+ {command}")
    subprocess.run(shlex.split(command), cwd=REPO_ROOT, check=True)


def run_test_commands(commands: list[str], label: str) -> int:
    if not commands:
        print(f"No {label} commands matched.")
        return 0
    print(f"Running {label} commands:")
    for command in commands:
        run_command(command)
    return 0


def run_precommit(
    changed_files: list[str], base_ref: str | None, *, ci_static_only: bool = False
) -> int:
    files = [changed for changed in changed_files if (REPO_ROOT / changed).exists()]
    if not files:
        print("No existing changed files for pre-commit.")
        return 0

    environment = os.environ.copy()
    if base_ref:
        environment["BASE_REF"] = base_ref
    if ci_static_only:
        # These checks retain local-hook behavior. CI assigns generated contracts
        # and the trusted-base security scan one explicit execution owner each.
        owned_elsewhere = {
            "model-catalog-generated",
            "decision-runtime-catalog-generated",
            "cli-reference-generated",
            "configuration-reference-generated",
            "public-agent-skill",
            "supply-chain-security-scan",
        }
        requested = set(filter(None, environment.get("SKIP", "").split(",")))
        environment["SKIP"] = ",".join(sorted(requested | owned_elsewhere))

    precommit = Path(sys.executable).with_name("pre-commit")
    command = [str(precommit), "run", "--files", *files]
    print(f"+ {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    ).returncode


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
        config_path = GO_LINT_CONFIG
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
