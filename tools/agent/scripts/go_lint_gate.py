#!/usr/bin/env python3
"""Aggregate changed-file Go lint findings and enforce exact complexity debt."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from go_complexity_baseline import FREEZE_RELATIVE_PATH
from go_complexity_identity import COMPLEXITY_LINTERS
from go_complexity_manifest import (
    MANIFEST_RELATIVE_PATH,
    ComplexityRatchetError,
    detect_tool_version,
)
from go_complexity_ratchet import print_ratchet_result, run_complexity_ratchet
from go_complexity_source_policy import (
    BuildContext,
    candidate_build_contexts_for_source,
)
from go_lint_support import (
    filter_go_issues,
    go_issue_record,
    go_issues_from_payload,
    load_golangci_payload,
    print_go_issues,
    repo_relative_go_issue,
    resolve_golangci_lint,
)
from module_file_groups import group_files_by_module


def _package_dirs(module_root: Path, files: list[Path]) -> list[str]:
    return sorted(
        {
            (
                "."
                if file.parent == module_root
                else f"./{file.parent.relative_to(module_root).as_posix()}"
            )
            for file in files
        }
    )


def unresolved_go_paths(
    repo_root: Path,
    changed_go_paths: set[str],
    grouped: dict[Path, list[Path]],
) -> set[str]:
    grouped_paths = {
        file.relative_to(repo_root).as_posix()
        for files in grouped.values()
        for file in files
    }
    return changed_go_paths - grouped_paths


def existing_unresolved_go_paths(
    repo_root: Path, unresolved_paths: set[str]
) -> set[str]:
    return {path for path in unresolved_paths if (repo_root / path).is_file()}


def _complexity_changed_paths(
    grouped: dict[Path, list[Path]],
    deleted_paths: set[str],
    repo_root: Path,
) -> set[str]:
    owned = {
        file.relative_to(repo_root).as_posix()
        for files in grouped.values()
        for file in files
    }
    return owned | deleted_paths


def _tracked_go_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.go"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ComplexityRatchetError("cannot enumerate tracked Go files")
    try:
        paths = [part.decode("utf-8") for part in result.stdout.split(b"\x00") if part]
    except UnicodeDecodeError as exc:
        raise ComplexityRatchetError("tracked Go path is not UTF-8") from exc
    return paths


def is_complexity_contract_path(path: str, repo_root: Path) -> bool:
    exact = {
        "tools/agent/requirements.txt",
        "tools/agent/scripts/agent_changed_files.py",
        "tools/agent/scripts/agent_gate.py",
        "tools/agent/scripts/agent_support.py",
        "tools/agent/scripts/go_lint_gate.py",
        "tools/agent/scripts/go_lint_support.py",
        "tools/agent/scripts/module_file_groups.py",
        "tools/make/agent.mk",
        (repo_root / "tools/linter/go/.golangci.agent.yml")
        .relative_to(repo_root)
        .as_posix(),
        MANIFEST_RELATIVE_PATH.as_posix(),
        FREEZE_RELATIVE_PATH.as_posix(),
    }
    return (
        path in exact
        or path.startswith("tools/agent/scripts/go_complexity_")
        or Path(path).name in {"go.mod", "go.work", "go.work.sum"}
    )


def go_lint_tool_required(changed_files: list[str], repo_root: Path) -> bool:
    """Return whether changed-file lint needs the pinned Go lint tool."""

    return any(
        Path(path).suffix == ".go" or is_complexity_contract_path(path, repo_root)
        for path in changed_files
    )


def _collect_module_issues(
    command: str,
    repo_root: Path,
    module_root: Path,
    config_path: Path,
    files: list[Path],
    build_context: BuildContext | None = None,
) -> tuple[int, list[dict]]:
    lint_command = [
        command,
        "run",
        "--config",
        str(config_path),
        "--issues-exit-code",
        "0",
        "--output.json.path",
        "stdout",
        "--path-mode",
        "abs",
    ]
    environment = _context_environment(build_context)
    if build_context is not None and build_context.build_tags:
        lint_command.extend(["--build-tags", ",".join(build_context.build_tags)])
    lint_command.extend(_package_dirs(module_root, files))
    print(f"+ {' '.join(lint_command)} (cwd={module_root})")
    result = subprocess.run(
        lint_command,
        cwd=module_root,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode, []
    try:
        payload = load_golangci_payload(result.stdout)
        issues = go_issues_from_payload(payload)
    except (TypeError, ValueError) as exc:
        print(f"golangci-lint returned invalid JSON: {exc}", file=sys.stderr)
        return 2, []
    tool_errors = _golangci_tool_errors(payload, result.stderr)
    if tool_errors:
        for message in tool_errors:
            print(f"golangci-lint analysis error: {message}", file=sys.stderr)
        return 2, []
    changed_paths = {file.relative_to(repo_root).as_posix() for file in files}
    return 0, filter_go_issues(repo_root, module_root, issues, changed_paths)


def _context_environment(build_context: BuildContext | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["GO111MODULE"] = "on"
    if build_context is not None:
        environment.update(
            {
                "GOOS": build_context.goos,
                "GOARCH": build_context.goarch,
                "CGO_ENABLED": "1" if build_context.cgo_enabled else "0",
            }
        )
    return environment


def _golangci_tool_errors(payload: dict, stderr: str) -> list[str]:
    errors = []
    report = payload.get("Report") or {}
    for item in report.get("Warnings", []) or []:
        if isinstance(item, dict):
            errors.append(str(item.get("Text") or item))
        else:
            errors.append(str(item))
    if "level=error" in stderr:
        errors.append(stderr.strip())
    return errors


def _go_list_files(
    cwd: Path,
    target: str,
    build_context: BuildContext | None,
) -> tuple[set[str], str | None]:
    command = ["go", "list", "-json"]
    if build_context is not None and build_context.build_tags:
        command.extend(["-tags", ",".join(build_context.build_tags)])
    command.append(target)
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=_context_environment(build_context),
    )
    if result.returncode != 0:
        return set(), result.stderr.strip() or "go list failed"
    try:
        package = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return set(), f"cannot parse go list output: {exc}"
    fields = ("GoFiles", "CgoFiles", "TestGoFiles", "XTestGoFiles")
    files = {
        Path(filename).name
        for field in fields
        for filename in package.get(field, [])
        if isinstance(filename, str)
    }
    return files, None


def _supported_go_platforms() -> set[tuple[str, str]]:
    result = subprocess.run(
        ["go", "tool", "dist", "list"],
        capture_output=True,
        text=True,
        check=False,
        env=_context_environment(None),
    )
    if result.returncode != 0:
        raise ComplexityRatchetError("cannot enumerate supported Go platforms")
    platforms = set()
    for line in result.stdout.splitlines():
        if "/" not in line:
            raise ComplexityRatchetError("invalid platform from go tool dist list")
        goos, goarch = line.split("/", 1)
        platforms.add((goos, goarch))
    if not platforms:
        raise ComplexityRatchetError("go tool dist list returned no platforms")
    return platforms


def _partition_context_files(
    files: list[Path], build_context: BuildContext | None
) -> tuple[list[Path], list[Path], list[str]]:
    covered: list[Path] = []
    missing: list[Path] = []
    errors: list[str] = []
    by_directory: dict[Path, list[Path]] = {}
    for file in files:
        by_directory.setdefault(file.parent, []).append(file)
    for directory, expected in by_directory.items():
        listed, error = _go_list_files(directory, ".", build_context)
        if error:
            errors.append(f"cannot verify Go context in {directory}: {error}")
            continue
        for file in expected:
            (covered if file.name in listed else missing).append(file)
    return covered, missing, errors


def _inactive_context_groups(
    repo_root: Path,
    grouped: dict[Path, list[Path]],
    default_config: Path,
    module_overrides: dict[Path, Path],
) -> tuple[dict[tuple[Path, Path, BuildContext], list[Path]], set[str], list[str]]:
    contexts: dict[tuple[Path, Path, BuildContext], list[Path]] = {}
    covered_paths: set[str] = set()
    errors = []
    supported_platforms = _supported_go_platforms()
    list_cache: dict[tuple[Path, BuildContext], tuple[set[str], str | None]] = {}
    for module_root, files in grouped.items():
        config_path = module_overrides.get(module_root, default_config)
        covered, missing, context_errors = _partition_context_files(files, None)
        if config_path == default_config:
            covered_paths.update(
                file.relative_to(repo_root).as_posix() for file in covered
            )
        errors.extend(context_errors)
        for file in missing:
            try:
                context = _context_loading_file(file, supported_platforms, list_cache)
            except (OSError, UnicodeError, ValueError) as exc:
                relative = file.relative_to(repo_root).as_posix()
                errors.append(f"cannot resolve Go build context for {relative}: {exc}")
                continue
            if context is None:
                errors.append(
                    f"no Go build context loaded {file.relative_to(repo_root)}"
                )
                continue
            contexts.setdefault((module_root, config_path, context), []).append(file)
    return contexts, covered_paths, errors


def _context_loading_file(
    file: Path,
    supported_platforms: set[tuple[str, str]],
    list_cache: dict[tuple[Path, BuildContext], tuple[set[str], str | None]],
) -> BuildContext | None:
    source = file.read_bytes()
    for context in candidate_build_contexts_for_source(
        source, file, supported_platforms
    ):
        cache_key = (file.parent, context)
        if cache_key not in list_cache:
            list_cache[cache_key] = _go_list_files(file.parent, ".", context)
        listed, error = list_cache[cache_key]
        if error is None and file.name in listed:
            return context
    return None


@dataclass
class _GateFindings:
    complexity: list[dict] = field(default_factory=list)
    ordinary: list[dict] = field(default_factory=list)
    covered_paths: set[str] = field(default_factory=set)
    failed: bool = False


def _append_issues(
    findings: _GateFindings,
    repo_root: Path,
    module_root: Path,
    issues: list[dict],
    include_complexity: bool,
) -> None:
    for issue in issues:
        record = go_issue_record(repo_root, module_root, issue)
        if record["linter"] in COMPLEXITY_LINTERS:
            if include_complexity:
                findings.complexity.append(record)
            continue
        findings.ordinary.append(repo_relative_go_issue(repo_root, module_root, issue))


def _lint_groups(
    command: str,
    repo_root: Path,
    grouped: dict[Path, list[Path]],
    default_config: Path,
    module_overrides: dict[Path, Path],
    findings: _GateFindings,
    include_complexity: bool = True,
) -> int:
    for module_root, files in grouped.items():
        config_path = module_overrides.get(module_root, default_config)
        returncode, issues = _collect_module_issues(
            command, repo_root, module_root, config_path, files
        )
        if returncode != 0:
            return returncode
        _append_issues(
            findings,
            repo_root,
            module_root,
            issues,
            include_complexity,
        )
    return 0


def _lint_alternate_contexts(
    command: str,
    repo_root: Path,
    grouped: dict[Path, list[Path]],
    default_config: Path,
    module_overrides: dict[Path, Path],
    findings: _GateFindings,
    include_complexity: bool = True,
) -> int:
    contexts, covered, errors = _inactive_context_groups(
        repo_root, grouped, default_config, module_overrides
    )
    print(
        "Go alternate build contexts: "
        f"groups={len(contexts)} files={sum(len(files) for files in contexts.values())}"
    )
    if include_complexity:
        findings.covered_paths.update(covered)
    if errors:
        findings.failed = True
        for message in errors:
            print(message)
    for (module_root, config_path, context), files in contexts.items():
        verified, missing, verification_errors = _partition_context_files(
            files, context
        )
        if verification_errors or missing:
            findings.failed = True
            for message in verification_errors:
                print(message)
            for file in missing:
                print(
                    f"alternate Go context did not load {file.relative_to(repo_root)}"
                )
            continue
        returncode, issues = _collect_module_issues(
            command,
            repo_root,
            module_root,
            config_path,
            files,
            context,
        )
        if returncode != 0:
            return returncode
        if include_complexity:
            findings.covered_paths.update(
                file.relative_to(repo_root).as_posix() for file in verified
            )
        _append_issues(
            findings,
            repo_root,
            module_root,
            issues,
            include_complexity,
        )
    return 0


def _expanded_lint_groups(
    repo_root: Path,
    grouped: dict[Path, list[Path]],
    contract_changed: bool,
) -> tuple[dict[Path, list[Path]], set[str]]:
    expanded = dict(grouped)
    if not contract_changed:
        return expanded, set()
    tracked_go_files = _tracked_go_files(repo_root)
    full_grouped = group_files_by_module(repo_root, tracked_go_files, "go.mod", {".go"})
    expanded.update(full_grouped)
    unresolved = unresolved_go_paths(repo_root, set(tracked_go_files), full_grouped)
    return expanded, existing_unresolved_go_paths(repo_root, unresolved)


def _override_groups(
    grouped: dict[Path, list[Path]], module_overrides: dict[Path, Path]
) -> dict[Path, list[Path]]:
    return {
        module_root: files
        for module_root, files in grouped.items()
        if module_root in module_overrides
    }


def _lint_ordinary_overrides(
    command: str,
    repo_root: Path,
    grouped: dict[Path, list[Path]],
    default_config: Path,
    module_overrides: dict[Path, Path],
    findings: _GateFindings,
) -> int:
    override_grouped = _override_groups(grouped, module_overrides)
    if not override_grouped:
        return 0
    returncode = _lint_groups(
        command,
        repo_root,
        override_grouped,
        default_config,
        module_overrides,
        findings,
        include_complexity=False,
    )
    if returncode != 0:
        return returncode
    return _lint_alternate_contexts(
        command,
        repo_root,
        override_grouped,
        default_config,
        module_overrides,
        findings,
        include_complexity=False,
    )


def _run_ratchet(
    command: str,
    records: list[dict],
    repo_root: Path,
    default_config: Path,
    debt_manifest: Path,
    changed_go_paths: set[str],
    base_ref: str | None,
    covered_paths: set[str],
    evaluated_paths: set[str],
    require_complete_manifest: bool,
) -> bool:
    effective_base_ref = base_ref or os.getenv("AGENT_BASE_REF") or "origin/main"
    try:
        tool_version = detect_tool_version(command, repo_root)
        result = run_complexity_ratchet(
            records=records,
            repo_root=repo_root,
            config_path=default_config,
            manifest_path=debt_manifest,
            changed_paths=changed_go_paths,
            base_ref=effective_base_ref,
            tool_version=tool_version,
            covered_paths=covered_paths,
            evaluated_paths=evaluated_paths,
            require_complete_manifest=require_complete_manifest,
        )
        print_ratchet_result(result)
        return result.passed
    except ComplexityRatchetError as exc:
        print(f"Go complexity debt contract failed: {exc}")
        return False


def run_go_lint_gate(
    changed_files: list[str],
    base_ref: str | None,
    repo_root: Path,
    default_config: Path,
    debt_manifest: Path,
    module_overrides: dict[Path, Path],
) -> int:
    grouped = group_files_by_module(repo_root, changed_files, "go.mod", {".go"})
    all_changed_go_paths = {
        changed for changed in changed_files if Path(changed).suffix == ".go"
    }
    contract_changed = any(
        is_complexity_contract_path(path, repo_root) for path in changed_files
    )
    unresolved_paths = unresolved_go_paths(repo_root, all_changed_go_paths, grouped)
    unresolved_existing = existing_unresolved_go_paths(repo_root, unresolved_paths)
    changed_go_paths = _complexity_changed_paths(
        grouped,
        unresolved_paths - unresolved_existing,
        repo_root,
    )
    uses_complexity_config = contract_changed or bool(changed_go_paths)
    if not grouped and not uses_complexity_config and not unresolved_existing:
        print("No changed Go files detected.")
        return 0

    findings = _GateFindings(failed=bool(unresolved_existing))
    for path in sorted(unresolved_existing):
        print(f"changed Go file has no owning go.mod and cannot be linted: {path}")
    golangci_lint = resolve_golangci_lint(repo_root)
    lint_grouped, unowned_tracked = _expanded_lint_groups(
        repo_root, grouped, contract_changed
    )
    if unowned_tracked:
        findings.failed = True
        for path in sorted(unowned_tracked):
            print(f"tracked Go file has no owning go.mod and cannot be linted: {path}")
    returncode = _lint_groups(
        golangci_lint,
        repo_root,
        lint_grouped,
        default_config,
        {},
        findings,
    )
    if returncode != 0:
        return returncode
    returncode = _lint_alternate_contexts(
        golangci_lint,
        repo_root,
        lint_grouped,
        default_config,
        {},
        findings,
    )
    if returncode != 0:
        return returncode
    returncode = _lint_ordinary_overrides(
        golangci_lint,
        repo_root,
        grouped,
        default_config,
        module_overrides,
        findings,
    )
    if returncode != 0:
        return returncode
    if findings.ordinary:
        print_go_issues(findings.ordinary)
        print(f"{len(findings.ordinary)} changed-file Go lint issue(s) found.")
        findings.failed = True
    if uses_complexity_config and not _run_ratchet(
        golangci_lint,
        findings.complexity,
        repo_root,
        default_config,
        debt_manifest,
        changed_go_paths,
        base_ref,
        findings.covered_paths,
        findings.covered_paths | changed_go_paths
        if contract_changed
        else changed_go_paths,
        contract_changed,
    ):
        findings.failed = True
    return int(findings.failed)
