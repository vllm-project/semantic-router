#!/usr/bin/env python3
"""Helpers for Go lint execution in agent scripts."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_golangci_lint(repo_root: Path) -> str:
    result = subprocess.run(
        ["go", "env", "GOPATH"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        candidate = Path(result.stdout.strip()) / "bin" / "golangci-lint"
        if candidate.exists():
            return str(candidate)

    binary = shutil.which("golangci-lint")
    if binary:
        return binary
    return "golangci-lint"


def normalize_go_issue_path(repo_root: Path, module_root: Path, filename: str) -> str:
    issue_path = Path(filename)
    if not issue_path.is_absolute():
        issue_path = (module_root / issue_path).resolve()
    return issue_path.relative_to(repo_root).as_posix()


def filter_go_issues(
    repo_root: Path, module_root: Path, issues: list[dict], changed_paths: set[str]
) -> list[dict]:
    filtered: list[dict] = []
    for issue in issues:
        if not isinstance(issue, dict):
            raise ValueError("Go lint issue payload must be an object")
        pos = issue.get("Pos", {})
        if not isinstance(pos, dict):
            raise ValueError("Go lint issue position must be an object")
        filename = pos.get("Filename")
        if not filename:
            raise ValueError("Go lint issue has no filename")
        relative_path = normalize_go_issue_path(repo_root, module_root, filename)
        if relative_path in changed_paths:
            filtered.append(issue)
    return filtered


def go_issues_from_payload(payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("golangci-lint JSON payload must be an object")
    issues = payload.get("Issues")
    if not isinstance(issues, list):
        raise ValueError("golangci-lint JSON payload has no issue list")
    return issues


def go_issue_record(
    repo_root: Path, module_root: Path, issue: dict
) -> dict[str, object]:
    pos = issue.get("Pos", {})
    filename = pos.get("Filename")
    if not filename:
        raise ValueError("Go lint issue has no filename")
    return {
        "path": normalize_go_issue_path(repo_root, module_root, filename),
        "line": int(pos.get("Line", 0)),
        "column": int(pos.get("Column", 0)),
        "linter": str(issue.get("FromLinter", "")),
        "message": str(issue.get("Text", "")),
    }


def repo_relative_go_issue(repo_root: Path, module_root: Path, issue: dict) -> dict:
    normalized = dict(issue)
    position = dict(issue.get("Pos", {}))
    filename = position.get("Filename")
    if not filename:
        raise ValueError("Go lint issue has no filename")
    position["Filename"] = normalize_go_issue_path(repo_root, module_root, filename)
    normalized["Pos"] = position
    return normalized


def print_go_issues(issues: list[dict]) -> None:
    for issue in issues:
        pos = issue.get("Pos", {})
        source_line = ""
        source_lines = issue.get("SourceLines") or []
        if source_lines:
            source_line = f"\n{source_lines[0]}"
        print(
            f"{pos.get('Filename')}:{pos.get('Line')}:{pos.get('Column')}: "
            f"{issue.get('Text')} ({issue.get('FromLinter')}){source_line}"
        )


def extract_golangci_json(stdout: str) -> str:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
    return stdout


def load_golangci_payload(stdout: str) -> dict:
    try:
        return json.loads(extract_golangci_json(stdout or "{}"))
    except json.JSONDecodeError:
        sys.stderr.write(stdout)
        raise
