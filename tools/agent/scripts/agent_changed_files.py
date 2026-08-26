#!/usr/bin/env python3
"""Changed-file input parsing for agent harness commands."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path, PurePosixPath

from agent_support import REPO_ROOT


def split_changed_files(raw: str | None) -> list[str]:
    """Parse the legacy comma/whitespace CLI form.

    Exact paths containing separators must use ``--changed-files-path``. Git
    discovery and path-file input never pass through this legacy parser.
    """

    if not raw:
        return []
    parts = re.split(r"[\s,]+", raw)
    return normalize_changed_paths(part for part in parts if part)


def normalize_changed_path(raw_path: str, repo_root: Path | None = None) -> str:
    root = (repo_root or REPO_ROOT).resolve()
    path = raw_path
    if not path or any(character in path for character in ("\x00", "\n", "\r")):
        raise ValueError("changed file paths must be non-empty single-line paths")
    while path.startswith("./"):
        path = path[2:]
    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or "\\" in path
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.as_posix() != path
    ):
        raise ValueError(
            f"changed file path must be canonical repo-relative POSIX: {path!r}"
        )
    resolved = (root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"changed file path escapes the repository: {path!r}") from exc
    return path


def normalize_changed_paths(raw_paths, repo_root: Path | None = None) -> list[str]:
    cleaned = [
        normalize_changed_path(raw_path, repo_root)
        for raw_path in raw_paths
        if raw_path != ""
    ]
    return sorted(dict.fromkeys(cleaned))


def load_changed_files(changed_files_path: str | None) -> list[str] | None:
    if not changed_files_path:
        return None

    path = Path(changed_files_path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    try:
        raw = path.read_bytes()
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise ValueError(
            f"unable to read changed files from '{path}': {reason}"
        ) from exc
    if b"\x00" in raw:
        raise ValueError(f"changed-files path '{path}' must be newline-delimited")
    try:
        lines = raw.decode("utf-8").split("\n")
    except UnicodeDecodeError as exc:
        raise ValueError(f"changed-files path '{path}' is not UTF-8") from exc
    return normalize_changed_paths(lines)


def git_changed_files(base_ref: str | None) -> list[str]:
    explicit_base_ref = base_ref or os.getenv("AGENT_BASE_REF")
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
    base_exists = run_git("rev-parse", "--verify", base_ref).returncode == 0
    if base_exists:
        result = run_git("merge-base", "HEAD", base_ref)
        if result.returncode == 0:
            merge_base = result.stdout.strip()

    if explicit_base_ref and not merge_base:
        raise ValueError(
            f"unable to resolve an explicit changed-file base: {explicit_base_ref}"
        )

    if not merge_base:
        result = run_git("rev-parse", "--verify", "HEAD^")
        if result.returncode == 0:
            merge_base = "HEAD^"

    if not merge_base:
        return []

    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            f"{merge_base}...HEAD",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("unable to enumerate changed files from git diff")
    try:
        paths = [part.decode("utf-8") for part in result.stdout.split(b"\x00") if part]
    except UnicodeDecodeError as exc:
        raise ValueError("git diff returned a non-UTF-8 changed path") from exc

    return normalize_changed_paths(paths)


def get_changed_files(
    explicit: str | None,
    base_ref: str | None,
    changed_files_path: str | None = None,
) -> list[str]:
    files = split_changed_files(explicit) if explicit and explicit.strip() else []
    if not files and changed_files_path:
        files = load_changed_files(changed_files_path) or []
    if files:
        return files
    return git_changed_files(base_ref)
