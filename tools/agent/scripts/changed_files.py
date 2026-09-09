#!/usr/bin/env python3
"""Changed-file input parsing for agent harness commands."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from check_support import REPO_ROOT


def split_changed_files(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[\s,]+", raw)
    cleaned = [normalize_changed_path(part) for part in parts if part.strip()]
    return sorted(dict.fromkeys(cleaned))


def normalize_changed_path(raw_path: str) -> str:
    path = raw_path.strip()
    while path.startswith("./"):
        path = path[2:]
    return path


def load_changed_files(changed_files_path: str | None) -> str | None:
    if not changed_files_path:
        return None

    path = Path(changed_files_path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise ValueError(
            f"unable to read changed files from '{path}': {reason}"
        ) from exc


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def resolve_base_ref(base_ref: str | None) -> str | None:
    git_output("rev-parse", "--git-dir")
    requested = base_ref or os.getenv("BASE_REF")
    candidates = (requested,) if requested else ("origin/main", "HEAD^")
    for candidate in candidates:
        if run_git("rev-parse", "--verify", f"{candidate}^{{commit}}").returncode == 0:
            return candidate
    if requested:
        raise ValueError(f"unable to resolve requested base revision '{requested}'")
    return None


def git_output(*args: str) -> str:
    result = run_git(*args)
    if result.returncode:
        reason = result.stderr.strip() or f"exit status {result.returncode}"
        raise ValueError(f"git {' '.join(args)} failed: {reason}")
    return result.stdout


def git_changed_files(base_ref: str | None) -> list[str]:
    base_ref = resolve_base_ref(base_ref)

    changed: set[str] = set()
    if base_ref:
        merge_base = git_output("merge-base", "HEAD", base_ref).strip()
        output = git_output("diff", "--name-only", "-z", f"{merge_base}...HEAD")
        changed.update(path for path in output.split("\0") if path)

    # Local checks must include work that has not been committed yet. `git diff
    # HEAD` covers staged and unstaged tracked paths; the final query adds
    # untracked paths without pulling ignored build artifacts into the result.
    has_head = run_git("rev-parse", "--verify", "HEAD").returncode == 0
    args = (
        ("diff", "--name-only", "-z", "HEAD")
        if has_head
        else ("diff", "--cached", "--name-only", "-z")
    )
    output = git_output(*args)
    changed.update(path for path in output.split("\0") if path)

    output = git_output("ls-files", "--others", "--exclude-standard", "-z")
    changed.update(path for path in output.split("\0") if path)

    return sorted(changed)


def get_changed_files(
    explicit: str | None,
    base_ref: str | None,
    changed_files_path: str | None = None,
) -> list[str]:
    if explicit and explicit.strip():
        return split_changed_files(explicit)
    if changed_files_path:
        # A supplied file is authoritative, including an empty selection. Keep
        # spaces in paths from the newline-delimited GitHub changed-file list.
        raw = load_changed_files(changed_files_path) or ""
        return sorted(
            {normalize_changed_path(path) for path in raw.splitlines() if path.strip()}
        )
    return git_changed_files(base_ref)
