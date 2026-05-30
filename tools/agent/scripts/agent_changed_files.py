#!/usr/bin/env python3
"""Changed-file input parsing for agent harness commands."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from agent_support import REPO_ROOT


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


def git_changed_files(base_ref: str | None) -> list[str]:
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
    if run_git("rev-parse", "--verify", base_ref).returncode == 0:
        result = run_git("merge-base", "HEAD", base_ref)
        if result.returncode == 0:
            merge_base = result.stdout.strip()

    if not merge_base:
        result = run_git("rev-parse", "--verify", "HEAD^")
        if result.returncode == 0:
            merge_base = "HEAD^"

    if not merge_base:
        return []

    result = run_git("diff", "--name-only", f"{merge_base}...HEAD")
    if result.returncode != 0:
        return []

    return split_changed_files(result.stdout)


def get_changed_files(
    explicit: str | None,
    base_ref: str | None,
    changed_files_path: str | None = None,
) -> list[str]:
    raw_changed_files = explicit
    if raw_changed_files is None or (
        changed_files_path and not raw_changed_files.strip()
    ):
        raw_changed_files = load_changed_files(changed_files_path)

    files = split_changed_files(raw_changed_files)
    if files:
        return files
    return git_changed_files(base_ref)
