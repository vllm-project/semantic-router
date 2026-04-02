#!/usr/bin/env python3
"""Run repo-native changed-file agent lint from a pre-commit hook."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from agent_resolution import git_changed_files, split_changed_files

REPO_ROOT = Path(__file__).resolve().parents[3]
SKIP_ENV = "AGENT_SKIP_PRECOMMIT_CHANGED_LINT"
MAX_PRECOMMIT_PATHS = 256
MAX_CHANGED_FILES_ARG_CHARS = 8_192


def should_fallback_to_git_diff(changed_files: list[str]) -> bool:
    serialized_len = sum(len(path) + 1 for path in changed_files)
    return (
        len(changed_files) > MAX_PRECOMMIT_PATHS
        or serialized_len > MAX_CHANGED_FILES_ARG_CHARS
    )


def diff_fallback_base_refs() -> list[str | None]:
    candidates: list[str | None] = []
    env_base_ref = os.getenv("AGENT_BASE_REF")
    if env_base_ref:
        candidates.append(env_base_ref)
    candidates.extend([None, "HEAD^"])

    unique_candidates: list[str | None] = []
    seen: set[str | None] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def resolve_changed_files() -> list[str]:
    explicit_files = split_changed_files("\n".join(sys.argv[1:]))
    if explicit_files and not should_fallback_to_git_diff(explicit_files):
        return explicit_files

    for base_ref in diff_fallback_base_refs():
        diff_files = git_changed_files(base_ref)
        if diff_files:
            return diff_files
    return explicit_files


def write_changed_files(changed_files: list[str]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="agent-precommit-changed-files-",
        suffix=".txt",
        delete=False,
    ) as handle:
        handle.write("\n".join(changed_files))
        return handle.name


def main() -> int:
    if os.getenv(SKIP_ENV) == "1":
        return 0

    changed_files = resolve_changed_files()
    if not changed_files:
        return 0

    changed_files_path = write_changed_files(changed_files)
    env = {**os.environ, "AGENT_CHANGED_FILES_PATH": changed_files_path}
    try:
        result = subprocess.run(
            [
                "make",
                "agent-lint",
                "AGENT_SKIP_PRECOMMIT_BASELINE=1",
            ],
            cwd=REPO_ROOT,
            check=False,
            env=env,
        )
        return result.returncode
    finally:
        Path(changed_files_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
