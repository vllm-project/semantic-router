#!/usr/bin/env python3
"""Run repo-native changed-file agent lint from a pre-commit hook."""

from __future__ import annotations

import os
import subprocess
import sys
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


def resolve_changed_files() -> list[str]:
    explicit_files = split_changed_files("\n".join(sys.argv[1:]))
    if explicit_files and not should_fallback_to_git_diff(explicit_files):
        return explicit_files

    diff_files = git_changed_files(None)
    if diff_files:
        return diff_files
    return explicit_files


def main() -> int:
    if os.getenv(SKIP_ENV) == "1":
        return 0

    changed_files = resolve_changed_files()
    if not changed_files:
        return 0

    changed_csv = ",".join(changed_files)
    result = subprocess.run(
        [
            "make",
            "agent-lint",
            "AGENT_SKIP_PRECOMMIT_BASELINE=1",
            f"CHANGED_FILES={changed_csv}",
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
