#!/usr/bin/env python3
"""Run repo-native changed-file agent lint from a pre-commit hook."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from agent_resolution import split_changed_files

REPO_ROOT = Path(__file__).resolve().parents[3]
SKIP_ENV = "AGENT_SKIP_PRECOMMIT_CHANGED_LINT"


def main() -> int:
    if os.getenv(SKIP_ENV) == "1":
        return 0

    changed_files = split_changed_files("\n".join(sys.argv[1:]))
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
