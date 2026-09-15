#!/usr/bin/env python3
"""Format changed Go files with golangci-lint fmt, one invocation per Go module."""

from __future__ import annotations

import subprocess
import sys

from check_support import GO_LINT_CONFIG, REPO_ROOT, group_files_by_module
from go_lint_support import resolve_golangci_lint


def run_go_fmt(changed_files: list[str]) -> int:
    grouped = group_files_by_module(changed_files, "go.mod", {".go"})
    if not grouped:
        return 0
    golangci_lint = resolve_golangci_lint(REPO_ROOT)
    for module_root, files in grouped.items():
        command = [
            golangci_lint,
            "fmt",
            "--config",
            str(GO_LINT_CONFIG),
            *sorted(file.relative_to(module_root).as_posix() for file in files),
        ]
        result = subprocess.run(command, cwd=module_root, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(run_go_fmt(sys.argv[1:]))
