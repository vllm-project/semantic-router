#!/usr/bin/env python3
"""Format changed Go files with golangci-lint fmt, one invocation per Go module.

Go sources outside every module root, such as the repository tools built from
src/semantic-router in tools/make/go-tools.mk, are formatted from the Router
module so they keep the same coverage gofmt -w used to provide.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from check_support import GO_LINT_CONFIG, REPO_ROOT, group_files_by_module
from go_lint_support import resolve_golangci_lint

ROUTER_MODULE_ROOT = REPO_ROOT / "src" / "semantic-router"


def group_go_files_for_fmt(changed_files: list[str]) -> dict[Path, list[Path]]:
    grouped = group_files_by_module(changed_files, "go.mod", {".go"})
    covered = {file for files in grouped.values() for file in files}
    for changed in changed_files:
        path = REPO_ROOT / changed
        if path.suffix == ".go" and path.exists() and path not in covered:
            grouped.setdefault(ROUTER_MODULE_ROOT, []).append(path)
    return grouped


def relative_to_module(module_root: Path, file: Path) -> str:
    return Path(os.path.relpath(file, module_root)).as_posix()


def run_go_fmt(changed_files: list[str]) -> int:
    grouped = group_go_files_for_fmt(changed_files)
    if not grouped:
        return 0
    golangci_lint = resolve_golangci_lint(REPO_ROOT)
    for module_root, files in grouped.items():
        command = [
            golangci_lint,
            "fmt",
            "--config",
            str(GO_LINT_CONFIG),
            *sorted(relative_to_module(module_root, file) for file in files),
        ]
        result = subprocess.run(command, cwd=module_root, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(run_go_fmt(sys.argv[1:]))
