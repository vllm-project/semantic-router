#!/usr/bin/env python3
"""Run Node-based pre-commit tools through the repo-local agent runtime."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_VENV = Path(os.environ.get("AGENT_VENV", REPO_ROOT / ".venv-agent"))
LOCAL_NODE_BIN = AGENT_VENV / "nodeenv" / "bin"
MARKDOWNLINT = AGENT_VENV / "node-tools" / "node_modules" / ".bin" / "markdownlint"
WEBSITE_DIR = REPO_ROOT / "website"
ESLINT_CONFIG_NAMES = ("eslint.config.js", "eslint.config.mjs", "eslint.config.cjs")
JAVASCRIPT_SUFFIXES = {".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx"}


def run_make(target: str) -> None:
    subprocess.run(["make", target], cwd=REPO_ROOT, check=True)


def ensure_node_runtime() -> None:
    if shutil.which("node") and shutil.which("npm"):
        return
    run_make("harness-node-bootstrap")
    local_node = LOCAL_NODE_BIN / "node"
    local_npm = LOCAL_NODE_BIN / "npm"
    if not local_node.is_file() or not local_npm.is_file():
        raise RuntimeError("harness-node-bootstrap did not provide npm")
    os.environ["PATH"] = f"{LOCAL_NODE_BIN}{os.pathsep}{os.environ['PATH']}"


def resolve_npm() -> str:
    ensure_node_runtime()
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("Node runtime did not provide npm")
    return npm


def project_dependencies_current(project: Path) -> bool:
    lock = project / "package-lock.json"
    stamp = project / "node_modules" / ".agent-package-lock.json"
    return (
        stamp.is_file() and lock.is_file() and lock.read_bytes() == stamp.read_bytes()
    )


def run_markdownlint(files: list[str]) -> int:
    if not files:
        return 0
    ensure_node_runtime()
    run_make("harness-markdown-bootstrap")
    command = [
        str(MARKDOWNLINT),
        "-c",
        "tools/linter/markdown/markdownlint.yaml",
        *files,
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def run_project_lint(project: Path, files: list[str]) -> int:
    npm = resolve_npm()
    if not project_dependencies_current(project):
        install = subprocess.run(
            [npm, "ci", "--no-audit", "--no-fund"],
            cwd=project,
            check=False,
        )
        if install.returncode != 0:
            return install.returncode
        stamp = project / "node_modules" / ".agent-package-lock.json"
        stamp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / "package-lock.json", stamp)
    return subprocess.run(
        [npm, "exec", "--no", "--", "eslint", "--", *files],
        cwd=project,
        check=False,
    ).returncode


def javascript_projects(files: list[str]) -> dict[Path, list[str]]:
    """Resolve changed files against their nearest checked-in ESLint project."""
    projects: dict[Path, list[str]] = {}
    for filename in files:
        path = (REPO_ROOT / filename).resolve()
        if not path.is_file() or path.suffix not in JAVASCRIPT_SUFFIXES:
            continue
        for project in path.parents:
            if not project.is_relative_to(REPO_ROOT):
                break
            if (project / "package.json").is_file() and any(
                (project / name).is_file() for name in ESLINT_CONFIG_NAMES
            ):
                projects.setdefault(project, []).append(
                    path.relative_to(project).as_posix()
                )
                break
    return projects


def run_javascript_lint(files: list[str]) -> int:
    returncode = 0
    for project, changed in javascript_projects(files).items():
        result = run_project_lint(project, list(dict.fromkeys(changed)))
        if result and not returncode:
            returncode = result
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=("markdown", "website", "javascript"))
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    if args.tool == "markdown":
        return run_markdownlint(args.files)
    if args.tool == "javascript":
        return run_javascript_lint(args.files)
    return run_project_lint(WEBSITE_DIR, args.files or ["."])


if __name__ == "__main__":
    raise SystemExit(main())
