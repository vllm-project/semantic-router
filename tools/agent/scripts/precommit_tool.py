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
WEBSITE_LOCK = WEBSITE_DIR / "package-lock.json"
WEBSITE_STAMP = WEBSITE_DIR / "node_modules" / ".agent-package-lock.json"


def run_make(target: str) -> None:
    subprocess.run(["make", target], cwd=REPO_ROOT, check=True)


def ensure_node_runtime() -> None:
    if shutil.which("node") and shutil.which("npm"):
        return
    run_make("agent-node-bootstrap")
    local_node = LOCAL_NODE_BIN / "node"
    local_npm = LOCAL_NODE_BIN / "npm"
    if not local_node.is_file() or not local_npm.is_file():
        raise RuntimeError("agent-node-bootstrap did not provide npm")
    os.environ["PATH"] = f"{LOCAL_NODE_BIN}{os.pathsep}{os.environ['PATH']}"


def resolve_npm() -> str:
    ensure_node_runtime()
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("Node runtime did not provide npm")
    return npm


def website_dependencies_current() -> bool:
    return (
        WEBSITE_STAMP.is_file()
        and WEBSITE_LOCK.read_bytes() == WEBSITE_STAMP.read_bytes()
    )


def run_markdownlint(files: list[str]) -> int:
    if not files:
        return 0
    ensure_node_runtime()
    run_make("agent-markdown-bootstrap")
    command = [
        str(MARKDOWNLINT),
        "-c",
        "tools/linter/markdown/markdownlint.yaml",
        *files,
    ]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def run_website_lint() -> int:
    npm = resolve_npm()
    if not website_dependencies_current():
        install = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund"],
            cwd=WEBSITE_DIR,
            check=False,
        )
        if install.returncode != 0:
            return install.returncode
        WEBSITE_STAMP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WEBSITE_LOCK, WEBSITE_STAMP)
    return subprocess.run([npm, "run", "lint"], cwd=WEBSITE_DIR, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=("markdown", "website"))
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    if args.tool == "markdown":
        return run_markdownlint(args.files)
    return run_website_lint()


if __name__ == "__main__":
    raise SystemExit(main())
