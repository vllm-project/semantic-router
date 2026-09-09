#!/usr/bin/env python3
"""Run file-scoped checks; only the explicit format command rewrites sources."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_VENV = Path(os.environ.get("AGENT_VENV", REPO_ROOT / ".venv-agent"))
LOCAL_NODE_BIN = AGENT_VENV / "nodeenv" / "bin"
MARKDOWNLINT = AGENT_VENV / "node-tools" / "node_modules" / ".bin" / "markdownlint"
JS_SUFFIXES = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
WHITESPACE_SUFFIXES = {".go", ".rs", ".py", ".js"}


def run(command: list[str], cwd: Path = REPO_ROOT) -> int:
    print(f"+ {' '.join(command)} (cwd={cwd})", flush=True)
    return subprocess.run(command, cwd=cwd, check=False).returncode


def ensure_node_runtime() -> None:
    if shutil.which("node") and shutil.which("npm"):
        return
    subprocess.run(["make", "harness-node-bootstrap"], cwd=REPO_ROOT, check=True)
    if (
        not (LOCAL_NODE_BIN / "node").is_file()
        or not (LOCAL_NODE_BIN / "npm").is_file()
    ):
        raise RuntimeError("harness-node-bootstrap did not provide Node and npm")
    os.environ["PATH"] = f"{LOCAL_NODE_BIN}{os.pathsep}{os.environ['PATH']}"


def owning_directory(path: str, marker: str) -> Path:
    directory = (REPO_ROOT / path).resolve().parent
    root = REPO_ROOT.resolve()
    while directory == root or root in directory.parents:
        if (directory / marker).is_file():
            return directory
        directory = directory.parent
    raise ValueError(f"No owning {marker} for {path}")


def grouped_files(files: list[str], marker: str) -> dict[Path, list[str]]:
    grouped: dict[Path, list[str]] = defaultdict(list)
    for file in files:
        owner = owning_directory(file, marker)
        grouped[owner].append(
            (REPO_ROOT / file).resolve().relative_to(owner).as_posix()
        )
    return grouped


def run_whitespace(files: list[str], fix: bool = False) -> int:
    failed = False
    for file in files:
        path = REPO_ROOT / file
        original = path.read_bytes()
        if not original:
            continue
        lines = original.splitlines(keepends=True)
        normalized = (
            b"".join(
                line.rstrip(b"\r\n").rstrip(b" \t") + b"\n" for line in lines
            ).rstrip(b"\n")
            + b"\n"
        )
        if original == normalized:
            continue
        if fix:
            path.write_bytes(normalized)
        else:
            print(
                f"{file}: trailing whitespace, missing newline, or extra blank lines at EOF"
            )
            failed = True
    return int(failed)


def run_go(files: list[str], fix: bool = False) -> int:
    if fix:
        return run(["gofmt", "-w", *files])
    result = subprocess.run(
        ["gofmt", "-l", *files],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode or int(bool(result.stdout.strip()))


def run_python(files: list[str], fix: bool = False) -> int:
    return run([sys.executable, "-m", "black", *([] if fix else ["--check"]), *files])


def run_rust(files: list[str], fix: bool = False) -> int:
    for crate in grouped_files(files, "Cargo.toml"):
        command = ["cargo", "fmt", "--manifest-path", str(crate / "Cargo.toml")]
        if not fix:
            command.extend(["--", "--check"])
        if code := run(command, crate):
            return code
    return 0


def ensure_package_dependencies(package: Path) -> None:
    lock = package / "package-lock.json"
    stamp = package / "node_modules" / ".agent-package-lock.json"
    if stamp.is_file() and lock.is_file() and stamp.read_bytes() == lock.read_bytes():
        return
    subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=package, check=True)
    shutil.copyfile(lock, stamp)


def run_javascript(files: list[str], fix: bool = False) -> int:
    ensure_node_runtime()
    packages: dict[Path, list[str]] = defaultdict(list)
    for file in files:
        try:
            package = owning_directory(file, "package.json")
        except ValueError:
            if Path(file).suffix not in {".js", ".mjs", ".cjs"}:
                raise
            # Standalone Node tools have no package ESLint configuration.
            if code := run(["node", "--check", file]):
                return code
            continue
        packages[package].append(
            (REPO_ROOT / file).resolve().relative_to(package).as_posix()
        )
    for package, relative_files in packages.items():
        scripts = json.loads((package / "package.json").read_text()).get("scripts", {})
        ensure_package_dependencies(package)
        if "lint" in scripts:
            command = [
                str(package / "node_modules" / ".bin" / "eslint"),
                *(["--fix"] if fix else []),
                *relative_files,
            ]
        elif "check" in scripts and not fix:
            command = ["npm", "run", "check"]
        elif fix and (package / "node_modules" / ".bin" / "prettier").is_file():
            command = [
                str(package / "node_modules" / ".bin" / "prettier"),
                "--write",
                *relative_files,
            ]
        else:
            raise ValueError(
                f"No {'formatter' if fix else 'lint or check script'} in {package / 'package.json'}"
            )
        if code := run(command, package):
            return code
    return 0


def run_markdownlint(files: list[str]) -> int:
    ensure_node_runtime()
    subprocess.run(["make", "harness-markdown-bootstrap"], cwd=REPO_ROOT, check=True)
    return run(
        [str(MARKDOWNLINT), "-c", "tools/linter/markdown/markdownlint.yaml", *files]
    )


def run_shell(files: list[str]) -> int:
    config = REPO_ROOT / "tools/linter/shellcheck/.shellcheckrc"
    disabled = next(
        line.removeprefix("disable=")
        for line in config.read_text().splitlines()
        if line.startswith("disable=")
    )
    return run(["shellcheck", "-e", disabled, *files])


def run_codespell(files: list[str]) -> int:
    skip = ",".join(
        (REPO_ROOT / "tools/linter/codespell/.codespell.skip").read_text().splitlines()
    )
    return run(
        [
            str(AGENT_VENV / "bin" / "codespell"),
            "--skip",
            skip,
            "--ignore-words",
            "tools/linter/codespell/.codespell.ignorewords",
            "--check-filenames",
            *files,
        ]
    )


def run_format(files: list[str]) -> int:
    whitespace = [file for file in files if Path(file).suffix in WHITESPACE_SUFFIXES]
    run_whitespace(whitespace, fix=True)
    for suffixes, formatter in (
        ({".go"}, run_go),
        ({".py"}, run_python),
        ({".rs"}, run_rust),
        (JS_SUFFIXES, run_javascript),
    ):
        selected = [file for file in files if Path(file).suffix in suffixes]
        if selected and (code := formatter(selected, fix=True)):
            return code
    return 0


def main() -> int:
    checks = {
        "whitespace": run_whitespace,
        "go": run_go,
        "python": run_python,
        "rust": run_rust,
        "javascript": run_javascript,
        "markdown": run_markdownlint,
        "shell": run_shell,
        "codespell": run_codespell,
        "format": run_format,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=checks)
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    files = [file for file in args.files if (REPO_ROOT / file).is_file()]
    if not files:
        return 0
    try:
        return checks[args.tool](files)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
