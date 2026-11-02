"""Require a current uv.lock beside every training pyproject.toml."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "src" / "training"

# Installed environments are not training projects. A local .venv can vendor
# another package's pyproject.toml, and that file must not fail the check.
_SKIPPED_DIRECTORIES = frozenset(
    {"venv", ".venv", "site-packages", "__pycache__", "node_modules"}
)


class TrainingLockError(Exception):
    """A training project is missing a current uv.lock."""


def discover_projects(training_root: Path) -> list[Path]:
    """Return every training directory that contains a pyproject.toml."""
    if not training_root.is_dir():
        return []
    projects: list[Path] = []
    for directory, children, filenames in os.walk(training_root):
        children[:] = sorted(
            name
            for name in children
            if name not in _SKIPPED_DIRECTORIES and not name.startswith(".")
        )
        if "pyproject.toml" in filenames:
            projects.append(Path(directory))
    projects.sort()
    return projects


def project_folder(directory: Path, repo_root: Path) -> str:
    return directory.resolve().relative_to(repo_root.resolve()).as_posix()


def check_project(directory: Path, repo_root: Path, *, uv: str = "uv") -> None:
    """Fail when pyproject.toml has no sibling lock or the lock is stale."""
    folder = project_folder(directory, repo_root)
    if not (directory / "uv.lock").is_file():
        raise TrainingLockError(
            f"{folder}: pyproject.toml has no sibling uv.lock. "
            f"Run `uv lock` in {folder}."
        )
    try:
        result = subprocess.run(
            [uv, "lock", "--check"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TrainingLockError(
            f"{folder}: could not run `{uv} lock --check`. Run `uv lock` in {folder}."
        ) from exc
    if result.returncode:
        output = "\n".join(
            chunk.strip()
            for chunk in (result.stdout, result.stderr)
            if chunk and chunk.strip()
        )
        message = f"{folder}: uv.lock is not current. Run `uv lock` in {folder}."
        if output:
            message = f"{message}\n{output}"
        raise TrainingLockError(message)


def check_training_locks(repo_root: Path, *, uv: str = "uv") -> list[Path]:
    """Check every training project. An empty walk is a failure."""
    projects = discover_projects(repo_root / "src" / "training")
    if not projects:
        raise TrainingLockError(
            "no training pyproject.toml files were found under src/training"
        )
    failures: list[str] = []
    for directory in projects:
        try:
            check_project(directory, repo_root, uv=uv)
        except TrainingLockError as exc:
            failures.append(str(exc))
    if failures:
        raise TrainingLockError("\n".join(failures))
    return projects
