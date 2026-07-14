#!/usr/bin/env python3
"""Discover repository-owned public source documentation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PUBLIC_DOCUMENT_SUFFIXES = frozenset({".md", ".mdx", ".html", ".htm"})

# These trees are not source documentation shipped for public consumption.
# Keep this list path-based and narrow; content exceptions would make it easy
# to hide a real credential pair behind an inline suppression comment.
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".agent-harness",
        ".cache",
        ".docusaurus",
        ".git",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".venv-agent",
        "__fixtures__",
        "build",
        "coverage",
        "dist",
        "fixtures",
        "generated",
        "htmlcov",
        "node_modules",
        "out",
        "site",
        "target",
        "test-fixtures",
        "testdata",
        "vendor",
    }
)


def iter_public_document_paths(repo_root: Path) -> list[Path]:
    """Return public source documents, excluding fixtures and derived/private trees."""
    git_paths = _git_visible_document_paths(repo_root)
    if git_paths is not None:
        return git_paths

    paths: list[Path] = []
    for current_root, directories, files in os.walk(repo_root):
        directories[:] = sorted(
            directory
            for directory in directories
            if directory.casefold() not in EXCLUDED_DIRECTORY_NAMES
        )
        current = Path(current_root)
        for filename in sorted(files):
            path = current / filename
            if (
                path.suffix.casefold() in PUBLIC_DOCUMENT_SUFFIXES
                and not path.is_symlink()
            ):
                paths.append(path)
    return paths


def _git_visible_document_paths(repo_root: Path) -> list[Path] | None:
    """Use Git visibility when available so ignored working files stay private."""
    try:
        result = subprocess.run(
            (
                "git",
                "-C",
                str(repo_root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ),
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None

    paths: list[Path] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative_path = Path(os.fsdecode(raw_path))
        if relative_path.suffix.casefold() not in PUBLIC_DOCUMENT_SUFFIXES:
            continue
        if any(
            part.casefold() in EXCLUDED_DIRECTORY_NAMES
            for part in relative_path.parts[:-1]
        ):
            continue
        path = repo_root / relative_path
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    return sorted(paths)
