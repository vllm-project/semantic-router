#!/usr/bin/env python3
"""Locate language module roots for changed repository files."""

from __future__ import annotations

from pathlib import Path


def nearest_manifest_root(
    repo_root: Path, path: Path, manifest_name: str
) -> Path | None:
    root = repo_root.resolve()
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    current = resolved.parent
    while True:
        if (current / manifest_name).exists():
            return current
        if current == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def group_files_by_module(
    repo_root: Path,
    changed_files: list[str],
    manifest_name: str,
    extensions: set[str],
) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for changed in changed_files:
        path = (repo_root / changed).resolve(strict=False)
        try:
            path.relative_to(repo_root.resolve())
        except ValueError:
            continue
        if path.suffix not in extensions or not path.exists():
            continue
        module_root = nearest_manifest_root(repo_root, path, manifest_name)
        if module_root is not None:
            grouped.setdefault(module_root, []).append(path)
    return grouped
