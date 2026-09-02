#!/usr/bin/env python3
"""Verify that deployment assets are classified in the public support matrix."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MATRIX_PATH = Path("website/docs/installation/support-matrix.md")
CLASSIFICATIONS = {
    "Maintained reference stack",
    "Supported integration",
    "Experimental example",
    "Deprecated",
}
MATRIX_ROW = re.compile(
    r"^\|\s*`(?P<path>(?:deploy|config/runtime)/[^`]+/)`\s*"
    r"\|\s*(?P<classification>[^|]+?)\s*\|"
)


def discover_assets(repo_root: Path) -> set[str]:
    """Return deploy and runtime-config directories requiring classification."""
    assets = {
        f"deploy/{path.name}/"
        for path in (repo_root / "deploy").iterdir()
        if path.is_dir() and path.name != "kubernetes"
    }
    assets.update(
        f"deploy/kubernetes/{path.name}/"
        for path in (repo_root / "deploy" / "kubernetes").iterdir()
        if path.is_dir()
    )
    assets.update(
        f"config/runtime/{path.name}/"
        for path in (repo_root / "config" / "runtime").iterdir()
        if path.is_dir()
    )
    return assets


def parse_matrix(matrix_path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse classified asset rows and return entries plus duplicate paths."""
    entries: dict[str, str] = {}
    duplicates: list[str] = []
    for line in matrix_path.read_text(encoding="utf-8").splitlines():
        match = MATRIX_ROW.match(line)
        if match is None:
            continue
        path = match.group("path")
        if path in entries:
            duplicates.append(path)
        entries[path] = match.group("classification").strip()
    return entries, duplicates


def validate(repo_root: Path, matrix_path: Path | None = None) -> list[str]:
    """Return human-readable validation errors."""
    matrix_path = matrix_path or repo_root / MATRIX_PATH
    if not matrix_path.is_file():
        return [f"support matrix is missing: {matrix_path}"]

    expected = discover_assets(repo_root)
    entries, duplicates = parse_matrix(matrix_path)
    actual = set(entries)
    errors: list[str] = []

    if duplicates:
        errors.append("duplicate classifications: " + ", ".join(sorted(duplicates)))

    missing = expected - actual
    if missing:
        errors.append("unclassified assets: " + ", ".join(sorted(missing)))

    unexpected = actual - expected
    if unexpected:
        errors.append(
            "matrix entries without asset directories: " + ", ".join(sorted(unexpected))
        )

    invalid = {
        f"{path} ({classification})"
        for path, classification in entries.items()
        if classification not in CLASSIFICATIONS
    }
    if invalid:
        errors.append("invalid classifications: " + ", ".join(sorted(invalid)))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        help="support matrix path (default: repository public documentation)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    matrix_path = args.matrix.resolve() if args.matrix else None
    errors = validate(repo_root, matrix_path)
    if errors:
        for error in errors:
            print(f"deployment support matrix: {error}")
        return 1

    print(
        "deployment support matrix: "
        f"all {len(discover_assets(repo_root))} assets are classified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
