#!/usr/bin/env python3
"""Synchronize Evaluation runtime catalog mirrors from Python package data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_CATALOG_ROOT = REPO_ROOT / "src" / "vllm-sr" / "cli" / "evaluation" / "golden"


@dataclass(frozen=True)
class CatalogMirrors:
    canonical: Path
    mirrors: tuple[Path, ...]


CATALOGS = (
    CatalogMirrors(
        canonical=PYTHON_CATALOG_ROOT / "metric_analysis_catalog.v1.json",
        mirrors=(
            REPO_ROOT
            / "dashboard"
            / "backend"
            / "evaluationplane"
            / "metric_analysis_catalog.v1.json",
            REPO_ROOT
            / "dashboard"
            / "frontend"
            / "src"
            / "contracts"
            / "metric_analysis_catalog.v1.json",
        ),
    ),
    CatalogMirrors(
        canonical=PYTHON_CATALOG_ROOT / "research_benchmark_inventory.v1.json",
        mirrors=(
            REPO_ROOT
            / "dashboard"
            / "backend"
            / "evaluationplane"
            / "research_benchmark_inventory.v1.json",
        ),
    ),
)


def _relative(path: Path) -> Path:
    return path.relative_to(REPO_ROOT)


def _canonical_bytes(catalog: CatalogMirrors) -> bytes:
    if catalog.canonical.is_symlink() or not catalog.canonical.is_file():
        raise RuntimeError(
            "canonical Evaluation catalog must be a regular file: "
            f"{_relative(catalog.canonical)}"
        )
    try:
        return catalog.canonical.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read canonical Evaluation catalog {_relative(catalog.canonical)}: {exc}"
        ) from exc


def check() -> int:
    errors: list[str] = []
    for catalog in CATALOGS:
        try:
            canonical = _canonical_bytes(catalog)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        for mirror in catalog.mirrors:
            if mirror.is_symlink() or not mirror.is_file():
                errors.append(
                    "Evaluation catalog mirror must be a regular file: "
                    f"{_relative(mirror)}"
                )
                continue
            try:
                mirrored = mirror.read_bytes()
            except OSError as exc:
                errors.append(
                    f"missing Evaluation catalog mirror {_relative(mirror)}: {exc}"
                )
                continue
            if mirrored != canonical:
                errors.append(
                    "stale Evaluation catalog mirror "
                    f"{_relative(mirror)}; canonical={_relative(catalog.canonical)}"
                )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(
            "run: python3 tools/ci/sync_evaluation_catalogs.py",
            file=sys.stderr,
        )
        return 1
    return 0


def sync() -> int:
    try:
        for catalog in CATALOGS:
            canonical = _canonical_bytes(catalog)
            for mirror in catalog.mirrors:
                if mirror.is_symlink() or not mirror.is_file():
                    raise RuntimeError(
                        "Evaluation catalog mirror must be a regular file: "
                        f"{_relative(mirror)}"
                    )
                mirror.write_bytes(canonical)
                print(f"synced {_relative(catalog.canonical)} -> {_relative(mirror)}")
    except (OSError, RuntimeError) as exc:
        print(f"cannot synchronize Evaluation catalog mirrors: {exc}", file=sys.stderr)
        return 1
    return check()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="reject runtime mirrors that differ from canonical Python package data",
    )
    args = parser.parse_args()
    return check() if args.check else sync()


if __name__ == "__main__":
    raise SystemExit(main())
