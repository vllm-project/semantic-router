#!/usr/bin/env python3
"""Stage built-in model catalog resources into the installable CLI package."""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "config" / "recipes" / "built-in"
DESTINATION = REPO_ROOT / "src" / "vllm-sr" / "cli" / "model_assets"
CLI_ROOT = REPO_ROOT / "src" / "vllm-sr"
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.model_bundle import MODEL_BUNDLE_FILES, model_bundle_digest  # noqa: E402

_VERSION_DIRECTORY = re.compile(r"^(?:latest|v\d+\.\d+)$")


def _resource_paths(version_dir: Path) -> tuple[Path, ...]:
    catalog_path = version_dir / "catalog.yaml"
    if not catalog_path.is_file():
        raise ValueError(
            f"built-in catalog is missing {catalog_path.relative_to(REPO_ROOT)}"
        )
    document = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    assets = document.get("assets") if isinstance(document, dict) else None
    if not isinstance(assets, list) or not assets:
        raise ValueError(
            f"built-in catalog has no assets: {catalog_path.relative_to(REPO_ROOT)}"
        )
    bundles: list[str] = []
    for asset in assets:
        bundle = asset.get("bundle") if isinstance(asset, dict) else None
        if (
            not isinstance(bundle, str)
            or Path(bundle).name != bundle
            or not re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", bundle)
        ):
            raise ValueError(
                f"built-in catalog asset path is invalid: {catalog_path.relative_to(REPO_ROOT)}"
            )
        if bundle in bundles:
            raise ValueError(f"built-in catalog asset is duplicated: {bundle}")
        bundles.append(bundle)
        bundle_path = version_dir / bundle
        actual_digest = model_bundle_digest(bundle_path)
        expected_digest = asset.get("sha256")
        if actual_digest != expected_digest:
            raise ValueError(f"built-in catalog asset digest drifted: {bundle}")

    declared = {"catalog.yaml", *bundles}
    actual = {path.name for path in version_dir.iterdir()}
    if actual != declared:
        raise ValueError(
            "built-in catalog version contents differ from the declared assets; "
            f"missing={sorted(declared - actual)}, extra={sorted(actual - declared)}"
        )
    return (
        Path("catalog.yaml"),
        *(Path(bundle) / name for bundle in bundles for name in MODEL_BUNDLE_FILES),
    )


def _expected_files() -> dict[Path, Path]:
    expected: dict[Path, Path] = {}
    for version_dir in sorted(path for path in SOURCE.iterdir() if path.is_dir()):
        if not _VERSION_DIRECTORY.fullmatch(version_dir.name):
            raise ValueError(
                f"built-in catalog version directory is invalid: {version_dir.name}"
            )
        for relative in _resource_paths(version_dir):
            source = version_dir / relative
            if not source.is_file():
                raise ValueError(
                    f"built-in catalog is missing {source.relative_to(REPO_ROOT)}"
                )
            expected[version_dir.relative_to(SOURCE) / relative] = source
    return expected


def _actual_resource_files() -> set[Path]:
    return {
        path.relative_to(DESTINATION)
        for path in DESTINATION.rglob("*")
        if path.is_file()
        and path.name != "__init__.py"
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    }


def check() -> int:
    expected = _expected_files()
    errors: list[str] = []
    for relative, source in expected.items():
        destination = DESTINATION / relative
        if not destination.is_file():
            errors.append(f"missing generated model asset: {relative}")
        elif not filecmp.cmp(source, destination, shallow=False):
            errors.append(f"stale generated model asset: {relative}")
    extra = sorted(_actual_resource_files() - set(expected))
    errors.extend(f"unexpected generated model asset: {relative}" for relative in extra)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


def sync() -> int:
    expected = _expected_files()
    for relative, source in expected.items():
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for relative in _actual_resource_files() - set(expected):
        (DESTINATION / relative).unlink()
    for directory in sorted(DESTINATION.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return check()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="reject package-resource drift"
    )
    args = parser.parse_args()
    return check() if args.check else sync()


if __name__ == "__main__":
    raise SystemExit(main())
