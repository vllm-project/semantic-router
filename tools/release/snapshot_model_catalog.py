#!/usr/bin/env python3
"""Create or verify an immutable built-in model snapshot from ``latest``."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILT_IN_ROOT = REPO_ROOT / "config" / "recipes" / "built-in"
CLI_ROOT = REPO_ROOT / "src" / "vllm-sr"
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.model_bundle import (  # noqa: E402
    MODEL_BUNDLE_FILES,
    model_bundle_digest,
    model_bundle_digest_from_files,
)

_RELEASE_VERSION = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)" r"(?:\.\d+(?:[-+][0-9A-Za-z.-]+)?)?$"
)
_STABLE_RELEASE_TAG = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
_COMMIT_ID = re.compile(r"^[0-9A-Fa-f]{7,64}$")


def snapshot_name_for_version(version: str) -> str:
    match = _RELEASE_VERSION.fullmatch(version.strip())
    if match is None:
        raise ValueError(
            "release version must use MAJOR.MINOR or semantic version syntax"
        )
    return f"v{match.group('major')}.{match.group('minor')}"


def _load_catalog(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"built-in catalog is missing: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"built-in catalog is invalid: {path}")
    return document


def _latest_bundles(latest_dir: Path) -> tuple[dict[str, Any], list[str]]:
    catalog = _load_catalog(latest_dir / "catalog.yaml")
    expected_identity = {
        "schema_version": "vllm-sr/model-catalog/v1",
        "catalog_version": "latest",
        "channel": "latest",
        "release": "unreleased",
    }
    for key, expected in expected_identity.items():
        if catalog.get(key) != expected:
            raise ValueError(
                f"latest catalog {key} must be {expected!r} before snapshotting"
            )

    assets = catalog.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("latest catalog must declare at least one bundle")
    bundles: list[str] = []
    for asset in assets:
        bundle = asset.get("bundle") if isinstance(asset, dict) else None
        if (
            not isinstance(bundle, str)
            or Path(bundle).name != bundle
            or not re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", bundle)
        ):
            raise ValueError("latest catalog contains an invalid bundle path")
        if bundle in bundles:
            raise ValueError(f"latest catalog duplicates bundle: {bundle}")
        bundles.append(bundle)
        actual_digest = model_bundle_digest(latest_dir / bundle)
        if asset.get("sha256") != actual_digest:
            raise ValueError(f"latest catalog bundle digest drifted: {bundle}")

    expected_entries = {"catalog.yaml", *bundles}
    actual_entries = {path.name for path in latest_dir.iterdir()}
    if actual_entries != expected_entries:
        raise ValueError(
            "latest catalog contents differ from declared bundles; "
            f"missing={sorted(expected_entries - actual_entries)}, "
            f"extra={sorted(actual_entries - expected_entries)}"
        )
    return catalog, bundles


def expected_release_files(latest_dir: Path, snapshot: str) -> dict[Path, bytes]:
    """Render the exact release tree expected from one latest directory."""

    catalog, bundles = _latest_bundles(latest_dir)
    files: dict[Path, bytes] = {}
    digests: dict[str, str] = {}
    for bundle in bundles:
        bundle_files = {
            name: (latest_dir / bundle / name).read_bytes()
            for name in MODEL_BUNDLE_FILES
        }
        latest_prefix = f"config/recipes/built-in/latest/{bundle}".encode()
        release_prefix = f"config/recipes/built-in/{snapshot}/{bundle}".encode()
        for binding_file in ("metadata.yaml", "probes.yaml"):
            if latest_prefix not in bundle_files[binding_file]:
                raise ValueError(
                    f"latest bundle {binding_file} does not bind its latest "
                    f"asset path: {bundle}"
                )
        bundle_files = {
            name: content.replace(latest_prefix, release_prefix)
            for name, content in bundle_files.items()
        }
        digests[bundle] = model_bundle_digest_from_files(bundle_files)
        for name, content in bundle_files.items():
            files[Path(bundle) / name] = content

    rendered = dict(catalog)
    rendered["channel"] = "release"
    rendered["catalog_version"] = snapshot
    rendered["release"] = snapshot
    asset_ids: dict[str, str] = {}
    for asset in rendered["assets"]:
        bundle = asset["bundle"]
        asset["sha256"] = digests[bundle]
        asset_ids[asset["id"]] = bundle
    for model in rendered.get("models", []):
        if not isinstance(model, dict) or model.get("asset") not in asset_ids:
            raise ValueError("latest catalog model references an unknown bundle")
        verification = model.get("verification")
        if not isinstance(verification, dict):
            raise ValueError("latest catalog model is missing verification")
        verification["asset_sha256"] = digests[asset_ids[model["asset"]]]
    files[Path("catalog.yaml")] = yaml.safe_dump(
        rendered,
        sort_keys=False,
        width=120,
    ).encode("utf-8")
    return files


def _release_tree_errors(
    release_dir: Path,
    expected: dict[Path, bytes],
    *,
    source_label: str = "latest",
) -> list[str]:
    if not release_dir.is_dir():
        return [f"missing release catalog snapshot: {release_dir}"]
    actual_paths = {
        path.relative_to(release_dir)
        for path in release_dir.rglob("*")
        if path.is_file()
    }
    errors = [
        f"missing release snapshot file: {path}"
        for path in sorted(set(expected) - actual_paths)
    ]
    errors.extend(
        f"unexpected release snapshot file: {path}"
        for path in sorted(actual_paths - set(expected))
    )
    for relative in sorted(set(expected) & actual_paths):
        if (release_dir / relative).read_bytes() != expected[relative]:
            errors.append(f"release snapshot drifted from {source_label}: {relative}")
    return errors


def release_snapshot_errors(built_in_root: Path, snapshot: str) -> list[str]:
    expected = expected_release_files(built_in_root / "latest", snapshot)
    release_dir = built_in_root / snapshot
    if not release_dir.is_dir():
        return [f"missing release catalog snapshot: config/recipes/built-in/{snapshot}"]
    return _release_tree_errors(release_dir, expected)


def published_snapshots_from_git(
    repo_root: Path, built_in_root: Path, base_ref: str
) -> list[tuple[str, str, dict[Path, bytes]]]:
    """Read release snapshots from stable tags reachable from ``base_ref``."""

    if _COMMIT_ID.fullmatch(base_ref) is None:
        raise ValueError("base ref must be a hexadecimal commit ID")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{base_ref}^{{commit}}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    tags = subprocess.run(
        ["git", "-C", str(repo_root), "tag", "--merged", base_ref, "--list"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    relative_root = built_in_root.relative_to(repo_root).as_posix()
    published: list[tuple[str, str, dict[Path, bytes]]] = []
    for tag in sorted(tags):
        match = _STABLE_RELEASE_TAG.fullmatch(tag)
        if match is None:
            continue
        snapshot = f"v{match.group('major')}.{match.group('minor')}"
        prefix = f"{relative_root}/{snapshot}/"
        output = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-tree",
                "-r",
                "-z",
                "--name-only",
                tag,
                "--",
                f"{relative_root}/{snapshot}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        tagged_files: dict[Path, bytes] = {}
        for raw_path in output.split(b"\0"):
            if not raw_path:
                continue
            source_path = raw_path.decode("utf-8")
            if not source_path.startswith(prefix):
                continue
            relative = Path(source_path.removeprefix(prefix))
            tagged_files[relative] = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "cat-file",
                    "blob",
                    f"{tag}:{source_path}",
                ],
                check=True,
                capture_output=True,
            ).stdout
        if tagged_files:
            published.append((tag, snapshot, tagged_files))
    return published


def published_snapshot_errors(
    built_in_root: Path,
    published: list[tuple[str, str, dict[Path, bytes]]],
) -> list[str]:
    """Reject byte or inventory changes to snapshots already present in a tag."""

    errors: list[str] = []
    for tag, snapshot, expected in published:
        for message in _release_tree_errors(
            built_in_root / snapshot,
            expected,
            source_label=f"published tag {tag}",
        ):
            errors.append(f"{snapshot}: {message}")
    return errors


def create_release_snapshot(built_in_root: Path, snapshot: str) -> Path:
    destination = built_in_root / snapshot
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing snapshot: {destination}")
    expected = expected_release_files(built_in_root / "latest", snapshot)
    staging = Path(tempfile.mkdtemp(prefix=f".{snapshot}.staging-", dir=built_in_root))
    try:
        for relative, content in expected.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        errors = _release_tree_errors(staging, expected)
        if errors:
            raise ValueError("; ".join(errors))
        if destination.exists():
            raise ValueError(f"refusing to overwrite existing snapshot: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="release version or tag")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing release snapshot instead of creating it",
    )
    parser.add_argument(
        "--check-published",
        action="store_true",
        help="reject changes to snapshots already bound to reachable release tags",
    )
    parser.add_argument(
        "--base-ref",
        help="base commit used to discover reachable release tags",
    )
    args = parser.parse_args()
    try:
        if args.check_published:
            if args.version is not None or args.check or args.base_ref is None:
                parser.error(
                    "--check-published requires --base-ref and cannot use --version/--check"
                )
            published = published_snapshots_from_git(
                REPO_ROOT, BUILT_IN_ROOT, args.base_ref
            )
            errors = published_snapshot_errors(BUILT_IN_ROOT, published)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
            print(
                "Published built-in model snapshots are byte-for-byte immutable "
                f"({len(published)} tagged snapshot binding(s))"
            )
            return 0
        if args.version is None or args.base_ref is not None:
            parser.error("--version is required unless --check-published is used")
        snapshot = snapshot_name_for_version(args.version)
        if args.check:
            errors = release_snapshot_errors(BUILT_IN_ROOT, snapshot)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
            print(f"Built-in model snapshot {snapshot} matches latest")
            return 0
        destination = create_release_snapshot(BUILT_IN_ROOT, snapshot)
        print(f"Created {destination.relative_to(REPO_ROOT)}")
        return 0
    except (
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(f"built-in model snapshot failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
