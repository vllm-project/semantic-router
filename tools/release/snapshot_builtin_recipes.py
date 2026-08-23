#!/usr/bin/env python3
"""Create or verify immutable built-in Recipe snapshots from ``latest``."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from recipe_bundle import RECIPE_BUNDLE_FILES, recipe_bundle_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILT_IN_ROOT = REPO_ROOT / "config" / "recipes" / "built-in"

_RELEASE_VERSION = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)" r"(?:\.\d+(?:[-+][0-9A-Za-z.-]+)?)?$"
)
_STABLE_RELEASE_TAG = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
_COMMIT_ID = re.compile(r"^[0-9A-Fa-f]{7,64}$")
_BUNDLE_NAME = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def snapshot_name_for_version(version: str) -> str:
    match = _RELEASE_VERSION.fullmatch(version.strip())
    if match is None:
        raise ValueError(
            "release version must use MAJOR.MINOR or semantic version syntax"
        )
    return f"v{match.group('major')}.{match.group('minor')}"


def latest_bundles(latest_dir: Path) -> list[str]:
    """Discover and validate the complete Recipe-only latest distribution."""

    if not latest_dir.is_dir():
        raise ValueError(f"built-in Recipe source is missing: {latest_dir}")
    bundles: list[str] = []
    for entry in sorted(latest_dir.iterdir(), key=lambda item: item.name):
        if (
            entry.is_symlink()
            or not entry.is_dir()
            or _BUNDLE_NAME.fullmatch(entry.name) is None
        ):
            raise ValueError(
                "built-in Recipe source may contain only named bundle directories: "
                f"{entry.name}"
            )
        recipe_bundle_digest(entry)
        bundles.append(entry.name)
    if not bundles:
        raise ValueError("built-in Recipe source contains no bundles")
    return bundles


def expected_release_files(latest_dir: Path, snapshot: str) -> dict[Path, bytes]:
    """Render the exact release tree expected from one latest directory."""

    files: dict[Path, bytes] = {}
    for bundle in latest_bundles(latest_dir):
        latest_prefix = f"config/recipes/built-in/latest/{bundle}".encode()
        release_prefix = f"config/recipes/built-in/{snapshot}/{bundle}".encode()
        for name in RECIPE_BUNDLE_FILES:
            content = (latest_dir / bundle / name).read_bytes()
            if name in {"metadata.yaml", "probes.yaml"}:
                if latest_prefix not in content:
                    raise ValueError(
                        f"latest bundle {name} does not bind its source path: {bundle}"
                    )
                content = content.replace(latest_prefix, release_prefix)
            files[Path(bundle) / name] = content
    return files


def _release_tree_errors(
    release_dir: Path,
    expected: dict[Path, bytes],
    *,
    source_label: str = "latest",
) -> list[str]:
    if not release_dir.is_dir():
        return [f"missing built-in Recipe snapshot: {release_dir}"]
    actual_paths = {
        path.relative_to(release_dir)
        for path in release_dir.rglob("*")
        if path.is_file()
    }
    errors = [
        f"missing Recipe snapshot file: {path}"
        for path in sorted(set(expected) - actual_paths)
    ]
    errors.extend(
        f"unexpected Recipe snapshot file: {path}"
        for path in sorted(actual_paths - set(expected))
    )
    for relative in sorted(set(expected) & actual_paths):
        if (release_dir / relative).read_bytes() != expected[relative]:
            errors.append(f"Recipe snapshot drifted from {source_label}: {relative}")
    return errors


def release_snapshot_errors(built_in_root: Path, snapshot: str) -> list[str]:
    expected = expected_release_files(built_in_root / "latest", snapshot)
    release_dir = built_in_root / snapshot
    if not release_dir.is_dir():
        return [
            "missing built-in Recipe snapshot: " f"config/recipes/built-in/{snapshot}"
        ]
    return _release_tree_errors(release_dir, expected)


def published_snapshots_from_git(
    repo_root: Path, built_in_root: Path, base_ref: str
) -> list[tuple[str, str, dict[Path, bytes]]]:
    """Read Recipe snapshots from stable tags reachable from ``base_ref``."""

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
                    "--check-published requires --base-ref and cannot use "
                    "--version/--check"
                )
            published = published_snapshots_from_git(
                REPO_ROOT, BUILT_IN_ROOT, args.base_ref
            )
            errors = published_snapshot_errors(BUILT_IN_ROOT, published)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
            print(
                "Published built-in Recipe snapshots are byte-for-byte immutable "
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
            print(f"Built-in Recipe snapshot {snapshot} matches latest")
            return 0
        destination = create_release_snapshot(BUILT_IN_ROOT, snapshot)
        print(f"Created {destination.relative_to(REPO_ROOT)}")
        return 0
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"built-in Recipe snapshot failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
