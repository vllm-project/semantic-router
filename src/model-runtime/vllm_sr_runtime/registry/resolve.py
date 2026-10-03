"""Resolve a model argument to a local package at a pinned revision.

A local directory is used as is. A Hub repository is always resolved to a
40-hex commit: ``--revision`` when given, else the built-in table's pinned
revision; any other repository without ``--revision`` is refused. Only the
files a built-in entry pins, or else the files the package manifest lists,
are downloaded; a package with neither gets its pointer files only, and its
family fetches what it loads (``ModelFamily.fetch``). Tokens come from the
environment or the Hugging Face token file, never from arguments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..errors import PackageError
from ..plugins.base import PackageRef
from . import builtin

REVISION = re.compile(r"[0-9a-f]{40}\Z")
SHORT_REVISION = re.compile(r"[0-9a-f]{7,39}\Z")
REPO_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+\Z")
POINTER_FILES = ("config.json", "MODEL_MANIFEST.json")


def pinned_revision(repo_id: str, revision: str | None) -> str:
    """The 40-hex revision to fetch; a short revision must match the built-in pin."""
    known = builtin.lookup(repo_id)
    if revision is None:
        if known is None:
            raise PackageError(
                f"{repo_id} is not a built-in model; pass --revision with its 40-hex commit"
            )
        return known.revision
    revision = revision.strip().lower()
    if REVISION.fullmatch(revision):
        return revision
    if (
        SHORT_REVISION.fullmatch(revision)
        and known
        and known.revision.startswith(revision)
    ):
        return known.revision
    raise PackageError(f"--revision must be a 40-hex commit, got {revision!r}")


def resolve(
    model: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> PackageRef:
    path = Path(model).expanduser()
    if path.exists():
        if revision is not None:
            raise PackageError("--revision applies only to Hub repositories")
        if path.is_symlink() or not path.is_dir():
            raise PackageError(f"model path must be a package directory: {model}")
        return PackageRef(root=path.resolve())
    known = builtin.lookup(model)
    repo_id = known.repo_id if known else model
    if not REPO_ID.fullmatch(repo_id):
        raise PackageError(
            f"{model!r} is neither a package directory nor a Hub repository ID"
        )
    commit = pinned_revision(repo_id, revision)
    root = download(repo_id, commit, cache_dir=cache_dir, offline=offline)
    return PackageRef(root=root, repo_id=repo_id, revision=commit)


def download(
    repo_id: str,
    revision: str,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> Path:
    """Fetch a built-in entry's pinned files, else the pointer and the files its manifest lists."""
    from huggingface_hub import snapshot_download

    common = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_files_only": offline,
    }
    known = builtin.lookup(repo_id)
    if known is not None and known.revision == revision and known.files:
        root = Path(snapshot_download(allow_patterns=sorted(known.files), **common))
        if root.name != revision:
            raise PackageError(
                f"{repo_id}: the Hub resolved {root.name}, not {revision}"
            )
        return root
    root = Path(snapshot_download(allow_patterns=list(POINTER_FILES), **common))
    if root.name != revision:
        raise PackageError(f"{repo_id}: the Hub resolved {root.name}, not {revision}")
    manifest_path = root / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        return root
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files_sha256") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise PackageError(f"{repo_id}@{revision}: MODEL_MANIFEST.json lists no files")
    root = Path(snapshot_download(allow_patterns=sorted(files), **common))
    return root


def fetch(
    ref: PackageRef,
    patterns: list[str],
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> PackageRef:
    """Download more files of a resolved Hub package (a family's inventory); local packages are complete."""
    if ref.repo_id is None or ref.revision is None or not patterns:
        return ref
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=ref.repo_id,
            revision=ref.revision,
            allow_patterns=sorted(set(patterns)),
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=offline,
        )
    )
    if root.name != ref.revision:
        raise PackageError(
            f"{ref.repo_id}: the Hub resolved {root.name}, not {ref.revision}"
        )
    return PackageRef(root=root, repo_id=ref.repo_id, revision=ref.revision)


def download_base(
    repo_id: str,
    revision: str,
    files: list[str],
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> Path:
    """Fetch the pinned base files an adapter package names (verified by the family)."""
    from huggingface_hub import snapshot_download

    if not REVISION.fullmatch(revision):
        raise PackageError(f"base revision of {repo_id} is not a 40-hex commit")
    root = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=sorted(files),
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=offline,
        )
    )
    if root.name != revision:
        raise PackageError(f"{repo_id}: the Hub resolved {root.name}, not {revision}")
    return root
