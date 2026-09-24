"""Verify release data before a Decision family loader imports model code."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection
from pathlib import Path

_QWEN_SHARD = re.compile(r"backbone/model-(\d{5})-of-(\d{5})\.safetensors")


class ReleaseArtifactError(ValueError):
    """A local artifact differs from the caller's pinned release manifest."""


def verify_release_manifest(
    root: Path,
    *,
    manifest_name: str,
    expected_sha256: str,
    required_files: tuple[str, ...],
    qwen_weight_layout: bool = False,
) -> None:
    """Bind inference files to a trusted manifest digest from the caller.

    The caller owns the model/revision choice and obtains this digest from that
    pinned release.  Only files consumed by the owned runtime are read; bundled
    model-repository Python is neither trusted nor executed.
    """

    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ReleaseArtifactError("expected release manifest digest is invalid")
    try:
        manifest_bytes = (root / manifest_name).read_bytes()
    except OSError as error:
        raise ReleaseArtifactError(
            "release manifest is missing or unreadable"
        ) from error
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_sha256:
        raise ReleaseArtifactError("release manifest digest mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseArtifactError("release manifest is invalid") from error
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise ReleaseArtifactError("release manifest has no file inventory")

    if qwen_weight_layout:
        required_files += select_qwen_weight_files(files)

    for relative in required_files:
        entry = files.get(relative)
        if not isinstance(entry, dict):
            raise ReleaseArtifactError(f"release manifest is missing {relative}")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ReleaseArtifactError(f"release manifest entry is invalid: {relative}")
        path = root / relative
        try:
            if not path.is_file() or path.stat().st_size != size:
                raise ReleaseArtifactError(f"release file size mismatch: {relative}")
            actual = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    actual.update(block)
        except OSError as error:
            raise ReleaseArtifactError(
                f"release file is unreadable: {relative}"
            ) from error
        if actual.hexdigest() != digest:
            raise ReleaseArtifactError(f"release file digest mismatch: {relative}")


def select_qwen_weight_files(file_names: Collection[str]) -> tuple[str, ...]:
    """Select a complete, unambiguous Qwen weight layout from one manifest."""

    single = "backbone/model.safetensors"
    index = "backbone/model.safetensors.index.json"
    shards = {
        name: _QWEN_SHARD.fullmatch(name)
        for name in file_names
        if name.startswith("backbone/model-") and name.endswith(".safetensors")
    }
    if single in file_names:
        if index in file_names or shards:
            raise ReleaseArtifactError("Qwen weight layout is ambiguous")
        return (single,)
    if (
        index not in file_names
        or not shards
        or any(match is None for match in shards.values())
    ):
        raise ReleaseArtifactError("Qwen sharded weight layout is invalid")
    totals = {int(match.group(2)) for match in shards.values() if match}
    numbers = {int(match.group(1)) for match in shards.values() if match}
    if len(totals) != 1 or numbers != set(range(1, next(iter(totals)) + 1)):
        raise ReleaseArtifactError("Qwen weight shards are incomplete")
    return (index, *sorted(shards))
