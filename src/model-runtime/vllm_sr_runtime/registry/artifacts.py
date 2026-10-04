"""Format-agnostic artifact checks: hashes, inventories and safetensors headers.

Nothing here imports or executes package content; weights are never read,
only safetensors headers.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path, PurePosixPath
from typing import Any

from ..errors import PackageError

HUB_ADDED = (".gitattributes",)
HUB_REPOSITORY_PREFIXES = ("models--", "datasets--", "spaces--")
MAX_SAFETENSORS_HEADER = 256 << 20
HEADER_LENGTH_BYTES = 8
MIN_SAFETENSORS_HEADER = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def safe_relative(name: str) -> bool:
    relative = PurePosixPath(name)
    return (
        bool(name)
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == name
        and "\\" not in name
    )


def transient(relative: PurePosixPath) -> bool:
    """Bytecode and Hugging Face local-dir metadata never count as package content."""
    if relative.parts[:2] == (".cache", "huggingface"):
        return True
    return "__pycache__" in relative.parts and relative.suffix == ".pyc"


def inventory(root: Path, *, ignore_hub_added: bool = True) -> dict[str, str]:
    """SHA-256 of every regular file under ``root``; links and unsafe names are refused.

    A Hub snapshot is a tree of links into the blob store, so links are
    followed only when they resolve inside the Hugging Face cache that holds
    ``root``.
    """
    root = Path(root)
    if not root.is_dir():
        raise PackageError(f"package root is not a directory: {root}")
    store = _snapshot_store(root)
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        name = relative.as_posix()
        if path.is_symlink() and (store is None or not _inside(path.resolve(), store)):
            raise PackageError(f"package contains a link: {name}")
        if path.is_dir() or transient(relative):
            continue
        if ignore_hub_added and name in HUB_ADDED:
            continue
        if not path.is_file() or not safe_relative(name):
            raise PackageError(f"unsafe package entry: {name}")
        files[name] = sha256_file(path)
    return files


def _snapshot_store(root: Path) -> Path | None:
    """The Hugging Face cache directory when ``root`` is a Hub snapshot, else None.

    Snapshot files link into their repository's ``blobs/``, and Xet-backed
    caches link those blobs on into a store shared by every repository in the
    cache, so a link may resolve anywhere under the cache directory.
    """
    resolved = root.resolve()
    repository = resolved.parent.parent
    if (
        resolved.parent.name == "snapshots"
        and (repository / "blobs").is_dir()
        and repository.name.startswith(HUB_REPOSITORY_PREFIXES)
    ):
        return repository.parent
    return None


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def safetensors_header(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        raw = stream.read(HEADER_LENGTH_BYTES)
        if len(raw) != HEADER_LENGTH_BYTES:
            raise PackageError(f"truncated safetensors header: {path.name}")
        (size,) = struct.unpack("<Q", raw)
        if not MIN_SAFETENSORS_HEADER <= size <= MAX_SAFETENSORS_HEADER:
            raise PackageError(f"invalid safetensors header: {path.name}")
        data = stream.read(size)
        if len(data) != size:
            raise PackageError(f"truncated safetensors header: {path.name}")
    header = json.loads(data)
    if not isinstance(header, dict):
        raise PackageError(f"invalid safetensors header: {path.name}")
    return header


def safetensors_elements(path: Path) -> int:
    """Element count from the header only; weights are never read."""
    total = 0
    for name, meta in safetensors_header(path).items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape") if isinstance(meta, dict) else None
        if not isinstance(shape, list) or any(
            type(dim) is not int or dim < 0 for dim in shape
        ):
            raise PackageError(f"invalid tensor shape in {Path(path).name}: {name}")
        total += math.prod(shape)
    return total
