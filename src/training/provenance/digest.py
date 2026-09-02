from __future__ import annotations

import hashlib
from pathlib import Path

DIGEST_PREFIX = "sha256:"
_CHUNK = 1 << 20


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            hasher.update(chunk)
    return DIGEST_PREFIX + hasher.hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def tree_digest(files: dict[str, str]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(files):
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(files[name].encode())
        hasher.update(b"\n")
    return DIGEST_PREFIX + hasher.hexdigest()
