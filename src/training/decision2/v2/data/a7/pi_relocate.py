"""Relocate a protected-inventory manifest to files already present on a node.

Protected inventories are built on one node; a scan on the other node needs the
same bytes at local paths. For every entry of the source manifest this finds
`<role>.jsonl` in the given directories (first match wins), requires its SHA-256
to equal the entry's, and writes a manifest with identical roles and hashes and
only the paths changed. A missing or different file is an error, so the
relocated manifest pins exactly the protected set of the source manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def relocate(
    entries: Sequence[Mapping[str, Any]], directories: Sequence[Path]
) -> list[dict[str, str]]:
    out = []
    for entry in entries:
        role, digest = entry["role"], entry["sha256"].lower()
        for directory in directories:
            candidate = directory / f"{role}.jsonl"
            if candidate.is_file() and sha256_file(candidate) == digest:
                out.append(
                    {"path": str(candidate.resolve()), "role": role, "sha256": digest}
                )
                break
        else:
            raise FileNotFoundError(f"{role}: no local file with SHA-256 {digest}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    raw = args.manifest.read_bytes()
    entries = relocate(json.loads(raw), args.search_dir)
    data = (json.dumps(entries, indent=1, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    print(
        json.dumps(
            {
                "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "relocated_manifest_sha256": hashlib.sha256(data).hexdigest(),
                "roles": len(entries),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
