"""Frozen Triton autotune cache snapshot and per-run copies for M6 formal runs (stdlib only).

    python3 -m v2.06b.m6_cache tree DIR
    python3 -m v2.06b.m6_cache snapshot --source DIR --dest DIR --manifest OUT [--expect-tree SHA]
    python3 -m v2.06b.m6_cache seed --snapshot DIR --manifest MANIFEST --dest DIR
    python3 -m v2.06b.m6_cache record --manifest MANIFEST --cache DIR --output OUT

The tree hash is the eval track's convention: SHA-256 of the `sha256sum` listing
("<sha256>  ./<path>" per file, sorted by path), i.e. of
`find . -type f -print0 | sort -z | xargs -0 sha256sum`. `snapshot` copies the source,
makes the copy read-only and writes a manifest of every file. `seed` verifies the snapshot
against its manifest and copies it into a new writable per-run directory. `record` writes the
before (snapshot) and after (per-run cache) tree hashes and every added, removed or changed file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

SCHEMA = "dev2-06b-m6-triton-cache/1"


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def listing(root: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"not a regular file: {path}")
            rel = path.relative_to(root).as_posix()
            files[rel] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return files


def tree_sha256(files: dict[str, dict[str, Any]]) -> str:
    text = "".join(
        f"{files[rel]['sha256']}  ./{rel}\n"
        for rel in sorted(files, key=lambda r: ("./" + r).encode())
    )
    return hashlib.sha256(text.encode()).hexdigest()


def describe(files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "tree_sha256": tree_sha256(files),
        "files": len(files),
        "bytes": sum(v["bytes"] for v in files.values()),
        "autotune_entries": sorted(r for r in files if r.endswith(".autotune.json")),
    }


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def set_writable(root: Path, writable: bool) -> None:
    bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for directory, dirs, names in os.walk(root):
        for entry in [directory] + [os.path.join(directory, n) for n in names]:
            mode = os.stat(entry).st_mode
            os.chmod(entry, (mode | stat.S_IWUSR) if writable else (mode & ~bits))


def snapshot(
    source: Path, dest: Path, manifest: Path, expect: str | None
) -> dict[str, Any]:
    if dest.exists() or manifest.exists():
        raise FileExistsError(
            f"{dest} or {manifest} exists; a snapshot is written once"
        )
    before = listing(source)
    if expect and tree_sha256(before) != expect:
        raise ValueError(f"source tree {tree_sha256(before)} != expected {expect}")
    shutil.copytree(source, dest, symlinks=False)
    after = listing(dest)
    if after != before:
        raise ValueError("copy differs from the source")
    set_writable(dest, False)
    record = {
        "schema": SCHEMA,
        "kind": "snapshot",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(source),
        "snapshot": str(dest),
        "tree_rule": "sha256 of sorted `sha256sum` lines '<sha256>  ./<path>'",
        **describe(after),
        "manifest": after,
    }
    write_exclusive(manifest, record)
    return record


def verify_snapshot(snapshot_dir: Path, manifest: Path) -> dict[str, Any]:
    record = json.loads(manifest.read_text(encoding="utf-8"))
    files = listing(snapshot_dir)
    if files != record["manifest"] or tree_sha256(files) != record["tree_sha256"]:
        raise ValueError(f"{snapshot_dir} differs from {manifest}")
    return record


def seed(snapshot_dir: Path, manifest: Path, dest: Path) -> dict[str, Any]:
    if dest.exists():
        raise FileExistsError(f"{dest} exists; every run gets a fresh copy")
    record = verify_snapshot(snapshot_dir, manifest)
    shutil.copytree(snapshot_dir, dest, symlinks=False)
    set_writable(dest, True)
    if tree_sha256(listing(dest)) != record["tree_sha256"]:
        raise ValueError("per-run copy differs from the snapshot")
    return record


def diff(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "changed": sorted(r for r in set(before) & set(after) if before[r] != after[r]),
    }


def record(manifest: Path, cache: Path, output: Path) -> dict[str, Any]:
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    after = listing(cache)
    changes = diff(frozen["manifest"], after)
    out = {
        "schema": SCHEMA,
        "kind": "per-run",
        "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot": frozen["snapshot"],
        "snapshot_manifest": str(manifest),
        "before_tree_sha256": frozen["tree_sha256"],
        "before_files": frozen["files"],
        "cache": str(cache),
        "after_tree_sha256": tree_sha256(after),
        "after_files": len(after),
        "unchanged": not any(changes.values()),
        "autotune_entries_added": [
            r for r in changes["added"] if r.endswith(".autotune.json")
        ],
        **changes,
    }
    write_exclusive(output, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("tree")
    p.add_argument("dir", type=Path)
    p = commands.add_parser("snapshot")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--expect-tree")
    p = commands.add_parser("seed")
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p = commands.add_parser("record")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "tree":
        out = describe(listing(args.dir))
    elif args.command == "snapshot":
        full = snapshot(args.source, args.dest, args.manifest, args.expect_tree)
        out = {k: full[k] for k in ("snapshot", "tree_sha256", "files", "bytes")}
    elif args.command == "seed":
        full = seed(args.snapshot, args.manifest, args.dest)
        out = {"dest": str(args.dest), "tree_sha256": full["tree_sha256"]}
    else:
        full = record(args.manifest, args.cache, args.output)
        out = {
            k: full[k]
            for k in ("before_tree_sha256", "after_tree_sha256", "unchanged", "added")
        }
    json.dump(out, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
