"""Fresh copies of a frozen Triton autotune cache for ~27B kernel-path runs (host side, stdlib).

Tree hash (the recipe of the eval-track and release records)::

    cd DIR && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum

that is, SHA-256 over the lines ``"<sha256>  ./<relative path>\\n"`` of every
regular file (symlinks and directories excluded), sorted by the path's bytes.

``copy`` makes a ``cp -a`` copy of the frozen directory, refuses it unless its
tree hash equals the expected one, and writes ``<dest>.copy.json`` with the
per-file manifest. ``finish`` rehashes the copy after a run and writes
``<dest>.post.json`` with the post-run tree hash and the added, changed and
removed files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "decision2-27b-triton-cache/1"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for folder, _, names in os.walk(root):
        for name in names:
            path = Path(folder) / name
            if not stat.S_ISREG(os.lstat(path).st_mode):
                continue
            relative = path.relative_to(root).as_posix()
            if "\\" in relative or "\n" in relative:
                raise ValueError(f"sha256sum would escape the name {relative!r}")
            hashes[relative] = sha_file(path)
    return hashes


def tree_digest(hashes: dict[str, str]) -> str:
    lines = sorted(
        (f"./{name}".encode(), f"{sha}  ./{name}\n".encode())
        for name, sha in hashes.items()
    )
    return hashlib.sha256(b"".join(line for _, line in lines)).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=1, sort_keys=True)
        stream.write("\n")


def copy(frozen: Path, dest: Path, expect: str) -> dict:
    if dest.exists() or dest.with_name(dest.name + ".copy.json").exists():
        raise FileExistsError(f"{dest} exists; every run takes a fresh copy")
    if not frozen.is_dir():
        raise FileNotFoundError(frozen)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", str(frozen), str(dest)], check=True)
    hashes = file_hashes(dest)
    digest = tree_digest(hashes)
    if digest != expect:
        raise SystemExit(
            f"cache copy {dest} has tree hash {digest}, expected {expect}; not launching"
        )
    receipt = {
        "schema": SCHEMA,
        "frozen": str(frozen),
        "copy": str(dest),
        "copied_utc": utc(),
        "expected_sha256": expect,
        "copy_sha256": digest,
        "files": len(hashes),
        "autotune_entries": sum(n.endswith(".autotune.json") for n in hashes),
        "files_sha256": hashes,
    }
    write_json(dest.with_name(dest.name + ".copy.json"), receipt)
    return receipt


def finish(dest: Path) -> dict:
    before = json.loads(
        dest.with_name(dest.name + ".copy.json").read_text(encoding="utf-8")
    )
    old, new = before["files_sha256"], file_hashes(dest)
    result = {
        "schema": SCHEMA,
        "copy": str(dest),
        "finished_utc": utc(),
        "pre_sha256": before["copy_sha256"],
        "post_sha256": tree_digest(new),
        "files_before": len(old),
        "files_after": len(new),
        "autotune_entries_after": sum(n.endswith(".autotune.json") for n in new),
        "added": sorted(set(new) - set(old)),
        "changed": sorted(n for n in set(new) & set(old) if new[n] != old[n]),
        "removed": sorted(set(old) - set(new)),
    }
    result["unchanged"] = result["pre_sha256"] == result["post_sha256"]
    write_json(dest.with_name(dest.name + ".post.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    digest_parser = sub.add_parser("digest")
    digest_parser.add_argument("dir", type=Path)
    copy_parser = sub.add_parser("copy")
    copy_parser.add_argument("--frozen", type=Path, required=True)
    copy_parser.add_argument("--dest", type=Path, required=True)
    copy_parser.add_argument("--expect", required=True)
    finish_parser = sub.add_parser("finish")
    finish_parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "digest":
        print(tree_digest(file_hashes(args.dir)))
    elif args.command == "copy":
        receipt = copy(args.frozen, args.dest, args.expect)
        print(json.dumps({k: receipt[k] for k in ("copy", "copy_sha256", "files")}))
    else:
        result = finish(args.dest)
        print(
            json.dumps(
                {
                    "copy": result["copy"],
                    "post_sha256": result["post_sha256"],
                    "added": len(result["added"]),
                    "changed": len(result["changed"]),
                    "removed": len(result["removed"]),
                }
            )
        )


if __name__ == "__main__":
    main()
