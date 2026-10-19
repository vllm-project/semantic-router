"""Fetch pinned public SDK archives; never accept unchecked local inputs."""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


def read_manifest(path):
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain archive entries")
    names = set()
    for entry in entries:
        name = entry["filename"]
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", name
        ):
            raise ValueError("invalid archive filename")
        if name in names:
            raise ValueError("duplicate archive filename")
        names.add(name)
        url = urlsplit(entry["url"])
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("archive URLs must use HTTPS without credentials")
        if not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
            raise ValueError("invalid SHA-256")
    return entries


def verify(path, expected):
    if not path.exists() and not path.is_symlink():
        raise ValueError(f"archive not found: {path.name}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"archive must be a regular file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"archive checksum mismatch: {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("toolchains.json")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    entries = read_manifest(args.manifest)
    directory = args.output if args.output is not None else args.verify_only
    if args.output is not None:
        directory.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = directory / entry["filename"]
        if args.output is not None and not path.exists() and not path.is_symlink():
            descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=directory)
            os.close(descriptor)
            temporary = Path(temporary)
            try:
                subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--proto",
                        "=https",
                        "--proto-redir",
                        "=https",
                        "--tlsv1.2",
                        "--retry",
                        "3",
                        "--connect-timeout",
                        "20",
                        "--max-time",
                        "1800",
                        entry["url"],
                        "--output",
                        str(temporary),
                    ],
                    check=True,
                )
                verify(temporary, entry["sha256"])
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        verify(path, entry["sha256"])
        print(f"{entry['sha256']}  {path.name}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
