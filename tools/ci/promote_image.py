#!/usr/bin/env python3
"""Promote a built digest without moving a mutable tag to an older source."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable

REVISION = "org.opencontainers.image.revision"
CHANNEL = "ai.vllm.semantic-router.channel"


def should_promote(
    current: dict[str, str], sha: str, mode: str, ancestor: Callable[[str, str], bool]
) -> bool:
    previous = current.get(REVISION)
    if not previous:
        return True  # One-time migration of a tag without source annotations.
    if previous == sha:
        return not (current.get(CHANNEL) == "release" and mode == "main")
    if ancestor(previous, sha):
        return True
    if ancestor(sha, previous):
        return False
    raise ValueError(f"Mutable tag points to unrelated source commit {previous}")


def git_ancestor(older: str, newer: str) -> bool:
    # The promotion checkout includes full history; unknown source revisions fail closed.
    subprocess.run(
        ["git", "cat-file", "-e", f"{older}^{{commit}}"],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer], check=False
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("Unable to compare publication source commits")
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata")
    args = parser.parse_args()
    with open(args.metadata, encoding="utf-8") as handle:
        metadata = json.load(handle)
    for target in metadata["mutable_tags"]:
        result = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", "--raw", target],
            capture_output=True,
            text=True,
            check=False,
        )
        current = {}
        if result.returncode == 0:
            current = json.loads(result.stdout).get("annotations", {})
        elif not any(
            marker in result.stderr.lower()
            for marker in ("not found", "manifest unknown")
        ):
            raise RuntimeError(f"Cannot inspect {target}: {result.stderr}")
        if not should_promote(current, metadata["sha"], metadata["mode"], git_ancestor):
            print(f"Keeping newer qualified image at {target}")
            continue
        subprocess.run(
            [
                "docker",
                "buildx",
                "imagetools",
                "create",
                "--tag",
                target,
                "--annotation",
                f"index:{REVISION}={metadata['sha']}",
                "--annotation",
                f"index:{CHANNEL}={metadata['mode']}",
                f"{metadata['image']}@{metadata['digest']}",
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
