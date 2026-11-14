#!/usr/bin/env python3
"""Move each image's latest tag forward along main, never back.

Every main publication pushes its images under the commit SHA and then moves
latest. Runs overlap, so an older commit's run can finish after a newer one's.
An image's latest moves to this run's commit unless it already holds the image
of a newer main commit, found by comparing manifests with those commits' SHA
tags. main.yml runs one move at a time, and a commit merged after this
checkout cannot have published yet.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterable

import provider_mocker_image as mocker
from image_artifacts import PUBLICATION_ALIASES
from nightly_tags import registry_manifest

MOVE, CURRENT, NEWER = "move", "current", "newer"
# A commit this far behind main leaves latest to the newer commits' runs.
MAX_NEWER_COMMITS = 30


def newer_commits(source: str, main: str) -> list[str] | None:
    """Main's first-parent commits after source, newest first; None off main."""
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source, main], check=False
    )
    if ancestor.returncode == 1:
        return None
    ancestor.check_returncode()
    return subprocess.check_output(
        ["git", "rev-list", "--first-parent", f"{source}..{main}"], text=True
    ).split()


def decide(
    current: bytes | None, own: bytes, newer: Iterable[tuple[str, bytes | None]]
) -> tuple[str, str]:
    """Compare latest with this commit's manifest, then the newer commits' ones."""
    if current == own:
        return CURRENT, ""
    if current is not None:
        for commit, manifest in newer:
            if manifest == current:
                return NEWER, commit
    return MOVE, ""


def published_names(images: list[str]) -> list[str]:
    """Every name main tags with latest. Fixtures are addressed by their inputs."""
    return [
        name
        for image in images
        if image != mocker.IMAGE
        for name in (image, *PUBLICATION_ALIASES.get(image, []))
    ]


def move(repository: str, source: str) -> None:
    subprocess.run(
        [
            "skopeo",
            "copy",
            "--retry-times",
            "3",
            "--all",
            "--preserve-digests",
            f"docker://{repository}:{source}",
            f"docker://{repository}:latest",
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="JSON list main published")
    parser.add_argument("--source", required=True, help="This run's commit")
    parser.add_argument("--main", default="origin/main")
    parser.add_argument("--registry", required=True, help="For example ghcr.io/owner")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source):
        parser.error("--source must be a full lowercase commit SHA")
    newer = newer_commits(args.source, args.main)
    if newer is None:
        print(f"::notice::{args.source} is not on main, so latest stays unchanged.")
        return 0
    if len(newer) > MAX_NEWER_COMMITS:
        print(
            f"::notice::main is {len(newer)} commits past {args.source}; "
            "their runs move latest."
        )
        return 0
    results = {}
    for name in published_names(json.loads(args.images)):
        repository = f"{args.registry}/semantic-router/{name}"
        own = registry_manifest(f"{repository}:{args.source}")
        if own is None:
            raise RuntimeError(f"{repository}:{args.source} is not published")
        decision, commit = decide(
            registry_manifest(f"{repository}:latest"),
            own,
            ((sha, registry_manifest(f"{repository}:{sha}")) for sha in newer),
        )
        if decision == MOVE:
            move(repository, args.source)
        results[name] = f"{decision} {commit}".strip()
    print(json.dumps(results, indent=2))
    held = sorted(name for name, result in results.items() if result.startswith(NEWER))
    if held:
        print(
            "::notice::A newer main commit already holds latest for "
            + ", ".join(held)
            + f"; {args.source} keeps only its SHA tags there."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
