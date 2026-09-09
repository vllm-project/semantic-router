#!/usr/bin/env python3
"""Resolve Docker build inputs from the existing image registry.

The shared resolver approach follows PR #3491; ownership and build definitions
live together here in domains.yaml, without a second image inventory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from domain_registry import image_records

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ImageDefinition:
    context: str
    dockerfile: str
    platforms: str


def image_definition(image: str, *, pr: bool = False) -> ImageDefinition:
    record = image_records().get(image)
    if record is None:
        raise ValueError(f"Unknown image {image!r}")
    definition = ImageDefinition(
        record["context"],
        record["dockerfile"],
        record["pr_platforms" if pr else "platforms"],
    )
    if not (REPO_ROOT / definition.context).is_dir():
        raise ValueError(f"Missing build context for {image!r}")
    if not (REPO_ROOT / definition.dockerfile).is_file():
        raise ValueError(f"Missing Dockerfile for {image!r}")
    platforms = set(definition.platforms.split(","))
    if not platforms or not platforms <= {"linux/amd64", "linux/arm64"}:
        raise ValueError(f"Invalid platforms for {image!r}")
    return definition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--pr", action="store_true")
    args = parser.parse_args()
    try:
        definition = image_definition(args.image, pr=args.pr)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"context={definition.context}")
    print(f"dockerfile={definition.dockerfile}")
    print(f"platforms={definition.platforms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
