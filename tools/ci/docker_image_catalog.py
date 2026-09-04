#!/usr/bin/env python3
"""Resolve Docker image build inputs from the shared CI catalog."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "tools" / "ci" / "docker-image-catalog.tsv"
CATALOG_FIELD_COUNT = 4
PLATFORM_PREFIX = "linux/"


@dataclass(frozen=True)
class ImageDefinition:
    context: str
    dockerfile: str
    platforms: str


def platform_targets(platforms: str) -> tuple[str, ...]:
    """Return the non-empty Linux targets in a catalog platform field."""
    targets = tuple(platform.strip() for platform in platforms.split(","))
    if not targets or any(
        not target or not target.startswith(PLATFORM_PREFIX) for target in targets
    ):
        raise ValueError(f"invalid Linux platforms '{platforms}'")
    return targets


def load_image_catalog(path: Path = CATALOG_PATH) -> dict[str, ImageDefinition]:
    """Load and validate the tab-separated Docker image catalog."""
    catalog: dict[str, ImageDefinition] = {}
    with path.open(encoding="utf-8", newline="") as catalog_file:
        rows = csv.reader(catalog_file, delimiter="\t")
        for line_number, row in enumerate(rows, start=1):
            if not row or not any(field.strip() for field in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if len(row) != CATALOG_FIELD_COUNT or any(
                not field.strip() for field in row
            ):
                raise ValueError(
                    f"{path}: line {line_number} must contain four non-empty fields"
                )
            image, context, dockerfile, platforms = (field.strip() for field in row)
            normalized_platforms = ",".join(platform_targets(platforms))
            if image in catalog:
                raise ValueError(f"{path}: duplicate image '{image}'")
            catalog[image] = ImageDefinition(context, dockerfile, normalized_platforms)
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="image name from the CI catalog")
    args = parser.parse_args()
    try:
        catalog = load_image_catalog()
        definition = catalog[args.image]
    except KeyError:
        print(f"::error::Unknown image '{args.image}'", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    print(f"context={definition.context}")
    print(f"dockerfile={definition.dockerfile}")
    print(f"platforms={definition.platforms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
