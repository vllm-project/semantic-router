#!/usr/bin/env python3
"""Publish a nightly date once, so a tag users pulled never moves.

Before a Nightly Qualification run publishes nightly-YYYYMMDD, compare each
image tag already in the registry with this run's sealed build receipts. A tag
holding this run's own images is a publication the run may finish (a re-run of
its failed jobs). Any other existing image, or a chart without this run's
images, means an earlier qualification owns the date, and the run publishes
nothing; its commit is published under a later date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ABSENT, OURS, OTHER = "absent", "ours", "other"
MISSING = re.compile(r"manifest unknown|name unknown|not found", re.IGNORECASE)


def registry_manifest(reference: str) -> bytes | None:
    """A tag's raw manifest, or None when the registry has no such tag."""
    result = subprocess.run(
        ["skopeo", "inspect", "--retry-times", "3", "--raw", f"docker://{reference}"],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout
    error = result.stderr.decode(errors="replace").strip()
    if MISSING.search(error):
        return None
    raise RuntimeError(f"cannot read {reference}: {error}")


def published_platforms(raw: bytes) -> dict[str, str]:
    """Platform to image manifest digest; a bare manifest has no platform key."""
    document = json.loads(raw)
    if "manifests" not in document:
        return {"": "sha256:" + hashlib.sha256(raw).hexdigest()}
    return {
        f"{entry['platform']['os']}/{entry['platform']['architecture']}": entry[
            "digest"
        ]
        for entry in document["manifests"]
    }


def ownership(raw: bytes | None, receipt: dict) -> str:
    if raw is None:
        return ABSENT
    built = {image["platform"]: image["manifest"] for image in receipt["images"]}
    published = published_platforms(raw)
    if "" in published:
        return OURS if list(built.values()) == [published[""]] else OTHER
    return OURS if published == built else OTHER


def decide(states: dict[str, str], chart_published: bool) -> tuple[bool, str]:
    """Publish only a free date or this run's own partial publication."""
    other = sorted(image for image, state in states.items() if state == OTHER)
    if other:
        return False, "holds another qualification's " + ", ".join(other)
    if chart_published and OURS not in states.values():
        return False, "has another qualification's chart"
    return True, "is this run's own" if OURS in states.values() else "is free"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--images", required=True, help="JSON list the run publishes")
    parser.add_argument(
        "--receipts",
        type=Path,
        required=True,
        help="Directory of this run's downloaded ci-build-image-* artifacts",
    )
    parser.add_argument("--registry", required=True, help="For example ghcr.io/owner")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}", args.date):
        parser.error("--date must be YYYYMMDD")
    states = {}
    for image in json.loads(args.images):
        path = args.receipts / f"ci-build-image-{image}" / "manifest.json"
        receipt = json.loads(path.read_text())
        if receipt["date"] != args.date:
            raise ValueError(
                f"{image} was built for {receipt['date']}, not {args.date}"
            )
        tag = f"{args.registry}/semantic-router/{image}:nightly-{args.date}"
        states[image] = ownership(registry_manifest(tag), receipt)
    chart = f"{args.registry}/charts/semantic-router:0.0.0-nightly.{args.date}"
    chart_published = registry_manifest(chart) is not None
    publish, reason = decide(states, chart_published)
    print(json.dumps({"images": states, "chart_published": chart_published}, indent=2))
    if publish:
        print(f"nightly-{args.date} {reason}; publishing.")
    else:
        print(
            f"::notice::nightly-{args.date} {reason}, so this run publishes nothing; "
            "a published date never moves, and this commit is published under a later date."
        )
    if args.github_output:
        with args.github_output.open("a") as output:
            output.write(f"publish={str(publish).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
