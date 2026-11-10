#!/usr/bin/env python3
"""Set a timestamped package version in a disposable main-publish checkout."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

VERSION_LINE = re.compile(r'^version = "(\d+\.\d+\.\d+)"$', re.MULTILINE)
REVISION_LINE = re.compile(r'^SOURCE_REVISION = ""$', re.MULTILINE)
COMMIT = re.compile(r"[0-9a-f]{40}")


def development_epoch(revision: str = "HEAD", cwd: Path | None = None) -> int:
    """Committer time, made strictly increasing along the first-parent history.

    Main's squash merges can share a second, and a skewed clock can step back;
    either would give a newer commit a version that doesn't sort after its
    parent's. Such a commit takes the second after its parent's instead.
    """
    shallow = subprocess.check_output(
        ["git", "rev-parse", "--is-shallow-repository"], text=True, cwd=cwd
    )
    if shallow.strip() != "false":
        raise ValueError("a development version needs the complete commit history")
    times = subprocess.check_output(
        ["git", "log", "--first-parent", "--reverse", "--format=%ct", revision],
        text=True,
        cwd=cwd,
    )
    epoch = 0
    for value in map(int, times.split()):
        epoch = max(value, epoch + 1)
    return epoch


def prepare_version(project: Path, source_date_epoch: int) -> str:
    """Keep release sources untouched outside the publisher's checkout.

    Source time makes a retry use the same version and prevents a slow earlier
    commit's build from sorting after a newer commit on the dev channel.
    """
    timestamp = datetime.fromtimestamp(source_date_epoch, timezone.utc)
    pyproject = project / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    matches = list(VERSION_LINE.finditer(content))
    if len(matches) != 1:
        raise ValueError("expected one stable major.minor.patch package version")
    version = f"{matches[0].group(1)}.dev{timestamp:%Y%m%d%H%M%S}"
    pyproject.write_text(
        VERSION_LINE.sub(f'version = "{version}"', content), encoding="utf-8"
    )
    return version


def record_source_revision(project: Path, revision: str) -> None:
    """Pin the package's default images to the ones main publishes for its commit."""
    if not COMMIT.fullmatch(revision):
        raise ValueError("expected a full lowercase commit SHA")
    path = project / "cli" / "source_revision.py"
    content = path.read_text(encoding="utf-8")
    if len(REVISION_LINE.findall(content)) != 1:
        raise ValueError("expected one empty SOURCE_REVISION assignment")
    path.write_text(
        REVISION_LINE.sub(f'SOURCE_REVISION = "{revision}"', content),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("src/vllm-sr"))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    if not COMMIT.fullmatch(args.source_revision):
        parser.error("--source-revision must be a full lowercase commit SHA")
    version = prepare_version(args.project, development_epoch(args.source_revision))
    record_source_revision(args.project, args.source_revision)
    with args.github_output.open("a", encoding="utf-8") as output:
        output.write(f"version={version}\n")
    print(f"Development package version: {version} ({args.source_revision} images)")


if __name__ == "__main__":
    main()
