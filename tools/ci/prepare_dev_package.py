#!/usr/bin/env python3
"""Set a timestamped package version in a disposable main-publish checkout."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

VERSION_LINE = re.compile(r'^version = "(\d+\.\d+\.\d+)"$', re.MULTILINE)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("src/vllm-sr"))
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    version = prepare_version(args.project, args.source_date_epoch)
    with args.github_output.open("a", encoding="utf-8") as output:
        output.write(f"version={version}\n")
    print(f"Development package version: {version}")


if __name__ == "__main__":
    main()
