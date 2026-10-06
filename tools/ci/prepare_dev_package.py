#!/usr/bin/env python3
"""Set a timestamped package version in a disposable main-publish checkout."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

VERSION_LINE = re.compile(r'^version = "(\d+\.\d+\.\d+)"$', re.MULTILINE)
RUNTIME_PIN = re.compile(r'"vllm-srun==([^"]+)"')


def _set_runtime_version(pyproject: Path, base: str, version: str) -> None:
    content = pyproject.read_text(encoding="utf-8")
    found = VERSION_LINE.findall(content)
    if found != [base]:
        raise ValueError(f"expected {pyproject} to have version {base}, found {found}")
    pyproject.write_text(
        VERSION_LINE.sub(f'version = "{version}"', content), encoding="utf-8"
    )


def prepare_version(
    project: Path, source_date_epoch: int, runtime: Path | None = None
) -> str:
    """Keep release sources untouched outside the publisher's checkout.

    Source time makes a retry use the same version and prevents a slow earlier
    commit's build from sorting after a newer commit on the dev channel. The
    model runtime project, when given, gets the same version, and so does the
    CLI's ``runtime`` extra that pins it.
    """
    timestamp = datetime.fromtimestamp(source_date_epoch, timezone.utc)
    pyproject = project / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    matches = list(VERSION_LINE.finditer(content))
    if len(matches) != 1:
        raise ValueError("expected one stable major.minor.patch package version")
    base = matches[0].group(1)
    version = f"{base}.dev{timestamp:%Y%m%d%H%M%S}"
    if runtime is not None:
        pins = RUNTIME_PIN.findall(content)
        if pins != [base]:
            raise ValueError(f"expected the runtime extra to pin vllm-srun=={base}")
        _set_runtime_version(runtime / "pyproject.toml", base, version)
        content = RUNTIME_PIN.sub(f'"vllm-srun=={version}"', content)
    pyproject.write_text(
        VERSION_LINE.sub(f'version = "{version}"', content), encoding="utf-8"
    )
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("src/vllm-sr"))
    parser.add_argument(
        "--runtime-project", type=Path, default=Path("src/model-runtime")
    )
    parser.add_argument("--source-date-epoch", type=int, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    version = prepare_version(
        args.project, args.source_date_epoch, runtime=args.runtime_project
    )
    with args.github_output.open("a", encoding="utf-8") as output:
        output.write(f"version={version}\n")
    print(f"Development package version: {version}")


if __name__ == "__main__":
    main()
