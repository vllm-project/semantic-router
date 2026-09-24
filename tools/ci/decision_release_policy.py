#!/usr/bin/env python3
"""Select the protected stable package path from the source being released.

A Decision-capable CLI needs its default image lock even when none of its
files changed since the preceding tag. Keep this test about the source tree,
not a diff or a mutable repository variable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
DECISION_FILES = (
    "src/vllm-sr/cli/commands/drun.py",
    "src/vllm-sr/cli/decision_runtime/image_lock.py",
    "src/vllm-sr/decision_runtime/entrypoint.py",
)


def requires_qualification(root: Path) -> bool:
    """Conservatively include partial Decision installations in the gate."""

    project = tomllib.loads((root / "src/vllm-sr/pyproject.toml").read_text())
    scripts = project.get("project", {}).get("scripts", {})
    return "vllm-sr-decision-runtime" in scripts or any(
        (root / name).is_file() for name in DECISION_FILES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--expect", choices=("true", "false"))
    args = parser.parse_args()
    required = "true" if requires_qualification(ROOT) else "false"
    if args.expect is not None and args.expect != required:
        parser.error(
            "Decision package qualification input differs from the release source"
        )
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"decision_required={required}\n")
    print(required)


if __name__ == "__main__":
    main()
