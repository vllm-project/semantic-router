#!/usr/bin/env python3
"""Classify whether a PR should use the docs/website lightweight CI profile."""

from __future__ import annotations

import argparse
import os
import sys

LIGHT_CHANGE_KEYS = ("website", "docs", "agent_text")

HEAVY_CHANGE_KEYS = (
    "core",
    "dashboard",
    "helm",
    "e2e",
    "docker",
    "make",
    "ci",
    "agent_exec",
    "operator",
    "perf",
    "bindings",
)


def _as_bool(value: str | None) -> bool:
    return str(value or "").lower() == "true"


def is_docs_only(changes: dict[str, bool]) -> bool:
    """Return True when only lightweight docs/website paths changed."""
    has_light = any(changes.get(key) for key in LIGHT_CHANGE_KEYS)
    has_heavy = any(changes.get(key) for key in HEAVY_CHANGE_KEYS)
    return has_light and not has_heavy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (*LIGHT_CHANGE_KEYS, *HEAVY_CHANGE_KEYS):
        parser.add_argument(f"--{key.replace('_', '-')}", default="false")
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write docs_only=<true|false> to GITHUB_OUTPUT when set",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changes = {
        key: _as_bool(getattr(args, key)) for key in (*LIGHT_CHANGE_KEYS, *HEAVY_CHANGE_KEYS)
    }
    docs_only = is_docs_only(changes)
    value = "true" if docs_only else "false"
    print(f"docs_only={value}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if args.github_output and github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"docs_only={value}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
