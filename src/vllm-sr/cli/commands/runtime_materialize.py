"""Realize a raw config into one explicit runtime-owned file."""

from __future__ import annotations

import argparse
from pathlib import Path

from cli.commands.runtime_support import realize_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply runtime transforms and atomically materialize a config."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--algorithm")
    parser.add_argument(
        "--managed-listener",
        action="store_true",
        help="Validate the split stack management listener before publishing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = realize_runtime_config(
        args.source,
        args.target,
        algorithm=(args.algorithm or "").strip() or None,
        managed_listener=args.managed_listener,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
