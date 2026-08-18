"""Run one pinned mmBERT-32K training family."""

from __future__ import annotations

import argparse
import importlib
import shlex
import sys
from pathlib import Path

from .config import arguments_to_argv, load_config

_MODULES = {
    ("foundation", "prepare"): "foundation_data",
    ("foundation", "train"): "foundation",
    ("embedder", "train"): "embedder",
    ("reranker", "train"): "reranker",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train"), default="train")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Resolve and print the delegated command without importing ML packages.",
    )
    args, delegated = parser.parse_known_args(argv)
    args.delegated = delegated
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    family = config["family"]
    key = (family, args.stage)
    if key not in _MODULES:
        raise ValueError(f"stage {args.stage!r} is not supported for {family!r}")

    config_key = f"{args.stage}_arguments"
    configured = config.get(config_key)
    if not isinstance(configured, dict):
        raise ValueError(f"{args.config} must define an object named {config_key!r}")

    delegated_argv = arguments_to_argv(configured) + args.delegated
    module_name = f"{__package__}.{_MODULES[key]}"
    if args.print_command:
        print(shlex.join([sys.executable, "-m", module_name, *delegated_argv]))
        return 0

    module = importlib.import_module(module_name)
    original_argv = sys.argv
    try:
        sys.argv = [module_name, *delegated_argv]
        module.main()
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
