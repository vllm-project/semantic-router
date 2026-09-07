"""Configuration loading and CLI overrides for compact multimodal training."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        rendered = os.path.expanduser(os.path.expandvars(value))
        if "${" in rendered:
            raise ValueError(
                f"Unresolved environment variable in configuration: {value}"
            )
        return rendered
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and fail closed on unresolved environment variables."""
    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {source}")
    if config.get("schema_version") != 1:
        raise ValueError("Only schema_version 1 is supported")
    return _expand(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int)
    parser.add_argument("--train-cache")
    parser.add_argument("--validation-cache")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--print-config", action="store_true")
    return parser


def apply_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Apply only explicit command-line overrides."""
    resolved = copy.deepcopy(config)
    if args.stage is not None:
        resolved["training"]["stage"] = args.stage
    if args.train_cache is not None:
        resolved["data"]["train_cache"] = str(Path(args.train_cache).expanduser())
    if args.validation_cache is not None:
        resolved["data"]["validation_cache"] = str(
            Path(args.validation_cache).expanduser()
        )
    if args.output_dir is not None:
        resolved["runtime"]["output_dir"] = str(Path(args.output_dir).expanduser())
    if args.resume is not None:
        resolved["runtime"]["resume"] = str(Path(args.resume).expanduser())
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        resolved["training"]["max_steps"] = args.max_steps
    return resolved


def print_config(config: dict[str, Any]) -> None:
    print(json.dumps(config, indent=2, sort_keys=True))
