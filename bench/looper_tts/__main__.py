"""Offline validate and plan commands for the Looper benchmark contract."""

import argparse
import json
import sys
from pathlib import Path

from .config import validate_config
from .plan import build_plan
from .validation import ContractError, load_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    validate = subparsers.add_parser(
        "validate", help="validate a JSON experiment config"
    )
    validate.add_argument("--config", required=True, type=Path)
    plan = subparsers.add_parser("plan", help="write an offline experiment manifest")
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument(
        "--code-revision", required=True, help="revision of the evaluated code"
    )
    args = parser.parse_args(argv)
    try:
        config = validate_config(load_json(args.config))
        if args.action == "validate":
            print(json.dumps({"id": config["id"], "valid": True}))
            return 0
        command = [
            "python",
            "-m",
            __package__,
            *(sys.argv[1:] if argv is None else argv),
        ]
        manifest = build_plan(config, args.code_revision, command)
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / "manifest.json"
        if destination.exists():
            raise ContractError(f"refusing to overwrite {destination}")
        write_json(destination, manifest, exclusive=True)
        print(
            json.dumps({"manifest": str(destination), "cells": len(manifest["matrix"])})
        )
        return 0
    except (ContractError, OSError, ValueError) as error:
        print(f"looper-tts: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
