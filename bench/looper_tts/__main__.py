"""Validate, plan, and execute fixed-budget Looper benchmark matrices."""

import argparse
import json
import sys
from pathlib import Path

from .config import validate_config
from .executor import execute_manifest
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
    execute = subparsers.add_parser(
        "execute", help="execute a saved matrix with per-call budget accounting"
    )
    execute.add_argument("--manifest", required=True, type=Path)
    execute.add_argument("--output", required=True, type=Path)
    execute.add_argument(
        "--endpoint",
        help="OpenAI-compatible base URL (required unless --fake is set)",
    )
    execute.add_argument("--api-key", default="", help="provider API key")
    execute.add_argument("--timeout", type=int, default=600)
    execute.add_argument("--retries", type=int, default=0)
    execute.add_argument("--max-output-tokens", type=int)
    execute.add_argument(
        "--fake",
        action="store_true",
        help="use the deterministic provider for an offline smoke run",
    )
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            config = validate_config(load_json(args.config))
            print(json.dumps({"id": config["id"], "valid": True}))
            return 0
        if args.action == "execute":
            records = execute_manifest(
                args.manifest,
                args.output,
                endpoint=args.endpoint,
                api_key=args.api_key,
                retries=args.retries,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                fake=args.fake,
            )
            print(
                json.dumps(
                    {
                        "records": str(args.output / "records.json"),
                        "calls": len(records["calls"]),
                        "results": len(records["results"]),
                    }
                )
            )
            return 0
        config = validate_config(load_json(args.config))
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
