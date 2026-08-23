#!/usr/bin/env python3
"""Validate routing assets offline and evaluate an immutable live router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from router_calibration_manifest import load_probe_manifest, report_safe_probe_manifest
from router_calibration_support import (
    evaluate_probes,
    run_validate,
    write_json,
)


def cmd_eval(args: argparse.Namespace) -> int:
    manifest, probes = load_probe_manifest(Path(args.probes))
    evaluation = evaluate_probes(
        args.router_url,
        probes,
        manifest,
        selected_probe_ids=getattr(args, "probe_ids", None),
    )
    report = {
        "manifest": report_safe_probe_manifest(manifest),
        "evaluation": evaluation,
    }
    if args.output:
        write_json(Path(args.output), report)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if evaluation.get("passed", False) else 1


def cmd_validate(args: argparse.Namespace) -> int:
    result = run_validate(
        Path(args.dsl) if args.dsl else None,
        Path(args.yaml) if args.yaml else None,
    )
    if args.output:
        write_json(Path(args.output), result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("valid", False) else 1


def add_eval_subparser(subparsers: argparse._SubParsersAction) -> None:
    eval_parser = subparsers.add_parser(
        "eval", help="Run eval probes against the live router"
    )
    eval_parser.add_argument(
        "--router-url",
        required=True,
        help="Router base URL, for example http://host:8080",
    )
    eval_parser.add_argument("--probes", required=True, help="YAML probe manifest path")
    eval_parser.add_argument(
        "--id",
        dest="probe_ids",
        action="append",
        help=(
            "Evaluate only this exact decision:variant probe ID. Repeat the flag "
            "to run an ordered subset while validating traces against the complete recipe."
        ),
    )
    eval_parser.add_argument("--output", help="Optional JSON output path")
    eval_parser.set_defaults(func=cmd_eval)


def add_validate_subparser(subparsers: argparse._SubParsersAction) -> None:
    validate = subparsers.add_parser(
        "validate", help="Run local sr-dsl validate for a DSL or YAML asset"
    )
    validate.add_argument("--dsl", help="DSL path to validate directly")
    validate.add_argument("--yaml", help="YAML path to decompile then validate")
    validate.add_argument("--output", help="Optional JSON output path")
    validate.set_defaults(func=cmd_validate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate routing assets and evaluate an immutable router."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_eval_subparser(subparsers)
    add_validate_subparser(subparsers)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
