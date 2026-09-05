"""Command line entry point for provenance manifest validation.

    python -m provenance.cli validate src/training/model_eval/provenance/manifests/jailbreak

Exit status is 0 only when every manifest in the bundle is schema-valid,
publishable, and mutually consistent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .crossref import validate_bundle
from .manifest import ManifestError, load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="provenance", description="Validate Router Model provenance manifests"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser(
        "validate", help="Validate a manifest bundle directory"
    )
    bundle.add_argument("directory", type=Path, help="Directory holding the manifests")
    bundle.add_argument(
        "--json", action="store_true", help="Print the summary as JSON on success"
    )

    single = subparsers.add_parser("check", help="Validate one manifest file")
    single.add_argument("path", type=Path, help="Manifest file to validate")
    single.add_argument(
        "--kind",
        choices=("dataset", "run", "artifact", "evaluation"),
        help="Require the manifest to declare this kind",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            _report_bundle(validate_bundle(args.directory), as_json=args.json)
        else:
            manifest = load_manifest(args.path, expected_kind=args.kind)
            print(f"ok {manifest['kind']:<10} {manifest['id']}")
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _report_bundle(summary: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    for kind in ("datasets", "runs", "artifacts", "evaluations"):
        for identifier in summary[kind]:
            print(f"ok {kind[:-1]:<10} {identifier}")


if __name__ == "__main__":
    raise SystemExit(main())
