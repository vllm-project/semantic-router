from __future__ import annotations

import argparse
import sys
from pathlib import Path

from provenance.manifests import load_manifest, manifest_id
from provenance.validate import (
    ProvenanceError,
    load_bundle,
    validate_bundle,
    validate_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="provenance")
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("validate-bundle")
    bundle.add_argument("directory", type=Path)
    bundle.add_argument("--artifact-dir", type=Path)
    single = commands.add_parser("validate")
    single.add_argument("manifest", type=Path)
    identity = commands.add_parser("id")
    identity.add_argument("manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate-bundle":
            validate_bundle(load_bundle(args.directory), args.artifact_dir)
        elif args.command == "validate":
            validate_manifest(load_manifest(args.manifest))
        else:
            print(manifest_id(load_manifest(args.manifest)))
            return 0
    except (ProvenanceError, OSError, ValueError) as err:
        print(str(err), file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
