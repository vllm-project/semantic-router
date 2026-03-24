"""Validate and summarize a meta-routing learned-policy artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meta_routing_policy_support import (
    load_policy_artifact,
    summarize_policy_artifact,
    validate_policy_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a meta-routing policy artifact and print a summary."
    )
    parser.add_argument("input", type=Path, help="Artifact JSON path")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = load_policy_artifact(args.input)
    errors = validate_policy_artifact(artifact)
    summary = summarize_policy_artifact(artifact)
    summary["valid"] = not errors
    summary["errors"] = errors
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)

    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
