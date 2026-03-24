"""Summarize persisted meta-routing feedback records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meta_routing_feedback_support import (
    load_feedback_records,
    summarize_feedback_records,
)
from meta_routing_policy_support import (
    load_policy_artifact,
    summarize_policy_artifact,
    validate_policy_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize FeedbackRecord exports or recorder dumps for meta-routing."
    )
    parser.add_argument("input", type=Path, help="JSON array or NDJSON input file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--policy-artifact",
        type=Path,
        help="Optional learned-policy artifact to validate and include in the report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_feedback_records(args.input)
    summary = summarize_feedback_records(records)
    if args.policy_artifact is not None:
        artifact = load_policy_artifact(args.policy_artifact)
        summary["policy_artifact"] = summarize_policy_artifact(artifact)
        summary["policy_artifact"]["valid"] = not validate_policy_artifact(artifact)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)

    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
