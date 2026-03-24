"""Flatten meta-routing feedback records into JSONL features for calibration jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meta_routing_feedback_support import (
    flatten_feedback_records,
    load_feedback_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract flat feature rows from meta-routing FeedbackRecord exports."
    )
    parser.add_argument("input", type=Path, help="JSON array or NDJSON input file")
    parser.add_argument("output", type=Path, help="JSONL output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = flatten_feedback_records(load_feedback_records(args.input))
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
