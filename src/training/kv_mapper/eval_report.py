#!/usr/bin/env python3
"""Turn per-item arm scores into a paired-CI JSON report. No model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Direct execution resolves repository imports after adding the repository root.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.training.kv_mapper.eval import build_report, read_items, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=Path,
        required=True,
        help="JSON with metric, optional reference, and arms: {name: [{id, score}, ...]}",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = read_items(args.items)
    metric = payload["metric"]
    arms = payload["arms"]
    reference = payload.get("reference", "cold")
    report = build_report(
        metric, arms, reference=reference, n_boot=args.n_boot, seed=args.seed
    )
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
