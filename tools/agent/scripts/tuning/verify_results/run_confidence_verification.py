#!/usr/bin/env python3
"""Build a versioned offline confidence-calibration artifact.

The command consumes a manifest-backed set of recorded small/large-model
results.  It never calls the router and never mutates active configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from tuning.confidence_calibration import (  # noqa: E402
    ConfidenceCalibrationError,
    build_artifact,
    write_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a reproducible offline confidence-calibration artifact"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to a confidence-calibration/v1 JSON manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("confidence_calibration_artifact.json"),
        help="Output artifact path",
    )
    args = parser.parse_args(argv)

    try:
        artifact = build_artifact(args.manifest)
        write_artifact(artifact, args.output)
    except (ConfidenceCalibrationError, OSError, json.JSONDecodeError) as error:
        print(f"confidence calibration failed: {error}", file=sys.stderr)
        return 2

    selection = artifact["selection"]
    print(f"Artifact: {args.output}")
    print(f"Artifact ID: {artifact['artifact_id']}")
    print(f"Status: {artifact['status']}")
    if "threshold" in selection:
        print(f"Candidate threshold: {selection['threshold']:.4f}")
    else:
        print(f"Fallback threshold: {artifact['fallback']['effective_threshold']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
