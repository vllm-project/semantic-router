"""Record only the aggregate result of the preregistered v4 dry-run gate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path

from training.data import build_score_curriculum_v4 as curriculum

BUILDER_SHA256 = "dd837928d213d27eb2fc82c9f5b5f8562a93d03733f1408241e8828918cf8062"
PREREG_SHA256 = "aa7be2da837c31edac86bf787acc1c3d2b308c580b0e9996ec8c5301a94c2c61"
SOURCE_COMMIT = "fd039fd24cd8fcc480c72bfe7cce7b3742e57edc"


def record(output: Path) -> dict[str, object]:
    if (
        hashlib.sha256(Path(curriculum.__file__).read_bytes()).hexdigest()
        != BUILDER_SHA256
    ):
        raise ValueError("Signed v4 builder SHA mismatch")
    try:
        curriculum.generate()
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError(
            "The frozen negative feasibility screen unexpectedly passed"
        )
    match = re.fullmatch(
        r"Weighted single-signal held-out classifier exceeds 40%: "
        r"best=(\d+)/(\d+), by_position=(\{.*\})",
        message,
    )
    if match is None:
        raise ValueError("The v4 dry run failed outside its preregistered gate")
    by_position = ast.literal_eval(match.group(3))
    if not isinstance(by_position, dict) or set(by_position) != set(range(5)):
        raise ValueError("Malformed aggregate position counts")
    best, total = int(match.group(1)), int(match.group(2))
    if (best, total) != (112, 243) or max(by_position.values()) != best:
        raise ValueError("The frozen v4 negative result changed")
    report = {
        "schema_version": "decision20-score-v4-negative-feasibility/1",
        "status": "HOLD_NO_CORPUS",
        "builder_sha256": BUILDER_SHA256,
        "source_commit": SOURCE_COMMIT,
        "prereg_note_sha256": PREREG_SHA256,
        "best_single_signal_correct": best,
        "weighted_rows": total,
        "by_position_correct": {
            str(key): value for key, value in sorted(by_position.items())
        },
        "gate_max_correct": 97,
        "corpus_emitted": False,
        "gpu_used": False,
    }
    output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if output.parent.stat().st_mode & 0o077:
        raise PermissionError("Private feasibility output directory is not private")
    descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = record(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "best_single_signal_correct": report["best_single_signal_correct"],
                "weighted_rows": report["weighted_rows"],
                "receipt_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
