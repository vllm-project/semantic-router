"""Merge option-key teacher files into one hash-pinned teacher file.

    python3 -m v2.06b.teacher_merge --input PATH=SHA256 [--input ...] --output merged.jsonl --report report.json

Every input line is `{id, input_sha256, teacher_probs}` (the canonical own-Lux A0 file
and the research & data RP-v2 waves). Inputs must match their frozen hashes, ids may
not repeat across inputs, and each distribution must be finite, non-negative and sum
to one. Output lines are canonical JSON in input order; the report holds counts only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .common import canonical, file_sha256, write_json


def merge(
    inputs: list[tuple[Path, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    parts = []
    for path, expected in inputs:
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"{path}: teacher file differs from its frozen hash")
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if set(entry) != {"id", "input_sha256", "teacher_probs"}:
                raise ValueError(f"{path}: unexpected teacher fields {sorted(entry)}")
            if entry["id"] in seen:
                raise ValueError(f"{entry['id']}: repeated across teacher files")
            probs = list(entry["teacher_probs"].values())
            if (
                len(probs) < 2
                or not all(
                    isinstance(p, (int, float)) and math.isfinite(p) and p >= 0
                    for p in probs
                )
                or abs(sum(probs) - 1) > 1e-4
            ):
                raise ValueError(f"{entry['id']}: invalid teacher distribution")
            seen.add(entry["id"])
            rows.append(entry)
            count += 1
        parts.append({"sha256": actual, "rows": count})
    return rows, {"inputs": parts, "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", required=True, help="PATH=SHA256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite a merged teacher file")
    inputs = [
        (Path(item.rsplit("=", 1)[0]), item.rsplit("=", 1)[1]) for item in args.input
    ]
    rows, report = merge(inputs)
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    pending.replace(args.output)
    report["output_sha256"] = file_sha256(args.output)
    write_json(args.report, report, exclusive=True)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
