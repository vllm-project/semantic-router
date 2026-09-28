"""Merge option-key teacher files into one hash-pinned teacher file.

    python3 -m v2.06b.teacher_merge --input PATH=SHA256 [--input ...] --output merged.jsonl --report report.json \
        [--restrict-to MIXTURE=SHA256 ...] [--repeats priority]

Every input line is `{id, input_sha256, teacher_probs}` (the canonical own-Lux A0 file
and the research & data RP-v2 waves). Inputs must match their frozen hashes, ids may
not repeat across inputs, and each distribution must be finite, non-negative and sum
to one. Output lines are canonical JSON in input order; the report holds counts only.

`--restrict-to` keeps only the entries for rows of the given materialized mixtures (by
`teacher_source_id` or `id`). `--repeats priority` allows an id in several inputs: the
inputs are given in priority order and the first entry is kept; the report counts the
repeats whose entries are identical and those that differ, per pair of inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .common import canonical, file_sha256, write_json


def merge(
    inputs: list[tuple[Path, str]],
    keep: set[str] | None = None,
    priority: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, tuple[int, dict[str, Any]]] = {}
    repeats: Counter[str] = Counter()
    parts = []
    for index, (path, expected) in enumerate(inputs):
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"{path}: teacher file differs from its frozen hash")
        count = kept = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if set(entry) != {"id", "input_sha256", "teacher_probs"}:
                raise ValueError(f"{path}: unexpected teacher fields {sorted(entry)}")
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
            count += 1
            if keep is not None and entry["id"] not in keep:
                continue
            if entry["id"] in seen:
                if not priority:
                    raise ValueError(f"{entry['id']}: repeated across teacher files")
                first, previous = seen[entry["id"]]
                same = "identical" if previous == entry else "different"
                repeats[f"{first}>{index}:{same}"] += 1
                continue
            seen[entry["id"]] = (index, entry)
            rows.append(entry)
            kept += 1
        parts.append({"sha256": actual, "rows": count, "kept": kept})
    report: dict[str, Any] = {"inputs": parts, "rows": len(rows)}
    if priority:
        report["repeats"] = dict(sorted(repeats.items()))
    return rows, report


def mixture_ids(item: str) -> tuple[set[str], dict[str, Any]]:
    path, expected = item.rsplit("=", 1)
    if file_sha256(path) != expected:
        raise ValueError(f"{path}: mixture differs from its frozen hash")
    ids = set()
    rows = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ids.add(row.get("teacher_source_id", row["id"]))
        rows += 1
    return ids, {"sha256": expected, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", required=True, help="PATH=SHA256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--restrict-to", action="append", default=[], help="MIXTURE=SHA256"
    )
    parser.add_argument("--repeats", choices=("error", "priority"), default="error")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite a merged teacher file")
    inputs = [
        (Path(item.rsplit("=", 1)[0]), item.rsplit("=", 1)[1]) for item in args.input
    ]
    keep = None
    restricted = []
    for item in args.restrict_to:
        ids, meta = mixture_ids(item)
        keep = ids if keep is None else keep | ids
        restricted.append(meta)
    rows, report = merge(inputs, keep, args.repeats == "priority")
    if keep is not None:
        report["restricted_to"] = restricted
        report["restricted_ids"] = len(keep)
        report["restricted_ids_without_entry"] = len(keep - {r["id"] for r in rows})
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
