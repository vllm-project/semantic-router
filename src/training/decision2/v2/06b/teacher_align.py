"""Restrict option-key teacher files to the rows every teacher covers on one mixture.

    python3 -m v2.06b.teacher_align --mixture PATH=SHA256 \
        --teacher NAME=PATH=SHA256 [--teacher ...] --output-dir DIR --report REPORT.json

A mixture row is covered by a teacher when that teacher has an entry for the row's
`teacher_source_id` (or `id`) computed on exactly the row's input (`input_sha256`);
a matching input with different option keys is an error. `DIR/<NAME>.aligned.jsonl`
holds each teacher's entries for the rows that every teacher covers, in mixture
order, so arms that differ only in the teacher apply the teacher term on the same
rows. The report holds counts only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import canonical, file_sha256, write_json


def covered(
    rows: list[dict[str, Any]], entries: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Row id -> teacher entry for every row the teacher saw with exactly its input."""
    out = {}
    for row in rows:
        entry = entries.get(row.get("teacher_source_id", row["id"]))
        if entry is None or entry["input_sha256"] != row["input_sha256"]:
            continue
        if set(entry["teacher_probs"]) != {o["key"] for o in row["options"]}:
            raise ValueError(f"{row['id']}: teacher keys differ from option keys")
        out[row["id"]] = entry
    return out


def align(
    rows: list[dict[str, Any]], teachers: dict[str, dict[str, dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_teacher = {name: covered(rows, entries) for name, entries in teachers.items()}
    common = [
        row for row in rows if all(row["id"] in cov for cov in by_teacher.values())
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    for name, cov in by_teacher.items():
        seen: set[str] = set()
        out[name] = []
        for row in common:
            entry = cov[row["id"]]
            if entry["id"] not in seen:
                seen.add(entry["id"])
                out[name].append(entry)
    types = Counter(row["task_type"] for row in rows)
    report = {
        "mixture_rows": len(rows),
        "mixture_task_types": dict(sorted(types.items())),
        "covered_by_teacher": {
            name: {
                "rows": len(cov),
                "task_types": dict(
                    sorted(
                        Counter(r["task_type"] for r in rows if r["id"] in cov).items()
                    )
                ),
            }
            for name, cov in by_teacher.items()
        },
        "aligned_rows": len(common),
        "aligned_task_types": dict(
            sorted(Counter(row["task_type"] for row in common).items())
        ),
        "entries_written": {name: len(entries) for name, entries in out.items()},
    }
    return out, report


def main() -> None:
    from training.model.data import load_partition

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mixture", required=True, help="PATH=SHA256")
    parser.add_argument(
        "--teacher", action="append", required=True, help="NAME=PATH=SHA256"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    path, expected = args.mixture.rsplit("=", 1)
    if file_sha256(path) != expected:
        raise ValueError("Mixture differs from its frozen hash")
    rows = load_partition(Path(path), "train")
    teachers: dict[str, dict[str, dict[str, Any]]] = {}
    inputs = {}
    for item in args.teacher:
        name, rest = item.split("=", 1)
        tpath, tsha = rest.rsplit("=", 1)
        if name in teachers:
            raise ValueError(f"Teacher {name} given twice")
        if file_sha256(tpath) != tsha:
            raise ValueError(f"{name}: teacher file differs from its frozen hash")
        entries: dict[str, dict[str, Any]] = {}
        for line in Path(tpath).read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["id"] in entries:
                raise ValueError(f"{name}: repeated teacher id {entry['id']}")
            entries[entry["id"]] = entry
        teachers[name] = entries
        inputs[name] = {"sha256": tsha, "entries": len(entries)}
    out, report = align(rows, teachers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, entries in out.items():
        target = args.output_dir / f"{name}.aligned.jsonl"
        if target.exists():
            raise FileExistsError(target)
        pending = target.with_name(target.name + ".pending")
        with pending.open("x", encoding="utf-8") as stream:
            for entry in entries:
                stream.write(canonical(entry) + "\n")
        pending.replace(target)
        outputs[name] = {"path": target.name, "sha256": file_sha256(target)}
    report.update({"mixture_sha256": expected, "teachers": inputs, "outputs": outputs})
    write_json(args.report, report, exclusive=True)
    print(json.dumps({k: report[k] for k in ("aligned_rows", "outputs")}))


if __name__ == "__main__":
    main()
