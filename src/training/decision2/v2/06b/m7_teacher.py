"""Milestone 7(b) teacher files: an M6 own-Lux teacher minus the rows of dropped pools.

    python3 -m v2.06b.m7_teacher --teacher T.jsonl --teacher-sha256 SHA \
        --recipe R.ids.jsonl [--recipe ...] --drop-pool P [--drop-pool ...] \
        --mixture M.train.jsonl=SHA256=COVERED [--mixture ...] --expect-kept N \
        --output OUT.jsonl --report OUT.report.json

Standard library only (runs on a node's host python3). Every teacher line is kept or
removed by the pool of its id in the recipe id lists (`id`, `pool`); kept lines are written
byte-identical and in their original order. A mixture row counts as covered when the kept
teacher has an entry for its `teacher_source_id` (else `id`) with the same `input_sha256`,
as the trainer joins them. The run passes only if the kept count, every mixture's covered
count and hash match the expected values and uncovered mixture rows lie only in dropped
pools; otherwise it writes the report (pass false) but no teacher file and exits 1. An
unknown id, a repeated id or a hash mismatch stops it before any output. Outputs are
write-once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pools(paths: list[Path]) -> dict[str, str]:
    pool_of: dict[str, str] = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                known = pool_of.setdefault(entry["id"], entry["pool"])
                if known != entry["pool"]:
                    raise ValueError(f"{path}: id listed under two pools")
    return pool_of


def split_teacher(
    path: Path, pool_of: dict[str, str], drop: set[str]
) -> tuple[list[bytes], dict[str, str], Counter[str], Counter[str]]:
    """Kept raw lines, kept id -> input_sha256, kept and removed lines by pool."""
    kept_lines: list[bytes] = []
    kept: dict[str, str] = {}
    seen: set[str] = set()
    kept_by_pool: Counter[str] = Counter()
    removed_by_pool: Counter[str] = Counter()
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"teacher line {number} has no newline")
            entry = json.loads(line)
            key = entry["id"]
            if key in seen:
                raise ValueError(f"teacher line {number}: repeated id")
            seen.add(key)
            pool = pool_of.get(key)
            if pool is None:
                raise ValueError(f"teacher line {number}: id not in the recipe lists")
            if pool in drop:
                removed_by_pool[pool] += 1
                continue
            kept_lines.append(line)
            kept[key] = entry["input_sha256"]
            kept_by_pool[pool] += 1
    return kept_lines, kept, kept_by_pool, removed_by_pool


def mixture_coverage(
    path: Path, pool_of: dict[str, str], kept: dict[str, str]
) -> dict[str, Any]:
    rows = covered = 0
    covered_by_pool: Counter[str] = Counter()
    uncovered_by_pool: Counter[str] = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            pool = pool_of.get(row["id"])
            if pool is None:
                raise ValueError(f"{path.name}: row id not in the recipe lists")
            rows += 1
            if kept.get(row.get("teacher_source_id", row["id"])) == row["input_sha256"]:
                covered += 1
                covered_by_pool[pool] += 1
            else:
                uncovered_by_pool[pool] += 1
    return {
        "rows": rows,
        "covered": covered,
        "uncovered": rows - covered,
        "covered_by_pool": dict(sorted(covered_by_pool.items())),
        "uncovered_by_pool": dict(sorted(uncovered_by_pool.items())),
    }


def parse_mixture(item: str) -> tuple[Path, str, int]:
    path, sha, covered = item.rsplit("=", 2)
    return Path(path), sha, int(covered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--recipe", type=Path, action="append", required=True)
    parser.add_argument("--drop-pool", action="append", required=True)
    parser.add_argument(
        "--mixture", action="append", required=True, help="PATH=SHA256=COVERED"
    )
    parser.add_argument("--expect-kept", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    pending = args.output.with_name(args.output.name + ".pending")
    for path in (args.output, args.report, pending):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    teacher_sha = file_sha256(args.teacher)
    if teacher_sha != args.teacher_sha256:
        raise ValueError("teacher file differs from its expected sha256")
    mixtures = [parse_mixture(item) for item in args.mixture]
    mixture_sha = {path: file_sha256(path) for path, _, _ in mixtures}
    for path, sha, _ in mixtures:
        if mixture_sha[path] != sha:
            raise ValueError(f"{path.name}: mixture differs from its expected sha256")

    drop = set(args.drop_pool)
    pool_of = load_pools(args.recipe)
    lines, kept, kept_by_pool, removed_by_pool = split_teacher(
        args.teacher, pool_of, drop
    )
    per_mixture = {}
    for path, sha, expected in mixtures:
        entry = mixture_coverage(path, pool_of, kept)
        entry["sha256"] = sha
        entry["expected_covered"] = expected
        entry["covered_matches"] = entry["covered"] == expected
        entry["uncovered_only_in_dropped_pools"] = (
            set(entry["uncovered_by_pool"]) <= drop
        )
        per_mixture[path.name] = entry
    checks = {
        "kept_matches": len(lines) == args.expect_kept,
        "covered_match": all(e["covered_matches"] for e in per_mixture.values()),
        "uncovered_only_in_dropped_pools": all(
            e["uncovered_only_in_dropped_pools"] for e in per_mixture.values()
        ),
    }
    report: dict[str, Any] = {
        "tool": "v2.06b.m7_teacher",
        "input": {
            "path": str(args.teacher),
            "sha256": teacher_sha,
            "lines": len(lines) + sum(removed_by_pool.values()),
        },
        "recipes": {str(p): file_sha256(p) for p in args.recipe},
        "drop_pools": sorted(drop),
        "removed_lines": sum(removed_by_pool.values()),
        "removed_by_pool": dict(sorted(removed_by_pool.items())),
        "kept_by_pool": dict(sorted(kept_by_pool.items())),
        "expected_kept": args.expect_kept,
        "mixtures": per_mixture,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if report["pass"]:
        with pending.open("xb") as stream:
            stream.writelines(lines)
        output_sha = file_sha256(pending)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        os.replace(pending, args.output)
        report["output"] = {
            "path": str(args.output),
            "sha256": output_sha,
            "lines": len(lines),
        }
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "kept": len(lines),
                "removed": report["removed_lines"],
                "output_sha256": report.get("output", {}).get("sha256"),
                "covered": {k: e["covered"] for k, e in per_mixture.items()},
            }
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
