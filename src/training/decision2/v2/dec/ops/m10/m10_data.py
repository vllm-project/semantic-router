"""Decoder M10 TRAIN: the released 4B mixture minus the quarantined near-match group (standard library only).

The released DEV2.0-4B trained on N4XF's ``m4-xl-full-29m`` (c7d51219). Every M10 arm trains on that file without
the M7 quarantine group (3 HotpotQA rows; the M7 / M9 4B base), each line copied byte for byte, with N4XF's composed
own-Lux teacher (e2ff27ce, read partially: H7 / H8 rows are gold-only). The report records counts, teacher coverage
and the SHA-256 of every input and output.

usage: m10_data.py --base B --base-sha S --teacher T --teacher-sha S --quarantine Q --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(base: Path, teacher: Path, quarantine: set[str], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    kept: list[dict] = []
    dropped = 0
    with base.open("rb") as source, (output / "train.jsonl").open("xb") as sink:
        for line in source:
            row = json.loads(line)
            if row["group_id"] in quarantine:
                dropped += 1
                continue
            sink.write(line if line.endswith(b"\n") else line + b"\n")
            kept.append(
                {
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "input_sha256": row["input_sha256"],
                    "task_type": row["task_type"],
                    "source": row["source"],
                }
            )
    ids = [row["id"] for row in kept]
    if len(set(ids)) != len(ids):
        raise ValueError("TRAIN ids are not unique")
    by_id = {row["id"]: row for row in kept}
    covered = stale = extra = 0
    with teacher.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            row = by_id.get(record["id"])
            if row is None:
                extra += 1
            elif record["input_sha256"] != row["input_sha256"]:
                stale += 1
            else:
                covered += 1
    with (output / "train.ids.jsonl").open("x", encoding="utf-8") as stream:
        for row in kept:
            stream.write(
                json.dumps({"id": row["id"], "group_id": row["group_id"]}) + "\n"
            )
    return {
        "rows": len(kept),
        "quarantine_rows_dropped": dropped,
        "groups": len({row["group_id"] for row in kept}),
        "task_types": dict(sorted(Counter(row["task_type"] for row in kept).items())),
        "teacher_rows_covering_train": covered,
        "teacher_rows_hash_mismatch": stale,
        "teacher_rows_outside_train": extra,
        "gold_only_rows": len(kept) - covered,
        "train_sha256": sha256(output / "train.jsonl"),
        "train_ids_sha256": sha256(output / "train.ids.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--teacher-sha", required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.base, args.base_sha),
        (args.teacher, args.teacher_sha),
    ):
        if sha256(path) != expected:
            raise ValueError(f"{path} differs from its pinned SHA-256 {expected}")
    quarantine = set(json.loads(args.quarantine.read_text())["group_ids"])
    report = build(args.base, args.teacher, quarantine, args.output)
    report.update(
        {
            "base_sha256": args.base_sha,
            "teacher_sha256": args.teacher_sha,
            "quarantine_sha256": sha256(args.quarantine),
            "quarantine_group_ids": sorted(quarantine),
        }
    )
    (args.output / "data.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
