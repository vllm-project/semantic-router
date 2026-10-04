"""Replay rows: a fixed-seed sample of a released model's TRAIN union, excluding given ids.

The rows are later labelled with the released model itself (``v2.dec.teacher_label --teacher-kind dec
--uncalibrated``), so the KL term anchors the student to the model it continues from.

usage: python3 -m v2.reasoning.replay --train A.jsonl ... --exclude POOL.jsonl --count N --out REPLAY.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", default="reasoning-replay-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    excluded = set()
    for path in args.exclude:
        for line in path.open(encoding="utf-8"):
            excluded.add(json.loads(line)["id"])
    rows: dict[str, dict] = {}
    for path in args.train:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row["id"] not in excluded:
                rows.setdefault(row["id"], row)
    order = sorted(
        rows, key=lambda i: hashlib.sha256(f"{args.seed}:{i}".encode()).hexdigest()
    )
    chosen = [rows[i] for i in order[: args.count]]
    with args.out.open("w", encoding="utf-8") as sink:
        for row in chosen:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "union": len(rows),
                "excluded": len(excluded),
                "chosen": len(chosen),
                "by_type": Counter(r["task_type"] for r in chosen),
                "top_sources": Counter(r["source"] for r in chosen).most_common(12),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
