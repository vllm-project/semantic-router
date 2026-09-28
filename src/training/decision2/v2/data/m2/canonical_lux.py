"""Consolidate own-Lux teacher files over A0 TRAIN into one canonical file.

    python3 -m v2.data.m2.canonical_lux --train rights_clean.train.jsonl \\
        --native lux1-train.predictions.jsonl \\
        --compare dec=lux-train-teacher.jsonl --compare 9b=train-teacher.jsonl \\
        --out lux1-a0-train.canonical.jsonl --report report.json

The canonical targets come from the native System One collector output
(``inference.run --backend lux``: row id, model identity, revision, prompt
digest). Other track files are cross-checks only: per type argmax agreement
and maximum absolute probability difference on shared ids. Output rows are
``{id, input_sha256, teacher_probs}`` sorted by id; every TRAIN id must be
covered exactly once.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from training.model.data import canonical, file_sha256
from v2.data.m2.common import read_jsonl
from v2.data.replay_targets import teacher_distribution


def load_compare(
    path: Path, rows: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for record in read_jsonl(path):
        ident = record.get("id") or record.get("source_row_id")
        if ident not in rows:
            continue
        keys = [option["key"] for option in rows[ident]["options"]]
        if "teacher_probs" in record:
            probs = record["teacher_probs"]
        else:
            if record.get("valid") is False:
                continue
            values = record["probabilities"]
            probs = dict(zip(keys, values))
        out[ident] = {key: float(probs[key]) for key in keys}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = {row["id"]: row for row in read_jsonl(args.train)}
    identities: collections.Counter[tuple[str, str]] = collections.Counter()
    canonical_probs: dict[str, dict[str, float]] = {}
    missing = 0
    for record in read_jsonl(args.native):
        row = rows.get(record["id"])
        if row is None:
            raise ValueError(f"{record['id']}: not a TRAIN id")
        answer = record["answers"].get("decision")
        if answer is None or "error" in answer:
            missing += 1
            continue
        identities[(record.get("model_id"), record.get("model_revision"))] += 1
        canonical_probs[record["id"]] = teacher_distribution(row, answer)
    if set(canonical_probs) != set(rows) or missing:
        raise ValueError(
            f"native file covers {len(canonical_probs)} of {len(rows)} TRAIN ids ({missing} invalid)"
        )
    if len(identities) != 1:
        raise ValueError(f"mixed teacher identities {dict(identities)}")
    data = "".join(
        canonical(
            {
                "id": ident,
                "input_sha256": rows[ident]["input_sha256"],
                "teacher_probs": canonical_probs[ident],
            }
        )
        + "\n"
        for ident in sorted(canonical_probs)
    ).encode("utf-8")
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    (model_id, revision), _ = identities.most_common(1)[0]
    report: dict[str, Any] = {
        "schema": "decision2-canonical-lux/v1",
        "train_sha256": file_sha256(args.train),
        "native_file_sha256": file_sha256(args.native),
        "model_id": model_id,
        "model_revision": revision,
        "rows": len(canonical_probs),
        "content_sha256": hashlib.sha256(data).hexdigest(),
        "by_type": dict(
            sorted(
                collections.Counter(
                    rows[i]["task_type"] for i in canonical_probs
                ).items()
            )
        ),
        "compare": {},
    }
    for spec in args.compare:
        name, _, path = spec.partition("=")
        other = load_compare(Path(path), rows)
        stats: dict[str, dict[str, Any]] = {}
        for ident, probs in other.items():
            kind = rows[ident]["task_type"]
            mine = canonical_probs[ident]
            cell = stats.setdefault(
                kind,
                {
                    "shared": 0,
                    "argmax_agree": 0,
                    "max_abs_diff": 0.0,
                    "sum_abs_diff": 0.0,
                },
            )
            keys = list(mine)
            cell["shared"] += 1
            cell["argmax_agree"] += int(
                max(keys, key=mine.get) == max(keys, key=probs.get)
            )
            diff = max(abs(mine[k] - probs[k]) for k in keys)
            cell["max_abs_diff"] = max(cell["max_abs_diff"], diff)
            cell["sum_abs_diff"] += diff
        for cell in stats.values():
            cell["mean_max_abs_diff"] = round(
                cell.pop("sum_abs_diff") / cell["shared"], 6
            )
            cell["max_abs_diff"] = round(cell["max_abs_diff"], 6)
        report["compare"][name] = {
            "file_sha256": file_sha256(Path(path)),
            "rows": len(other),
            "by_type": stats,
        }
    fd = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, sort_keys=True)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
