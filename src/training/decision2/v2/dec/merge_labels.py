"""Merge teacher-label files for one TRAIN mixture into a single trainer file.

Shards written by ``teacher_label --shard-index/--shard-count`` are disjoint;
``--part`` files must cover every TRAIN row exactly once (id, canonical input
hash and the complete option-key set), unless ``--override`` files replace some
rows: each override record must also match its TRAIN row, and the merged file
then records which teacher supplied every row. Output rows follow TRAIN order;
the manifest records each input file's hash and teacher identity (from the
``.manifest.json`` beside it, when present) and agreement with TRAIN labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.model.data import file_sha256, load_partition

MERGE_VERSION = "dec-teacher-merge/1"
IDENTITY = ("teacher_repo", "teacher_revision", "teacher_source_fingerprint")


def read_labels(
    path: Path, by_id: dict[str, dict[str, Any]], subset: bool = False
) -> dict[str, dict[str, float]]:
    labels: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            record = json.loads(line)
            row = by_id.get(record.get("id"))
            if row is None and subset:
                continue
            if row is None:
                raise ValueError(f"{path}:{number}: id not in TRAIN")
            if record.get("input_sha256") != row["input_sha256"]:
                raise ValueError(f"{path}:{number}: input hash differs from TRAIN")
            probs = record.get("teacher_probs")
            keys = {option["key"] for option in row["options"]}
            if not isinstance(probs, dict) or set(probs) != keys:
                raise ValueError(f"{path}:{number}: option keys differ from TRAIN")
            values = list(probs.values())
            if (
                any(
                    type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in values
                )
                or abs(math.fsum(values) - 1.0) > 1e-6
            ):
                raise ValueError(f"{path}:{number}: invalid distribution")
            if record["id"] in labels:
                raise ValueError(f"{path}:{number}: repeated id")
            labels[record["id"]] = probs
    return labels


def identity(path: Path) -> dict[str, Any]:
    manifest = path.with_name(path.name + ".manifest.json")
    info: dict[str, Any] = {"file": str(path), "sha256": file_sha256(path)}
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        info.update({key: data.get(key) for key in IDENTITY})
        info["teacher_temperatures"] = data.get("teacher_temperatures")
        info["manifest_train_sha256"] = data.get("train_sha256")
        info["shard"] = data.get("shard")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--part", type=Path, action="append", required=True)
    parser.add_argument("--override", type=Path, action="append", default=[])
    parser.add_argument(
        "--override-subset",
        action="store_true",
        help="Override files may hold rows of other mixtures: ids absent from TRAIN are "
        "skipped (a TRAIN id with a different input hash is still an error)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_partition(args.train, "train")
    by_id = {row["id"]: row for row in rows}
    train_sha = file_sha256(args.train)
    parts = [identity(p) for p in args.part]
    teachers = {json.dumps([p.get(k) for k in IDENTITY]) for p in parts}
    if len(teachers) != 1:
        raise ValueError("--part files come from different teachers")
    for info in parts + [identity(p) for p in args.override]:
        if info.get("manifest_train_sha256") not in (None, train_sha):
            raise ValueError(f"{info['file']} was labeled on a different TRAIN file")
    merged: dict[str, dict[str, float]] = {}
    origin: dict[str, str] = {}
    for path in args.part:
        for key, probs in read_labels(path, by_id).items():
            if key in merged:
                raise ValueError(f"{path}: id {key} already labeled by another part")
            merged[key] = probs
            origin[key] = "part"
    if set(merged) != set(by_id):
        raise ValueError(f"--part files cover {len(merged)} of {len(by_id)} TRAIN rows")
    overrides = [identity(p) for p in args.override]
    replaced: Counter = Counter()
    for index, path in enumerate(args.override):
        for key, probs in read_labels(path, by_id, args.override_subset).items():
            if origin[key] != "part":
                raise ValueError(f"{path}: id {key} already overridden")
            merged[key] = probs
            origin[key] = f"override{index}"
            replaced[f"override{index}"] += 1
    agreement: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        for row in rows:
            probs = merged[row["id"]]
            keys = [option["key"] for option in row["options"]]
            best = max(range(len(keys)), key=lambda i: probs[keys[i]])
            stats = agreement[origin[row["id"]]][row["task_type"]]
            stats[0] += int(best == row["label"])
            stats[1] += 1
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "input_sha256": row["input_sha256"],
                        "teacher_probs": {key: probs[key] for key in keys},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, args.output)
    manifest = {
        "merge_version": MERGE_VERSION,
        "train_sha256": train_sha,
        "rows": len(rows),
        "parts": parts,
        "overrides": overrides,
        "override_subset": args.override_subset,
        "rows_by_origin": dict(Counter(origin.values())),
        "overridden_rows": dict(replaced),
        "train_label_agreement": {
            source: {
                kind: {"correct": c, "n": n, "accuracy": c / n}
                for kind, (c, n) in sorted(by_type.items())
            }
            for source, by_type in sorted(agreement.items())
        },
        "output_sha256": file_sha256(args.output),
    }
    args.output.with_name(args.output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "rows_by_origin": manifest["rows_by_origin"]}))


if __name__ == "__main__":
    main()
