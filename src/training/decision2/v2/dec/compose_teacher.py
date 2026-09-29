"""Compose one trainer teacher file for a TRAIN mixture from published target files.

Sources are ``{id, input_sha256, teacher_probs}`` JSONL files (the research &
data own-Lux files and this track's ``teacher_label`` output share the format),
given in precedence order: the first source holding a TRAIN id supplies its
target; later records for that id are only compared with it (argmax agreement
and mean absolute difference per source pair, reported, never used). Every
record for a TRAIN id must carry the row's canonical input hash and exactly its
option keys with a finite distribution summing to 1 within 1e-6 (the trainer's
check); values are copied unchanged. TRAIN rows without a target are an error
unless each belongs to an allowed recipe pool (``--allow-missing-pool
<recipe ids file>:<pool>``, for example the gold-only H7 / H8 gap arms); such a
file needs ``train_dec --teacher-partial``.

``missing`` writes the uncovered TRAIN rows, plus an optional seed-keyed sample
of covered rows as a labeling cross-check, as a TRAIN file for ``teacher_label``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.model.data import file_sha256, load_partition

COMPOSE_VERSION = "dec-teacher-compose/1"


def _valid(probs: Any, keys: list[str]) -> bool:
    if not isinstance(probs, dict) or set(probs) != set(keys):
        return False
    values = [probs[key] for key in keys]
    return (
        all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values)
        and abs(math.fsum(values) - 1.0) <= 1e-6
    )


def _argmax(probs: dict[str, float], keys: list[str]) -> int:
    return max(range(len(keys)), key=lambda i: probs[keys[i]])


def collect(
    rows: list[dict[str, Any]], sources: list[Path]
) -> tuple[dict[str, tuple[int, dict[str, float]]], dict[str, Any]]:
    """First-wins targets for TRAIN ids, per-source counts and pairwise overlaps."""
    by_id = {row["id"]: row for row in rows}
    chosen: dict[str, tuple[int, dict[str, float]]] = {}
    counts = [Counter() for _ in sources]
    overlap: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0])
    for index, path in enumerate(sources):
        seen: set[str] = set()
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                record = json.loads(line)
                counts[index]["records"] += 1
                row = by_id.get(record.get("id"))
                if row is None:
                    counts[index]["absent_from_train"] += 1
                    continue
                if record["id"] in seen:
                    raise ValueError(f"{path}:{number}: repeated id")
                seen.add(record["id"])
                if record.get("input_sha256") != row["input_sha256"]:
                    raise ValueError(f"{path}:{number}: input hash differs from TRAIN")
                keys = [option["key"] for option in row["options"]]
                probs = record.get("teacher_probs")
                if not _valid(probs, keys):
                    raise ValueError(f"{path}:{number}: invalid teacher distribution")
                if record["id"] in chosen:
                    first, kept = chosen[record["id"]]
                    stats = overlap[f"{first}>{index}"]
                    stats[0] += 1
                    stats[1] += int(_argmax(kept, keys) == _argmax(probs, keys))
                    stats[2] += math.fsum(abs(kept[k] - probs[k]) for k in keys) / len(
                        keys
                    )
                    counts[index]["shadowed"] += 1
                    continue
                chosen[record["id"]] = (index, probs)
                counts[index]["used"] += 1
    report = {
        "sources": [
            {"file": str(path), "sha256": file_sha256(path), **dict(counts[i])}
            for i, path in enumerate(sources)
        ],
        "overlap": {
            pair: {
                "records": int(n),
                "argmax_agreement": agree / n,
                "mean_abs_prob_diff": diff / n,
            }
            for pair, (n, agree, diff) in sorted(overlap.items())
        },
    }
    return chosen, report


def allowed_missing(specs: list[str]) -> dict[str, str]:
    allowed: dict[str, str] = {}
    for spec in specs:
        file_name, pool = spec.rsplit(":", 1)
        with Path(file_name).open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if entry["pool"] == pool:
                    allowed[entry["id"]] = pool
    return allowed


def compose(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_partition(args.train, "train")
    chosen, report = collect(rows, args.source)
    allowed = allowed_missing(args.allow_missing_pool)
    missing = [row for row in rows if row["id"] not in chosen]
    stray = [row["id"] for row in missing if row["id"] not in allowed]
    if stray:
        raise ValueError(
            f"{len(stray)} TRAIN rows lack a target outside the allowed pools "
            f"(first: {stray[0]})"
        )
    agreement: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        for row in rows:
            if row["id"] not in chosen:
                continue
            index, probs = chosen[row["id"]]
            keys = [option["key"] for option in row["options"]]
            stats = agreement[str(index)][row["task_type"]]
            stats[0] += int(_argmax(probs, keys) == row["label"])
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
        "compose_version": COMPOSE_VERSION,
        "train_sha256": file_sha256(args.train),
        "rows": len(rows),
        "covered": len(rows) - len(missing),
        "missing_by_pool": dict(Counter(allowed[row["id"]] for row in missing)),
        "missing_by_type": dict(Counter(row["task_type"] for row in missing)),
        "allow_missing_pool": args.allow_missing_pool,
        **report,
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
    return manifest


def missing_rows(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_partition(args.train, "train")
    chosen, report = collect(rows, args.source)
    uncovered = [row for row in rows if row["id"] not in chosen]
    covered = sorted(
        (row for row in rows if row["id"] in chosen),
        key=lambda row: hashlib.sha256(
            f"{args.seed}\0{row['id']}".encode()
        ).hexdigest(),
    )[: args.check_sample]
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        for row in uncovered + covered:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(pending, args.output)
    load_partition(args.output, "train")
    manifest = {
        "compose_version": COMPOSE_VERSION,
        "train_sha256": file_sha256(args.train),
        "uncovered": len(uncovered),
        "uncovered_by_type": dict(Counter(row["task_type"] for row in uncovered)),
        "check_sample": len(covered),
        "check_ids": [row["id"] for row in covered],
        "seed": args.seed,
        **report,
        "output_sha256": file_sha256(args.output),
    }
    args.output.with_name(args.output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("compose", "missing"):
        command = commands.add_parser(name)
        command.add_argument("--train", type=Path, required=True)
        command.add_argument("--source", type=Path, action="append", required=True)
        command.add_argument("--output", type=Path, required=True)
    commands.choices["compose"].add_argument(
        "--allow-missing-pool", action="append", default=[], metavar="IDS:POOL"
    )
    commands.choices["missing"].add_argument("--check-sample", type=int, default=0)
    commands.choices["missing"].add_argument("--seed", default="dec-m4")
    args = parser.parse_args()
    manifest = compose(args) if args.command == "compose" else missing_rows(args)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("rows", "covered", "uncovered", "check_sample")
                if k in manifest
            }
        )
    )


if __name__ == "__main__":
    main()
