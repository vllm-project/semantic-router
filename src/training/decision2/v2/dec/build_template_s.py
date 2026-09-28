"""Build matrix v1 budget-template-S training mixtures for one testbed tokenizer.

Control ``C(rho) = A0 ∪ A0-resample(rho)``; treatment ``A0 ∪ X(rho)`` with
``rho = min(0.5 × A0 tokens, X tokens)``. Tokens are native ``encode`` lengths
(segmented options, no truncation) under the testbed tokenizer. Both the
resample and the arm subsample take **whole groups**, stratified by
source × task type × language, in a fixed seed-keyed hash order: each stratum
receives a share of rho proportional to its token mass, and groups are added in
hash order until the stratum share is reached. Resampled A0 rows keep their
group and input but get a suffixed ID so the partition stays unique. The output
is a plain TRAIN partition plus a manifest of counts, tokens and selected groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.model.data import canonical, file_sha256, load_partition

RESAMPLE_SUFFIX = "~s1"


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    return groups


def stratum(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    first = rows[0]
    return (first["source"], first["task_type"], first["language"])


def select_groups(
    groups: dict[str, list[dict[str, Any]]],
    tokens: dict[str, int],
    rho: int,
    seed: str,
) -> list[str]:
    """Whole groups, stratified, hash-ordered, until each stratum's share of rho."""
    by_stratum: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for group_id, rows in groups.items():
        by_stratum[stratum(rows)].append(group_id)
    total = sum(tokens[g] for g in groups)
    if rho > total:
        raise ValueError("rho exceeds the available token mass")
    chosen: list[str] = []
    for key in sorted(by_stratum):
        members = sorted(
            by_stratum[key],
            key=lambda g: hashlib.sha256(f"{seed}\0{g}".encode()).hexdigest(),
        )
        share = rho * sum(tokens[g] for g in members) / total
        taken = 0
        for group_id in members:
            if taken >= share:
                break
            chosen.append(group_id)
            taken += tokens[group_id]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--arm", type=Path, help="Arm TRAIN rows; omit for the control")
    parser.add_argument("--arm-name", default="control")
    parser.add_argument(
        "--rho", type=int, help="Token budget; default min(0.5×A0, arm)"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed", default="dec-template-s-v1")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    def lengths(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            row["id"]: len(encode(row, tokenizer, args.max_length)["ids"])
            for row in rows
        }

    a0 = load_partition(args.a0, "train")
    a0_len = lengths(a0)
    a0_tokens = sum(a0_len.values())
    if args.arm:
        pool = load_partition(args.arm, "train")
        if {r["id"] for r in pool} & {r["id"] for r in a0}:
            raise ValueError("Arm row IDs collide with A0")
        pool_len = lengths(pool)
    else:
        pool, pool_len = a0, a0_len
    pool_tokens = sum(pool_len.values())
    rho = args.rho if args.rho is not None else min(a0_tokens // 2, pool_tokens)
    groups = group_rows(pool)
    group_tokens = {
        g: sum(pool_len[r["id"]] for r in rows) for g, rows in groups.items()
    }
    chosen = select_groups(groups, group_tokens, rho, f"{args.seed}\0{args.arm_name}")
    added = [row for g in chosen for row in groups[g]]
    if not args.arm:
        added = [dict(row, id=row["id"] + RESAMPLE_SUFFIX) for row in added]
    mixture = a0 + added
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        for row in mixture:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    pending.replace(args.output)
    load_partition(args.output, "train")
    added_tokens = sum(group_tokens[g] for g in chosen)
    counts: dict[str, int] = defaultdict(int)
    for row in added:
        counts[row["task_type"]] += 1
    manifest = {
        "schema_version": "dec-template-s/1",
        "arm": args.arm_name,
        "a0_sha256": file_sha256(args.a0),
        "arm_sha256": file_sha256(args.arm) if args.arm else None,
        "tokenizer_files_sha256": {
            name: file_sha256(args.tokenizer / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
            if (args.tokenizer / name).is_file()
        },
        "seed": args.seed,
        "rho_tokens": rho,
        "a0_rows": len(a0),
        "a0_tokens": a0_tokens,
        "pool_tokens": pool_tokens,
        "added_groups": len(chosen),
        "added_rows": len(added),
        "added_tokens": added_tokens,
        "added_rows_by_type": dict(counts),
        "total_rows": len(mixture),
        "total_tokens": a0_tokens + added_tokens,
        "selected_groups_sha256": hashlib.sha256(
            canonical(sorted(chosen)).encode()
        ).hexdigest(),
        "output_sha256": file_sha256(args.output),
    }
    args.output.with_name(args.output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
