"""Create a deterministic, family-balanced pilot subset of one split.

Each invocation samples one already assigned TRAIN or SELECT file. Its rows
retain their original roles and cannot move across partitions. The subset is
an experiment artifact, not a replacement for source rights/provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(
    source: Path,
    output: Path,
    *,
    limit: int,
    seed: int,
    role: str = "train",
    floor: int = 4,
    tokenizer_path: Path | None = None,
    max_token_length: int | None = None,
) -> dict:
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(output)
    if limit < 1 or floor < 0:
        raise ValueError("limit must be positive and floor nonnegative")
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in lines]
    if role not in ("train", "select") or any(
        row.get("evaluation_role") != role for row in rows
    ):
        raise ValueError("Source rows must match the requested TRAIN or SELECT role")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source IDs")
    source_items = len(rows)
    source_sha = sha(source)
    length_filter = None
    if (tokenizer_path is None) != (max_token_length is None):
        raise ValueError("Tokenizer and max token length must be supplied together")
    if tokenizer_path is not None:
        if max_token_length < 1:
            raise ValueError("Max token length must be positive")
        from training.model.decision_model import segments
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        keep = []
        for index, row in enumerate(rows):
            prefix, options, suffix = segments(row)
            # Match the trainer's per-segment tokenization and strict length check.
            token_count = len(tokenizer.encode(prefix, add_special_tokens=False))
            token_count += sum(
                len(tokenizer.encode(option, add_special_tokens=False))
                for option in options
            )
            token_count += len(tokenizer.encode(suffix, add_special_tokens=False))
            if token_count <= max_token_length:
                keep.append(index)
        lines = [lines[index] for index in keep]
        rows = [rows[index] for index in keep]
        length_filter = {
            "tokenizer_path": str(tokenizer_path),
            "max_token_length": max_token_length,
            "excluded_items": source_items - len(rows),
            "eligible_items": len(rows),
        }
    if limit > len(rows):
        raise ValueError("Requested more than eligible source count")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["family"]].append(index)
    if floor * len(groups) > limit:
        raise ValueError("Family floor exceeds target size")
    rng = random.Random(seed)
    for bucket in groups.values():
        rng.shuffle(bucket)
    chosen = Counter()
    for family in sorted(groups):
        chosen[family] = min(floor, len(groups[family]))
    while sum(chosen.values()) < limit:
        family = min(
            (name for name in groups if chosen[name] < len(groups[name])),
            key=lambda name: (
                (chosen[name] - min(floor, len(groups[name])))
                / math.sqrt(len(groups[name])),
                name,
            ),
        )
        chosen[family] += 1
    indices = [
        index for family in sorted(groups) for index in groups[family][: chosen[family]]
    ]
    rng.shuffle(indices)
    selected = [rows[index] for index in indices]
    # A source group may intentionally provide both Choice and Noul views.
    # Its split assignment was fixed upstream; shared groups inside TRAIN are valid.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(lines[index] + "\n" for index in indices), encoding="utf-8"
    )
    manifest = {
        "version": "decision2-balanced-partition-subset/2",
        "source_sha256": source_sha,
        "output_sha256": sha(output),
        "source_items": source_items,
        "eligible_items": len(rows),
        "length_filter": length_filter,
        "items": len(selected),
        "seed": seed,
        "role": role,
        "family_floor": floor,
        "unique_group_ids": len({row["group_id"] for row in selected}),
        "family_counts": dict(sorted(chosen.items())),
        "task_type_counts": dict(
            sorted(Counter(row["task_type"] for row in selected).items())
        ),
        "language_counts": dict(
            sorted(Counter(row["language"] for row in selected).items())
        ),
        "selection": "Per-family minimum then square-root-frequency water filling; deterministic shuffled output",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--role", choices=("train", "select"), default="train")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--floor", type=int, default=4)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-token-length", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            sample(
                args.source,
                args.output,
                limit=args.limit,
                seed=args.seed,
                role=args.role,
                floor=args.floor,
                tokenizer_path=args.tokenizer_path,
                max_token_length=args.max_token_length,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
