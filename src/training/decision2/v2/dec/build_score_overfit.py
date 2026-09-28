"""Select a Score-only overfit set spanning every level count.

For each level count L, rows are taken one per group in a seed-keyed hash
order, round-robin over the gold level, until ``--quota`` rows (or the pool)
are reached, so every level of every level count appears as gold. Rows above
``--max-tokens`` native tokens are skipped. Writes the TRAIN partition, the same
rows under the SELECT contract (for scoring the fit with ``eval_rows``) and a
manifest of counts by level count and gold level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.model.data import file_sha256, load_partition


def gold_level(row: dict[str, Any]) -> int:
    return int(row["options"][row["label"]]["key"])


def select_rows(
    rows: list[dict[str, Any]], quota: int, seed: str
) -> list[dict[str, Any]]:
    by_count: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_count[len(row["options"])][gold_level(row)].append(row)
    chosen: list[dict[str, Any]] = []
    for count in sorted(by_count):
        queues = {
            level: sorted(
                members,
                key=lambda r: hashlib.sha256(f"{seed}\0{r['id']}".encode()).hexdigest(),
            )
            for level, members in sorted(by_count[count].items())
        }
        groups: set[str] = set()
        taken: list[dict[str, Any]] = []
        while len(taken) < quota and any(queues.values()):
            for level in sorted(queues):
                queue = queues[level]
                while queue and queue[0]["group_id"] in groups:
                    queue.pop(0)
                if queue and len(taken) < quota:
                    row = queue.pop(0)
                    groups.add(row["group_id"])
                    taken.append(row)
        chosen.extend(taken)
    return chosen


def as_select(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row, split="select", evaluation_role="select")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--quota", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed", default="dec-score-overfit-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)

    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in args.input:
        for row in load_partition(path, "train"):
            if row["task_type"] != "score" or row["input_sha256"] in seen:
                continue
            tokens = len(encode(row, tokenizer, 8192)["ids"])
            if tokens <= args.max_tokens:
                seen.add(row["input_sha256"])
                pool.append(row)
    chosen = select_rows(pool, args.quota, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (
        ("train.jsonl", chosen),
        ("select.jsonl", [as_select(r) for r in chosen]),
    ):
        path = args.output_dir / name
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        files[name] = file_sha256(path)
    load_partition(args.output_dir / "train.jsonl", "train")
    load_partition(args.output_dir / "select.jsonl", "select")
    cells = Counter((len(r["options"]), gold_level(r)) for r in chosen)
    manifest = {
        "schema_version": "dec-score-overfit/1",
        "inputs_sha256": {str(p): file_sha256(p) for p in args.input},
        "seed": args.seed,
        "quota_per_level_count": args.quota,
        "max_tokens": args.max_tokens,
        "pool_rows": len(pool),
        "rows": len(chosen),
        "rows_by_level_count": dict(
            sorted(Counter(len(r["options"]) for r in chosen).items())
        ),
        "rows_by_level_count_and_gold": {
            f"L{count}": {str(level): cells[(count, level)] for level in range(count)}
            for count in sorted({c for c, _ in cells})
        },
        "sources": dict(Counter(r["source"] for r in chosen)),
        "files_sha256": files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
