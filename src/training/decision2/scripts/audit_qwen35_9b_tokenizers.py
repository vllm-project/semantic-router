"""Compare native Decision tokenization for two pinned Qwen3.5 sources."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from training.model.data import canonical, load_partition
from training.model.decision_model import encode
from transformers import AutoTokenizer


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--posttrained", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    base = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    post = AutoTokenizer.from_pretrained(args.posttrained, local_files_only=True)
    report = {
        "source_hashes": {
            stage: {
                name: sha(directory / name)
                for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
            }
            for stage, directory in (
                ("base", args.base),
                ("posttrained", args.posttrained),
            )
        },
        "partition_hashes": {
            name: sha(path)
            for name, path in (
                ("train", args.train),
                ("select", args.select),
                ("cal", args.cal),
            )
        },
        "partitions": {},
    }
    for name, path in (
        ("train", args.train),
        ("select", args.select),
        ("cal", args.cal),
    ):
        rows = load_partition(path, name)
        mismatches = []
        base_tokens = post_tokens = 0
        for row in rows:
            one = encode(row, base, args.max_length)
            two = encode(row, post, args.max_length)
            base_tokens += len(one["ids"])
            post_tokens += len(two["ids"])
            if one["ids"] != two["ids"]:
                mismatches.append(row["id"])
        report["partitions"][name] = {
            "rows": len(rows),
            "base_tokens": base_tokens,
            "posttrained_tokens": post_tokens,
            "different_native_token_ids": len(mismatches),
            "first_different_ids": mismatches[:10],
        }
    print(canonical(report))
    if any(
        part["different_native_token_ids"] for part in report["partitions"].values()
    ):
        raise SystemExit("Native tokenizer IDs differ on the frozen inputs")


if __name__ == "__main__":
    main()
