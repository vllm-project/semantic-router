"""Audit pinned official Qwen source token exposure before native training."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)
from training.model.decision_model import encode

EXPECTED = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}


def _quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * fraction)]


def audit(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if not args.source_model.startswith("Qwen/") or not re.fullmatch(
        r"[0-9a-f]{40}", args.source_revision
    ):
        raise ValueError("Official Qwen source and immutable revision are required")
    paths = {name: getattr(args, name) for name in EXPECTED}
    digests = {name: file_sha256(path) for name, path in paths.items()}
    if digests != EXPECTED:
        raise ValueError("Pinned rights-clean v2 TRAIN/SELECT/CAL bytes differ")
    partitions = {name: load_partition(path, name) for name, path in paths.items()}
    if {name: len(rows) for name, rows in partitions.items()} != {
        "train": 7455,
        "select": 700,
        "cal": 700,
    }:
        raise ValueError("Partition cardinalities differ")
    check_partition_isolation(partitions)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    summaries = {}
    for name, rows in partitions.items():
        lengths = []
        overlength = []
        by_type = collections.Counter()
        by_language = collections.Counter()
        for row in rows:
            item = encode(row, tokenizer, 1_000_000)
            length = len(item["ids"])
            lengths.append(length)
            by_type[row["task_type"]] += 1
            by_language[row["language"]] += 1
            if length > args.max_length:
                overlength.append(row["id"])
        summaries[name] = {
            "rows": len(rows),
            "by_type": dict(sorted(by_type.items())),
            "by_language": dict(sorted(by_language.items())),
            "token_total": sum(lengths),
            "token_max": max(lengths),
            "token_p50": _quantile(lengths, 0.50),
            "token_p95": _quantile(lengths, 0.95),
            "token_p99": _quantile(lengths, 0.99),
            "overlength_count": len(overlength),
            "overlength_ids": overlength,
        }
    result = {
        "schema": "decision2-official-qwen-native-input-preflight/2",
        "source_model": args.source_model,
        "source_revision": args.source_revision,
        "max_length": args.max_length,
        "input_sha256": digests,
        "partitions": summaries,
        "status": (
            "PASS"
            if all(value["overlength_count"] == 0 for value in summaries.values())
            else "HOLD_OVERLENGTH"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    for name in EXPECTED:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", required=True, type=Path)
    result = audit(parser.parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "partitions": {
                    name: {
                        key: value[key]
                        for key in (
                            "rows",
                            "token_total",
                            "token_max",
                            "overlength_count",
                        )
                    }
                    for name, value in result["partitions"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
