"""CPU-only lock for the official Qwen3 0.6B external-teacher arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)
from training.model.decision_model import encode
from training.model.external_teacher import attach_external_teacher
from training.model.source import source_fingerprint

PARTITIONS = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "select", "cal", "model_path", "teacher"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    for name, expected in PARTITIONS.items():
        if file_sha256(getattr(args, name)) != expected:
            raise ValueError(f"Frozen {name} partition bytes differ")
    partitions = {
        name: load_partition(getattr(args, name), name) for name in PARTITIONS
    }
    check_partition_isolation(partitions)
    source = source_fingerprint(args.model_path)
    attached = attach_external_teacher(
        args.teacher,
        partitions["train"],
        train_sha256=PARTITIONS["train"],
        source_files_sha256=source["files_sha256"],
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), local_files_only=True
    )
    encoded = [encode(row, tokenizer, 8192) for row in partitions["train"]]
    tokens = sum(len(item["ids"]) for item in encoded)
    if (
        len(encoded) != 7455
        or tokens != 4_094_489
        or max(len(item["ids"]) for item in encoded) > 8192
        or sum(item["teacher_probs"] is not None for item in encoded) != 3690
        or any(
            item["teacher_probs"] is not None
            for item in encoded
            if item["task_type"] == "score"
        )
    ):
        raise ValueError("Frozen token, option or Score-hard-label parity differs")
    print(
        json.dumps(
            {
                "source": source,
                "partitions_sha256": PARTITIONS,
                "teacher": attached,
                "train_rows": len(encoded),
                "train_tokens": tokens,
                "train_max_tokens": max(len(item["ids"]) for item in encoded),
                "score_soft_targets": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
