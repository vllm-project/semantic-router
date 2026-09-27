"""Make a group-preserving, no-truncation TRAIN subset for a pinned tokenizer.

This tool never changes SELECT or CAL. Its output is a new private experiment
input, not a replacement for the frozen rights-clean control.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from training.model.data import (
    canonical,
    check_partition_isolation,
    file_sha256,
    load_partition,
)

EXPECTED = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
OFFICIAL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CONFIG_SHA256 = "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"


def choose_groups(
    rows: list[dict[str, Any]], lengths: list[int], max_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject whole original groups if any member exceeds the native limit."""
    if max_length < 2 or len(rows) != len(lengths):
        raise ValueError("Invalid length audit")
    if any(length < 1 for length in lengths):
        raise ValueError("Native token lengths must be positive")
    groups = collections.defaultdict(list)
    for row, length in zip(rows, lengths, strict=True):
        groups[row["group_id"]].append((row, length))
    rejected = {
        group_id
        for group_id, items in groups.items()
        if any(length > max_length for _, length in items)
    }
    retained = [row for row in rows if row["group_id"] not in rejected]
    if not retained:
        raise ValueError("Native length filter removed every TRAIN group")
    retained_lengths = [
        length
        for row, length in zip(rows, lengths, strict=True)
        if row["group_id"] not in rejected
    ]
    return retained, {
        "source_rows": len(rows),
        "source_groups": len(groups),
        "overlength_rows": sum(length > max_length for length in lengths),
        "removed_rows": len(rows) - len(retained),
        "removed_groups": len(rejected),
        "removed_ids": [row["id"] for row in rows if row["group_id"] in rejected],
        "retained_rows": len(retained),
        "retained_groups": len(groups) - len(rejected),
        "retained_native_tokens": sum(retained_lengths),
        "retained_max_length": max(retained_lengths),
        "retained_by_type": dict(
            sorted(collections.Counter(row["task_type"] for row in retained).items())
        ),
        "retained_by_language": dict(
            sorted(collections.Counter(row["language"] for row in retained).items())
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    if (
        args.source_model != "Qwen/Qwen3.5-9B"
        or args.source_revision != OFFICIAL_REVISION
    ):
        raise ValueError("The source must be pinned official Qwen3.5-9B")
    if (
        file_sha256(args.model_path / "config.json") != CONFIG_SHA256
        or file_sha256(args.model_path / "tokenizer.json") != TOKENIZER_SHA256
    ):
        raise ValueError("Pinned official model configuration or tokenizer differs")
    paths = {name: getattr(args, name) for name in EXPECTED}
    digests = {name: file_sha256(path) for name, path in paths.items()}
    if digests != EXPECTED:
        raise ValueError("Frozen rights-clean v2 partition bytes differ")
    partitions = {name: load_partition(path, name) for name, path in paths.items()}
    check_partition_isolation(partitions)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    lengths = [
        len(encode(row, tokenizer, 1_000_000)["ids"]) for row in partitions["train"]
    ]
    retained, summary = choose_groups(partitions["train"], lengths, args.max_length)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "train.jsonl"
    with output.open("x", encoding="utf-8") as stream:
        for row in retained:
            stream.write(canonical(row) + "\n")
    reloaded = load_partition(output, "train")
    check_partition_isolation({**partitions, "train": reloaded})
    if [row["id"] for row in reloaded] != [row["id"] for row in retained]:
        raise RuntimeError("TRAIN output order changed")
    manifest = {
        "schema": "decision2-qwen35-9b-native-length-filter/1",
        "source_model": args.source_model,
        "source_revision": args.source_revision,
        "max_length": args.max_length,
        "input_sha256": digests,
        "output_sha256": file_sha256(output),
        "tokenizer_json_sha256": TOKENIZER_SHA256,
        "summary": summary,
        "role": "private_prospective_training_arm",
        "truncation": False,
    }
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    for name in EXPECTED:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    manifest = build(parser.parse_args())
    public_summary = {
        key: value for key, value in manifest["summary"].items() if key != "removed_ids"
    }
    print(
        json.dumps(
            {"summary": public_summary, "output_sha256": manifest["output_sha256"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
