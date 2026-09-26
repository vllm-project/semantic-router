"""Freeze a source-native, TRAIN-only 0.8B continuation sample.

Run on an authorized experiment host. This module never reads benchmark gold,
SELECT labels for model choice, or teacher outputs. It preserves the published
Joyfox row encoding and fails closed on length or lineage mismatches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inference.joyfox import MODEL_REVISION, verify_release

from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)

CONTRACT = "joyfox08-native1024-cleanv2-lora64-v1"
SEED = "decision2-joyfox08-cleanv2-20260927-v1"
MAX_TOKENS = 1024
QUOTAS = {"choice": 128, "noul": 192, "score": 192}
HASHES = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    "rights_manifest": "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
}


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def to_record(row: dict[str, Any]) -> dict[str, Any]:
    """Map one flattened supervised row to the released native request schema."""
    kind = row["task_type"]
    question: dict[str, Any] = {
        "type": kind,
        "instructions": as_text(row["instructions"]),
    }
    if kind == "choice":
        question["criteria"] = {
            option["key"]: as_text(option["description"]) for option in row["options"]
        }
    elif kind == "score":
        ordered = sorted(row["options"], key=lambda option: int(option["key"]))
        if [option["key"] for option in ordered] != [
            str(i) for i in range(len(ordered))
        ]:
            raise ValueError("Score levels are not contiguous")
        question["criteria"] = [as_text(option["description"]) for option in ordered]
    elif kind != "noul":
        raise ValueError(f"Unsupported task type: {kind}")
    # The published adapter's Noul candidates are its canonical false/true
    # strings. Avoid a TRAIN-only criteria addition that changes their encoding.
    return {"state": row["state"], "questions": {"decision": question}}


def target_key(row: dict[str, Any]) -> str:
    return row["options"][row["label"]]["key"]


def _rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest()


def choose_rows(eligible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = []
    for kind, count in QUOTAS.items():
        pool = sorted((row for row in eligible if row["task_type"] == kind), key=_rank)
        if len(pool) < count:
            raise ValueError(
                f"Insufficient native-1024 {kind} TRAIN rows: {len(pool)} < {count}"
            )
        chosen.extend(pool[:count])
    return sorted(chosen, key=lambda row: row["id"])


def assess_eligibility(
    rows: list[dict[str, Any]],
    encode: Callable[..., dict[str, Any]],
    validate_record: Callable[..., None],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = []
    overflow: Counter[str] = Counter()
    schema_invalid: Counter[str] = Counter()
    longest = 0
    for row in rows:
        record = to_record(row)
        try:
            validate_record(record)
            encoded = encode(tokenizer, record, MAX_TOKENS)
        except ValueError as exc:
            if "input requires" in str(exc):
                overflow[row["task_type"]] += 1
                continue
            if "at most 128 candidates" in str(exc):
                schema_invalid[row["task_type"]] += 1
                continue
            if "candidate descriptions must be nonempty text" in str(exc):
                schema_invalid[row["task_type"]] += 1
                continue
            else:
                raise
        keys = [option["key"] for option in row["options"]]
        if row["task_type"] == "score":
            keys = [str(i) for i in range(len(keys))]
        if target_key(row) not in keys:
            raise ValueError("TRAIN target absent from encoded option set")
        length = max(
            encoded["segments"].count(0) + encoded["segments"].count(i)
            for i in set(encoded["segments"]) - {0}
        )
        longest = max(longest, length)
        eligible.append(row)
    return eligible, {
        "eligible_by_type": dict(Counter(row["task_type"] for row in eligible)),
        "overflow_by_type": dict(overflow),
        "native_schema_invalid_by_type": dict(schema_invalid),
        "eligible_max_tokens": longest,
    }


def prepare(
    *,
    model_path: Path,
    source_path: Path,
    train: Path,
    select: Path,
    cal: Path,
    rights_manifest: Path,
    selected_output: Path,
    manifest_output: Path,
) -> dict[str, Any]:
    if selected_output.exists() or manifest_output.exists():
        raise FileExistsError("Pilot output already exists")
    actual = {
        "train": file_sha256(train),
        "select": file_sha256(select),
        "cal": file_sha256(cal),
        "rights_manifest": file_sha256(rights_manifest),
    }
    if actual != HASHES:
        raise ValueError("Frozen rights-clean v2 source bytes changed")
    partitions = {
        name: load_partition(path, name)
        for name, path in (("train", train), ("select", select), ("cal", cal))
    }
    if {name: len(rows) for name, rows in partitions.items()} != {
        "train": 7455,
        "select": 700,
        "cal": 700,
    }:
        raise ValueError("Frozen rights-clean v2 row counts changed")
    check_partition_isolation(partitions)
    source = verify_release(
        model_path.resolve(strict=True),
        source_path.resolve(strict=True),
        MODEL_REVISION,
    )
    sys.path.insert(0, str((source_path / "src").resolve(strict=True)))
    from jev_inference.model import encode, validate_record
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path / "tokenizer", local_files_only=True
    )
    eligible, audit = assess_eligibility(
        partitions["train"], encode, validate_record, tokenizer
    )
    selected = choose_rows(eligible)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    with selected_output.open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = {
        "contract": CONTRACT,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "source": source,
        "source_model_revision": MODEL_REVISION,
        "source_data_sha256": HASHES,
        "selected_sha256": file_sha256(selected_output),
        "selected_count": len(selected),
        "selected_by_type": dict(Counter(row["task_type"] for row in selected)),
        "selected_by_language": dict(Counter(row["language"] for row in selected)),
        "selected_by_family": dict(Counter(row["family"] for row in selected)),
        **audit,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "model-path",
        "source-path",
        "train",
        "select",
        "cal",
        "rights-manifest",
        "selected-output",
        "manifest-output",
    ):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(**vars(args))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
