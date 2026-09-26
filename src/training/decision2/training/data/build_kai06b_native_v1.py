"""Convert the frozen rights-clean splits to Kai's exact native fine-tune format.

The published Kai SystemOne converter and tokenizer are imported from its pinned
snapshot. Every source row is either emitted or listed in a whole-group
quarantine receipt; no text is truncated and no row is silently discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from inference.kai_lex import MODELS, verify_native_bundle

from training.model.data import check_partition_isolation, load_partition

PARENT_SHA = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    "manifest": "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
}
PARENT_FILES = {
    "train": "rights_clean.train.jsonl",
    "select": "select.jsonl",
    "cal": "cal.jsonl",
    "manifest": "rights_clean.manifest.json",
}
MODEL_NAME = "Decision-1.0-Kai"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def convert(row: dict[str, Any], converter: Any) -> tuple[dict[str, Any], bool]:
    """Preserve source candidate order and native SystemOne input semantics."""
    kind = row["task_type"]
    options = row["options"]
    label = row["label"]
    fallback = False
    question: dict[str, Any] = {"type": kind, "instructions": row["instructions"]}
    if kind == "choice":
        question["criteria"] = {
            option["key"]: option["description"] for option in options
        }
        target = {"choice_id": options[label]["key"]}
        hard = options[label]["key"]
    elif kind == "score":
        criteria = []
        for option in options:
            description = option["description"]
            if description is None:
                description = option["key"]
                fallback = True
            criteria.append(description)
        question["criteria"] = criteria
        target = {"probabilities": [float(i == label) for i in range(len(options))]}
        hard = str(label)
    elif kind == "noul":
        keyed = {option["key"]: option["description"] for option in options}
        if set(keyed) != {"false", "true"}:
            raise ValueError("Native Noul requires original false/true candidate keys")
        criteria = {key: value for key, value in keyed.items() if value is not None}
        if criteria:
            question["criteria"] = criteria
        hard = "yes" if options[label]["key"] == "true" else "no"
        target = {"probability": float(hard == "yes")}
    else:
        raise ValueError("Unsupported frozen task type")
    request = {
        "model": MODEL_NAME,
        "state": row["state"],
        "questions": {"decision": question},
    }
    converted = converter(
        request,
        {"decision": target},
        request_id=row["id"],
        source_id=row["source"],
        component_id=row["group_id"],
        hard_target_ids={"decision": hard},
    )
    if len(converted) != 1:
        raise ValueError("A single source decision yielded multiple native rows")
    return converted[0], fallback


def _read_parent(
    parent: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    for role, name in PARENT_FILES.items():
        path = parent / name
        if path.is_symlink() or not path.is_file() or sha(path) != PARENT_SHA[role]:
            raise ValueError(f"Frozen rights-clean {role} bytes differ")
    manifest = json.loads((parent / PARENT_FILES["manifest"]).read_text())
    if (
        manifest.get("publication_eligible") is not True
        or manifest.get("derivation_version") != "decision2-goemotions-human-v2/1"
    ):
        raise ValueError("Expected the audited rights-clean human v2 manifest")
    splits = {
        role: load_partition(parent / PARENT_FILES[role], role)
        for role in ("train", "select", "cal")
    }
    check_partition_isolation(splits)
    return splits, manifest


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((canonical_text(row) + "\n").encode("utf-8") for row in rows)


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def _token_stats(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[int((len(ordered) - 1) * 0.95)],
        "max": ordered[-1],
    }


def build(parent: Path, model: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    splits, parent_manifest = _read_parent(parent)
    model = model.resolve(strict=True)
    identity = verify_native_bundle(model, "kai", MODELS["kai"]["revision"])
    sys.path.insert(0, str(model))
    from decision_finetune.data import input_signature, load_splits
    from decision_finetune.system_one import system_one_training_rows
    from native.policy.packing import MarkerCollator
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model / "native/tokenizer"), local_files_only=True
    )
    # A large cap measures the complete input. The published trainer later
    # performs its own independent exact 1,024-token admission check.
    collator = MarkerCollator(tokenizer, max_length=1_000_000, state_truncation="error")
    converted: dict[str, list[tuple[dict[str, Any], dict[str, Any], int]]] = {}
    fallback_ids: list[str] = []
    overlong: dict[str, set[str]] = {}
    for role in ("train", "select"):
        converted[role] = []
        overlong[role] = set()
        for row in splits[role]:
            native, fallback = convert(row, system_one_training_rows)
            encoded = collator.encode(native, labeled=True)
            length = encoded["input_tokens"]
            if fallback:
                fallback_ids.append(row["id"])
            converted[role].append((row, native, length))
            if length > 1024:
                overlong[role].add(row["group_id"])
    # Native serialization may collapse a distinction retained by the source
    # schema. Retain SELECT and quarantine the entire conflicting TRAIN group.
    select_signatures = {
        input_signature(native)
        for row, native, length in converted["select"]
        if row["group_id"] not in overlong["select"]
    }
    native_overlap_groups = {
        row["group_id"]
        for row, native, length in converted["train"]
        if row["group_id"] not in overlong["train"]
        and input_signature(native) in select_signatures
    }
    accepted: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    quarantine: list[dict[str, Any]] = []
    for role in ("train", "select"):
        excluded = overlong[role] | (
            native_overlap_groups if role == "train" else set()
        )
        accepted[role] = [
            native
            for row, native, length in converted[role]
            if row["group_id"] not in excluded
        ]
        for row, native, length in converted[role]:
            if row["group_id"] in excluded:
                reason = (
                    "over_1024_whole_group"
                    if row["group_id"] in overlong[role]
                    else "native_train_select_overlap_whole_group"
                )
                quarantine.append(
                    {
                        "role": role,
                        "source_id": row["source"],
                        "component_id": row["group_id"],
                        "source_row_id": row["id"],
                        "task_type": row["task_type"],
                        "full_input_tokens": length,
                        "reason": reason,
                    }
                )
        admitted = [
            (row, length)
            for row, native, length in converted[role]
            if row["group_id"] not in excluded
        ]
        if not admitted:
            raise ValueError(
                f"No native {role} rows remain after explicit group quarantine"
            )
        audit[role] = {
            "source_rows": len(splits[role]),
            "native_rows": len(admitted),
            "quarantined_rows": len(converted[role]) - len(admitted),
            "quarantined_groups": len(excluded),
            "quarantined_by_reason": dict(
                sorted(
                    Counter(
                        q["reason"] for q in quarantine if q["role"] == role
                    ).items()
                )
            ),
            "admitted_source_counts": _counts([row for row, _ in admitted], "source"),
            "admitted_task_type_counts": _counts(
                [row for row, _ in admitted], "task_type"
            ),
            "full_input_tokens": _token_stats([length for _, length in admitted]),
        }
    stage = output.with_name(output.name + ".pending")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(stage)
    stage.mkdir(parents=True, mode=0o700)
    for role in ("train", "select"):
        (stage / f"{role}.jsonl").write_bytes(_jsonl(accepted[role]))
    (stage / "quarantine.jsonl").write_bytes(_jsonl(quarantine))
    train, select = load_splits(
        stage / "train.jsonl", stage / "select.jsonl", "hard-accuracy"
    )
    if len(train) != len(accepted["train"]) or len(select) != len(accepted["select"]):
        raise AssertionError("Native fine-tune split validation changed row count")
    manifest = {
        "schema_version": "decision2-kai06b-native-rights-clean-v1/1",
        "generator_code_sha256": sha(Path(__file__)),
        "base_model": identity,
        "base_model_manifest_sha256": identity["model_config_sha256"],
        "parent_manifest_sha256": PARENT_SHA["manifest"],
        "parent_split_sha256": {
            role: PARENT_SHA[role] for role in ("train", "select", "cal")
        },
        "parent_publication_scope": parent_manifest["publication_scope"],
        "parent_source_rights": parent_manifest["source_rights"],
        "rights_basis": "inherited from the exact rights-clean v2 source manifest; admitted source counts account for the 1K quarantine",
        "native_conversion": "published SystemOne system_one_training_rows; original state/instructions/candidate order; one-hot source gold; original source/group IDs",
        "score_null_description_fallback_rows": len(fallback_ids),
        "full_input_limit": 1024,
        "truncation": "error",
        "quarantine_whole_group": True,
        "audit": audit,
        "quarantine_source_counts": dict(
            sorted(Counter(q["source_id"] for q in quarantine).items())
        ),
        "outputs": {
            name: {
                "sha256": sha(stage / name),
                "rows": sum(1 for _ in (stage / name).open()),
            }
            for name in ("train.jsonl", "select.jsonl", "quarantine.jsonl")
        },
        "limitations": [
            "Selection mixes synthetic oracle and projected GoEmotions labels; it is not a blind transfer panel.",
            "Complete-input 1K admission can quarantine source groups; see per-row private quarantine receipt.",
            "The Kai base model and inherited tokenizer retain their own source conditions; model release requires separate review.",
            "CAL700 is retained only as a parent split isolation receipt; the Kai native trainer does not use CAL or temperature fitting.",
        ],
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    os.replace(stage, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.parent, args.model, args.output)
    print(
        json.dumps(
            {"audit": result["audit"], "outputs": result["outputs"]},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
