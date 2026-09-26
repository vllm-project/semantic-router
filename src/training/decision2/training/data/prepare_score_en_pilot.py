"""Freeze the English-only Score v6 matched arms without model inference.

This CPU-only preparer never opens the r2 answer key or runs a model. The
English-only experiment remains blocked until the separate source
materialization/parity receipt and run gate in the signed protocol pass.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from training.model.data import (
    check_partition_isolation,
    file_sha256,
    load_partition,
)

PARENT_SHA = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
V6_SHA = "6aee966cc5499a87d2a77241676586c9c9f801b3c662a078daf025f001169f54"
R2_PACKET_SHA = "091e3023b84d64131a72b23b90b3eacf837027ed23d58045de801e92a331f683"
SOURCE_MODEL_SHA = "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PROTOCOL = "score-v6-en-matched-pilot-prereg-2026-09-27.md"
FAMILY_GROUPS = {
    "score_evidence_intersection": 60,
    "score_obligation_review": 61,
    "score_route_depth": 61,
    "score_timely_streak": 61,
}
R2_OPERATIONS = {
    "allocation_caps",
    "inclusive_coverage",
    "independent_quorum",
    "waiver_precedence",
}
SEED = "decision2-score-v6-en-matched-pilot-20260927"


def _hash_order(row: dict[str, Any], purpose: str) -> tuple[str, str]:
    value = f"{SEED}\0{purpose}\0{row['id']}".encode()
    return hashlib.sha256(value).hexdigest(), row["id"]


def _require_sha(path: Path, expected: str, role: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{role} SHA-256 mismatch: {actual}")


def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    return groups


def validate_v6_english(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 969 or collections.Counter(r["language"] for r in rows) != {
        "en": 729,
        "zh": 240,
    }:
        raise ValueError("The frozen v6 candidate count or language balance changed")
    english = [row for row in rows if row["language"] == "en"]
    groups = _groups(english)
    family_groups = collections.Counter(items[0]["family"] for items in groups.values())
    if len(groups) != 243 or family_groups != FAMILY_GROUPS:
        raise ValueError("English v6 family/group balance changed")
    for items in groups.values():
        if (
            len(items) != 3
            or {row["label"] for row in items} != {0, 1, 2}
            or {row["family"] for row in items} != {items[0]["family"]}
            or any(
                row["task_type"] != "score" or len(row["options"]) != 3 for row in items
            )
        ):
            raise ValueError("English v6 has an incomplete or inconsistent triplet")
    return english


def validate_r2_english(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 240 or collections.Counter(r.get("language") for r in rows) != {
        "en": 192,
        "zh": 48,
    }:
        raise ValueError("The frozen r2 packet count or language balance changed")
    if any("label" in row or "target" in row or "answer" in row for row in rows):
        raise ValueError("The r2 review packet unexpectedly contains answer fields")
    english = [row for row in rows if row["language"] == "en"]
    seen_aliases = set()
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in english:
        alias = row.get("review_id")
        if not isinstance(alias, str) or not alias or alias in seen_aliases:
            raise ValueError("r2 review aliases are missing or repeated")
        if (
            not isinstance(row.get("options"), list)
            or any(not isinstance(option, dict) for option in row["options"])
            or {option.get("key") for option in row["options"]} != {"0", "1", "2"}
        ):
            raise ValueError("r2 review row needs native Score options 0/1/2")
        seen_aliases.add(alias)
        groups[row["group_id"]].append(row)
    operation_groups = collections.Counter(
        items[0]["operation"] for items in groups.values()
    )
    if len(groups) != 64 or operation_groups != dict.fromkeys(R2_OPERATIONS, 16):
        raise ValueError("English r2 operation/group balance changed")
    if any(
        len(items) != 3
        or any(item["operation"] != items[0]["operation"] for item in items)
        for items in groups.values()
    ):
        raise ValueError("English r2 packet has an incomplete operation group")
    return english


def r2_native_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert opaque gold-free review rows to the existing native adapter."""
    return [
        {
            "id": row["review_id"],
            "state": row["state"],
            "questions": {
                "decision": {
                    "type": "score",
                    "instructions": row["instructions"],
                    "criteria": [
                        next(
                            option["description"]
                            for option in row["options"]
                            if option["key"] == str(level)
                        )
                        for level in range(3)
                    ],
                }
            },
        }
        for row in rows
    ]


def _length(row: dict[str, Any], tokenizer: Any) -> int:
    # Use the exact native segmented encoding, not a single concatenated encode.
    from training.model.decision_model import encode

    return len(encode(row, tokenizer, 1 << 20)["ids"])


def choose_control(
    candidates: list[tuple[dict[str, Any], int]],
    *,
    count: int,
    target_tokens: int,
    minimum_score: int,
) -> list[tuple[dict[str, Any], int]]:
    """Find distinct parent rows with exact count/tokens and a Score floor."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    pool = sorted(
        (
            (row, length)
            for row, length in candidates
            if row["language"] == "en" and 200 <= length <= 512
        ),
        key=lambda item: _hash_order(item[0], "control"),
    )
    if (
        len(pool) < count
        or sum(row["task_type"] == "score" for row, _ in pool) < minimum_score
    ):
        raise ValueError("Insufficient eligible parent control rows")
    matrix = np.stack(
        (
            np.ones(len(pool), dtype=np.float64),
            np.array([length for _, length in pool], dtype=np.float64),
            np.array(
                [row["task_type"] == "score" for row, _ in pool], dtype=np.float64
            ),
        )
    )
    result = milp(
        c=np.zeros(len(pool)),
        integrality=np.ones(len(pool)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            matrix,
            [count, target_tokens, minimum_score],
            [count, target_tokens, np.inf],
        ),
        options={"time_limit": 30},
    )
    if not result.success or result.x is None:
        raise ValueError(
            f"No exact parent-only matched control: solver status {result.status}"
        )
    selected = [item for item, flag in zip(pool, result.x) if round(flag) == 1]
    if (
        len(selected) != count
        or sum(length for _, length in selected) != target_tokens
        or sum(row["task_type"] == "score" for row, _ in selected) < minimum_score
        or len({row["id"] for row, _ in selected}) != count
    ):
        raise AssertionError("Integer control solution failed exact postcheck")
    return selected


def choose_replay(
    candidates: list[tuple[dict[str, Any], int]],
    control_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], int]]:
    blocked_ids = {row["id"] for row in control_rows}
    blocked_groups = {row["group_id"] for row in control_rows}
    replay = []
    for task_type in ("choice", "noul"):
        pool = sorted(
            (
                (row, length)
                for row, length in candidates
                if row["language"] == "en"
                and row["task_type"] == task_type
                and length <= 192
                and row["id"] not in blocked_ids
                and row["group_id"] not in blocked_groups
            ),
            key=lambda item: _hash_order(item[0], f"replay-{task_type}"),
        )
        if len(pool) < 1024:
            raise ValueError(f"Fewer than 1,024 group-disjoint {task_type} replay rows")
        replay.extend(pool[:1024])
    if len({row["id"] for row, _ in replay}) != 2048:
        raise AssertionError("Replay IDs must be unique")
    return replay


def _audit_isolation(
    v6: list[dict[str, Any]],
    parent_train: list[dict[str, Any]],
    select: list[dict[str, Any]],
    cal: list[dict[str, Any]],
    r2: list[dict[str, Any]],
) -> dict[str, Any]:
    from training.data import build_pilot as pilot
    from training.data import build_targeted_candidate as targeted

    check_partition_isolation(
        {"train": v6 + parent_train, "select": select, "cal": cal}
    )
    for field in ("id", "group_id", "input_sha256"):
        if {row[field] for row in v6} & {row[field] for row in parent_train}:
            raise ValueError(f"v6 and parent TRAIN share {field}")
    review = [{**row, "id": row["review_id"], "task_type": "score"} for row in r2]
    current = v6 + parent_train
    exact = {pilot.input_sha256(row) for row in current} & {
        pilot.input_sha256(row) for row in review
    }
    current_states = {targeted.text_hashes(row["state"]) for row in current}
    review_states = {targeted.text_hashes(row["state"]) for row in review}
    near_state = pilot.near_duplicates(
        targeted.context_rows(current), targeted.context_rows(review)
    )
    near_full = pilot.near_duplicates(current, review)
    if (
        exact
        or current_states & review_states
        or near_state["count"]
        or near_full["count"]
    ):
        raise ValueError("A train row overlaps the gold-free r2 English packet")
    return {
        "exact_full_prompt": 0,
        "exact_state": 0,
        "bounded_near_state": 0,
        "bounded_near_full_prompt": 0,
        "limit": "Approximate near matching cannot establish semantic independence",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from importlib.metadata import version

    from transformers import AutoTokenizer

    from training.model.infer import checkpoint_fingerprint

    paths = {
        "train": args.parent_train,
        "select": args.parent_select,
        "cal": args.parent_cal,
    }
    for role, path in paths.items():
        _require_sha(path, PARENT_SHA[role], f"parent {role}")
    _require_sha(args.v6_train, V6_SHA, "v6 TRAIN")
    _require_sha(args.r2_packet, R2_PACKET_SHA, "gold-free r2 packet")
    source_run = args.source_checkpoint.parent
    if (
        json.loads((source_run / "BEST.json").read_text())["checkpoint"]
        != args.source_checkpoint.name
    ):
        raise ValueError("The source checkpoint is not frozen BEST368")
    complete = json.loads((source_run / "COMPLETE.json").read_text())
    if (
        complete.get("status") != "complete"
        or complete.get("step") != 458
        or complete.get("planned_updates") != 458
        or complete.get("best") != args.source_checkpoint.name
    ):
        raise ValueError("The original source run is incomplete")
    metadata = json.loads((args.source_checkpoint / "decision_config.json").read_text())
    if (
        args.source_checkpoint.name != "checkpoint-0000368"
        or metadata.get("checkpoint_format") != "peft-lora/1"
        or metadata.get("base_revision") != SOURCE_REVISION
        or metadata.get("head_dim") != 256
        or metadata.get("lora", {}).get("rank") != 8
        or metadata.get("lora", {}).get("alpha") != 16
        or metadata.get("lora", {}).get("dropout") != 0.05
    ):
        raise ValueError("The source checkpoint is not the frozen 27B LoRA")
    identity = checkpoint_fingerprint(args.source_checkpoint, args.source_path)
    if identity["model_sha256"] != SOURCE_MODEL_SHA:
        raise ValueError("The source inference fingerprint changed")

    parent_train = load_partition(args.parent_train, "train")
    parent_select = load_partition(args.parent_select, "select")
    parent_cal = load_partition(args.parent_cal, "cal")
    v6 = validate_v6_english(load_partition(args.v6_train, "train"))
    packet = [
        json.loads(line)
        for line in args.r2_packet.read_text(encoding="utf-8").splitlines()
    ]
    r2 = validate_r2_english(packet)
    parent_en = [row for row in parent_train if row["language"] == "en"]
    select_en = [row for row in parent_select if row["language"] == "en"]
    cal_en = [row for row in parent_cal if row["language"] == "en"]
    if len(parent_en) != 6085 or len(select_en) != 588:
        raise ValueError("Parent English TRAIN or SELECT count changed")
    tokenizer = AutoTokenizer.from_pretrained(args.source_path, local_files_only=True)
    v6_lengths = [(row, _length(row, tokenizer)) for row in v6]
    target_tokens = sum(length for _, length in v6_lengths)
    if target_tokens != 220_932:
        raise ValueError("Frozen v6 English native token budget changed")
    parent_lengths = [
        (row, _length(row, tokenizer))
        for row in parent_en
        if row["task_type"] in ("choice", "noul", "score")
    ]
    control = choose_control(
        parent_lengths, count=729, target_tokens=target_tokens, minimum_score=200
    )
    replay = choose_replay(parent_lengths, [row for row, _ in control])
    arm_a = [row for row, _ in v6_lengths] + [row for row, _ in replay]
    arm_b = [row for row, _ in control] + [row for row, _ in replay]
    if any(length > 1024 for _, length in (*v6_lengths, *control, *replay)):
        raise ValueError("A TRAIN row exceeds the 1,024-token cap")
    if any(_length(row, tokenizer) > 1024 for row in select_en):
        raise ValueError("A parent English SELECT row exceeds 1,024 tokens")
    if any(
        _length(
            {
                **row,
                "id": row["review_id"],
                "family": row["operation"],
                "task_type": "score",
                "label": 0,  # Dummy encoding field; no r2 gold is opened.
            },
            tokenizer,
        )
        > 1024
        for row in r2
    ):
        raise ValueError("An r2 English packet prompt exceeds 1,024 tokens")
    isolation = _audit_isolation(v6, parent_en, select_en, cal_en, r2)
    check_partition_isolation({"train": arm_a, "select": select_en, "cal": cal_en})
    check_partition_isolation({"train": arm_b, "select": select_en, "cal": cal_en})
    if len(arm_a) != 2777 or len(arm_b) != 2777:
        raise AssertionError("Matched arms must have 2,777 rows")
    score_padded = sum(math.ceil(length / 8) * 8 for _, length in v6_lengths)
    control_padded = sum(math.ceil(length / 8) * 8 for _, length in control)
    padded_delta = abs(score_padded - control_padded) / score_padded
    if padded_delta > 0.05:
        raise ValueError("Arm padding workload differs by more than 5%")
    if args.output.exists():
        raise FileExistsError("Private pilot output already exists; never overwrite it")
    pending = args.output.with_name(args.output.name + ".pending")
    if pending.exists():
        raise FileExistsError("An interrupted private pilot output already exists")
    pending.mkdir(parents=True)
    files = {
        "arm_a_train": ("arm_a.train.jsonl", arm_a),
        "arm_b_train": ("arm_b.train.jsonl", arm_b),
        "parent_select_en": ("parent_en.select.jsonl", select_en),
        "parent_cal_en": ("parent_en.cal.jsonl", cal_en),
        "r2_en_gold_free": ("r2_en.goldfree.jsonl", r2),
        "r2_en_native_prompts": (
            "r2_en.native-prompts.jsonl",
            r2_native_prompts(r2),
        ),
    }
    for name, (filename, rows) in files.items():
        _write_jsonl(pending / filename, rows)
    private_manifest = {
        "schema_version": "decision2-score-v6-en-matched-arms/1",
        "status": "DATA_PREPARED_OPTIMIZER_BLOCKED_ON_SOURCE_PARITY",
        "protocol_sha256": file_sha256(
            Path(__file__).parents[2] / "research" / PROTOCOL
        ),
        "preparer_sha256": file_sha256(Path(__file__)),
        "source_model_sha256": identity["model_sha256"],
        "source_revision": SOURCE_REVISION,
        "input_sha256": {**PARENT_SHA, "v6": V6_SHA, "r2_packet": R2_PACKET_SHA},
        "output_sha256": {
            name: file_sha256(pending / filename)
            for name, (filename, _) in files.items()
        },
        "seed": SEED,
        "scipy_version": version("scipy"),
        "transformers_version": version("transformers"),
        "english_v6_groups": 243,
        "english_r2_groups": 64,
        "replay_rows": 2048,
        "control_score_rows": sum(row["task_type"] == "score" for row, _ in control),
        "arm_rows": 2777,
        "v6_native_tokens": target_tokens,
        "control_native_tokens": sum(length for _, length in control),
        "v6_padded_tokens": score_padded,
        "control_padded_tokens": control_padded,
        "padding_relative_delta": padded_delta,
        "isolation": isolation,
        "warning": "No model run or release result. Chinese subsets remain held.",
    }
    (pending / "manifest.json").write_text(
        json.dumps(private_manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(pending, args.output)
    return {
        "status": private_manifest["status"],
        "arm_rows": 2777,
        "v6_native_tokens": target_tokens,
        "control_native_tokens": private_manifest["control_native_tokens"],
        "control_score_rows": private_manifest["control_score_rows"],
        "manifest_sha256": file_sha256(args.output / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--parent-train", type=Path, required=True)
    parser.add_argument("--parent-select", type=Path, required=True)
    parser.add_argument("--parent-cal", type=Path, required=True)
    parser.add_argument("--v6-train", type=Path, required=True)
    parser.add_argument("--r2-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), sort_keys=True))


if __name__ == "__main__":
    main()
