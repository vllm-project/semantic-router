"""Fail-closed CPU audit of the pinned 27B pooled schedule candidate.

The schedule is input-only. TRAIN labels are opened only for the separately
pinned teacher-mask check. No SELECT/CAL or evaluation answer keys are read.
The private receipt contains aggregate counts and hashes, never row content.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from training.data.audit_27b_full_input_overlap import (
    audit_core_full_input,
    full_input_overlap_rows,
)
from training.data.audit_27b_quota_envelope import load_gold_free_train, token_count
from training.data.audit_27b_teacher_admission import (
    CHOICE_TEACHER_SHA256,
    RIGHTS_SHA256,
    SCORE_TEACHER_SHA256,
    SOURCE_REVISION,
    TRAIN_SHA256,
    AdmissionHold,
    _check,
    teacher_mask,
    verify_rights,
    verify_teacher,
    write_once,
)
from training.data.plan_27b_pooled_schedule import SCHEMA as SCHEDULE_SCHEMA
from training.data.plan_goldfree_inventory import (
    CORE_ROLES,
    project_partition_row,
    validate_core_rows,
)
from training.model.data import canonical, digest, file_sha256, load_partition

SCHEMA = "decision2-27b-pooled-admission/1"
CORE_SCHEMA = "decision2-projected-core-inputs-v1"
SCHEDULE_SHA256 = "d76eb49a12a6dd04861239a7d7e6697f6f96c0f80015246b4060362c5bc156d7"
CORE_MANIFEST_SHA256 = (
    "26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1"
)
EXPECTED_SCHEDULE_DIGEST = (
    "6ea5e54d207535bd1ca06e9f2ad4ff19fbdae0d4c6143fce0b20be8fcbe52181"
)
INPUT_KEYS = frozenset({"id", "state", "instructions", "options"})
RIGHTS_SOURCE_MAP = {
    "google_goemotions_official_train": "GoEmotions official TRAIN",
    "legacy:cosmos_qa": "CosmosQA",
    "legacy:snli": "SNLI",
    "legacy:squad2_answerability": "SQuAD 2.0",
    "css_flute_official_train": "FLUTE",
    "legacy:stage3_replay": "Stage3 replay",
    "decision2_programmatic_original_v1": "original_programmatic",
    "decision2_targeted_programmatic_v1": "original_programmatic",
    "legacy:stage4-general-composition-v2": "original_programmatic",
}


def selected_rows(
    schedule: dict[str, Any], train: list[dict[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    """Bind every selected ID and complete input hash to the pinned TRAIN row."""
    _check(schedule.get("schema") == SCHEDULE_SCHEMA, "SCHEDULE_SCHEMA")
    _check(schedule.get("status") == "SCHEDULE_CANDIDATE_ONLY", "SCHEDULE_STATUS")
    _check(schedule.get("source_revision") == SOURCE_REVISION, "SCHEDULE_SOURCE")
    _check(
        schedule.get("rows") == 2560 and schedule.get("updates") == 160,
        "SCHEDULE_BUDGET",
    )
    _check(schedule.get("max_length") == 4096, "SCHEDULE_CONTEXT")
    entries = schedule.get("schedule")
    _check(isinstance(entries, list) and len(entries) == 2560, "SCHEDULE_ROWS")
    _check(
        schedule.get("schedule_sha256") == EXPECTED_SCHEDULE_DIGEST == digest(entries),
        "SCHEDULE_DIGEST",
    )
    by_id = {row["id"]: row for row in train}
    _check(len(by_id) == len(train) == 7455, "TRAIN_IDENTITY")
    chosen = []
    seen = set()
    for item in entries:
        _check(
            isinstance(item, dict) and set(item) == {"id", "input_sha256", "tokens"},
            "SCHEDULE_ENTRY",
        )
        identifier = item["id"]
        _check(
            isinstance(identifier, str)
            and identifier not in seen
            and identifier in by_id,
            "SCHEDULE_ID",
        )
        row = by_id[identifier]
        _check(item["input_sha256"] == row["input_sha256"], "SCHEDULE_INPUT_HASH")
        _check(
            type(item["tokens"]) is int and 0 < item["tokens"] <= 4096,
            "SCHEDULE_TOKEN_BUDGET",
        )
        _check(item["tokens"] == token_count(row, tokenizer), "SCHEDULE_TOKEN_IDENTITY")
        seen.add(identifier)
        chosen.append(row)
    _check(
        dict(sorted(collections.Counter(row["task_type"] for row in chosen).items()))
        == schedule.get("by_type"),
        "SCHEDULE_TYPE_COUNTS",
    )
    _check(
        dict(sorted(collections.Counter(row["language"] for row in chosen).items()))
        == schedule.get("by_language"),
        "SCHEDULE_LANGUAGE_COUNTS",
    )
    _check(
        sum(item["tokens"] for item in entries) == schedule.get("raw_token_exposure"),
        "SCHEDULE_RAW_TOKENS",
    )
    _check(
        sum((item["tokens"] + 7) // 8 * 8 for item in entries)
        == schedule.get("padded_token_exposure"),
        "SCHEDULE_PADDED_TOKENS",
    )
    return chosen


def group_integrity(
    train: list[dict[str, Any]], chosen: list[dict[str, Any]]
) -> dict[str, int]:
    """Distinguish missing task types from missing *rows* in a source group."""
    all_groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    selected_groups: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in train:
        all_groups[(row["source"], row["group_id"])].append(row)
    for row in chosen:
        selected_groups[(row["source"], row["group_id"])].append(row)
    missing_type_groups = 0
    missing_row_groups = 0
    missing_same_type_rows = 0
    score_groups_missing_cross_type = 0
    score_cross_type_rows_needed = 0
    for group, selected in selected_groups.items():
        original = all_groups[group]
        selected_ids = {row["id"] for row in selected}
        all_types = {row["task_type"] for row in original}
        selected_types = {row["task_type"] for row in selected}
        missing_type_groups += selected_types != all_types
        missing_row_groups += len(selected) != len(original)
        missing_same_type_rows += sum(
            row["id"] not in selected_ids and row["task_type"] in selected_types
            for row in original
        )
        if "score" in selected_types:
            cross_type_missing = sum(
                row["id"] not in selected_ids and row["task_type"] != "score"
                for row in original
            )
            score_groups_missing_cross_type += bool(cross_type_missing)
            score_cross_type_rows_needed += cross_type_missing
    return {
        "selected_source_groups": len(selected_groups),
        "missing_task_type_groups": missing_type_groups,
        "missing_any_row_groups": missing_row_groups,
        "missing_same_type_rows": missing_same_type_rows,
        "score_groups_missing_cross_type_rows": score_groups_missing_cross_type,
        "score_cross_type_rows_needed": score_cross_type_rows_needed,
    }


def selected_rights(
    manifest: dict[str, Any],
    train: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
    *,
    source_map: dict[str, str] = RIGHTS_SOURCE_MAP,
) -> dict[str, Any]:
    """Match each raw source to the frozen ledger's coarser attribution."""
    declared = verify_rights(manifest, train)
    entries = [
        entry
        for entry in manifest["source_rights"]
        if entry.get("partition_scope", "TRAIN") == "TRAIN"
    ]
    ledger = {entry["source"]: entry for entry in entries}
    _check(len(ledger) == len(entries), "RIGHTS_DUPLICATE_SOURCE")
    train_sources = {row["source"] for row in train}
    _check(train_sources == set(source_map), "RIGHTS_SOURCE_MAPPING")
    projected = collections.Counter(source_map[row["source"]] for row in train)
    _check(
        {source: entry["rows"] for source, entry in ledger.items()} == dict(projected),
        "RIGHTS_PER_SOURCE_COUNT",
    )
    selected_sources = {row["source"] for row in chosen}
    _check(
        {source_map[source] for source in selected_sources} <= set(ledger),
        "RIGHTS_SELECTED_SOURCE",
    )
    return {**declared, "selected_source_count": len(selected_sources)}


def load_core_inventory(
    directory: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    manifest_path = directory / "manifest.json"
    _check(
        manifest_path.is_file() and file_sha256(manifest_path) == CORE_MANIFEST_SHA256,
        "CORE_MANIFEST_HASH",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("roles")
    _check(
        manifest.get("schema") == CORE_SCHEMA and isinstance(entries, dict),
        "CORE_SCHEMA",
    )
    _check(set(entries) == CORE_ROLES, "CORE_ROLES")
    roles = {}
    hashes = {}
    for role, record in sorted(entries.items()):
        _check(
            isinstance(record, dict) and record.get("path") == f"{role}.jsonl",
            "CORE_ROLE_PATH",
        )
        path = directory / record["path"]
        _check(
            path.is_file() and file_sha256(path) == record.get("sha256"),
            "CORE_ROLE_HASH",
        )
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                _check(
                    isinstance(row, dict) and set(row) <= INPUT_KEYS,
                    "CORE_INPUT_SCHEMA",
                )
                rows.append(row)
        roles[role] = rows
        hashes[role] = record["sha256"]
    validate_core_rows(roles)
    return roles, hashes


def distinctive_overlap(
    selected: list[dict[str, Any]], roles: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Require exact/near comparison of whole native inputs and state evidence.

    The broad legacy span screen matches repeated boilerplate instructions and
    option descriptions across unrelated rows. It remains visible in the
    receipt, but complete input and state comparisons determine this bounded
    collision gate. Neither comparison proves semantic non-overlap.
    """
    left = [project_partition_row(row, "train") for row in selected]

    def complete(row: dict[str, Any]) -> dict[str, str]:
        fields = {
            name: row[name]
            for name in ("state", "instructions", "options")
            if name in row
        }
        # Prefix prevents the reusable span checker from extracting shared
        # JSON leaves and counting scaffold options as full-input collisions.
        return {"id": row["id"], "state": "NATIVE INPUT\n" + canonical(fields)}

    def state(row: dict[str, Any]) -> dict[str, str]:
        return {"id": row["id"], "state": "STATE EVIDENCE\n" + row["state"]}

    left_complete = [complete(row) for row in left]
    left_state = [state(row) for row in left]
    by_role = {}
    for role, protected in sorted(roles.items()):
        if role == "rights_clean_train":
            continue
        by_role[role] = {
            "complete_input": full_input_overlap_rows(
                left_complete,
                [complete(row) for row in protected],
            )["counts"],
            "state_evidence": full_input_overlap_rows(
                left_state,
                [state(row) for row in protected],
            )["counts"],
        }
    blocked = any(
        any(
            surfaces["complete_input"][name]
            for name in ("same_row_ids", "exact_raw", "exact_normalized")
        )
        or any(surfaces["state_evidence"].values())
        for surfaces in by_role.values()
    )
    return {
        "status": (
            "HOLD_DISTINCTIVE_INPUT_OVERLAP"
            if blocked
            else "PASS_BOUNDED_DISTINCTIVE_INPUT_SCREEN"
        ),
        "by_role": by_role,
        "limitation": "Exact complete-input and exact/near state-evidence collisions block. Near complete-input matches are diagnostic only because shared instructions/options dominate them; common instruction/option spans are reported separately. Changed evidence and paraphrase, especially with generic state, and source/semantic leakage remain unresolved.",
    }


def audit(
    *,
    schedule_path: Path,
    train_path: Path,
    rights_path: Path,
    choice_teacher_path: Path,
    score_teacher_path: Path,
    core_directory: Path,
    tokenizer: Any,
) -> dict[str, Any]:
    pinned = {
        "schedule": (schedule_path, SCHEDULE_SHA256),
        "train": (train_path, TRAIN_SHA256),
        "rights": (rights_path, RIGHTS_SHA256),
        "choice_teacher": (choice_teacher_path, CHOICE_TEACHER_SHA256),
        "score_teacher": (score_teacher_path, SCORE_TEACHER_SHA256),
    }
    for role, (path, expected) in pinned.items():
        _check(
            path.is_file() and file_sha256(path) == expected,
            f"FROZEN_{role.upper()}_HASH",
        )
    train_input_only = load_gold_free_train(train_path)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    selected_input_only = selected_rows(schedule, train_input_only, tokenizer)
    groups = group_integrity(train_input_only, selected_input_only)
    train = load_partition(train_path, "train")
    _check(
        len(train) == len(train_input_only)
        and all(
            a["id"] == b["id"] and a["input_sha256"] == b["input_sha256"]
            for a, b in zip(train, train_input_only, strict=True)
        ),
        "TRAIN_LABEL_INPUT_IDENTITY",
    )
    selected = [dict(row) for row in selected_input_only]
    full_by_id = {row["id"]: row for row in train}
    for row in selected:
        row.update(full_by_id[row["id"]])
    rights = selected_rights(
        json.loads(rights_path.read_text(encoding="utf-8")), train, selected
    )
    choice_rows = [row for row in train if row["task_type"] != "score"]
    score_rows = [row for row in train if row["task_type"] == "score"]
    _check(
        len(choice_rows) == 6939 and len(score_rows) == 516, "TEACHER_TRAIN_TYPE_COUNTS"
    )
    choice_vectors = verify_teacher(
        json.loads(choice_teacher_path.read_text(encoding="utf-8")),
        choice_rows,
        task="choice_noul",
    )
    score_vectors = verify_teacher(
        json.loads(score_teacher_path.read_text(encoding="utf-8")),
        score_rows,
        task="score",
    )
    vectors = choice_vectors | score_vectors
    _check(len(vectors) == len(train), "TEACHER_VECTOR_COVERAGE")
    mask_result: dict[str, Any]
    try:
        mask_result = {
            "status": "PASS_FROZEN_MINIMA",
            **teacher_mask(selected, vectors),
        }
    except AdmissionHold as error:
        mask_result = {"status": "HOLD", "reason_code": str(error)}
    core_roles, core_hashes = load_core_inventory(core_directory)
    overlap = audit_core_full_input(selected, core_roles)
    distinctive = distinctive_overlap(selected, core_roles)
    blockers = []
    if groups["missing_any_row_groups"]:
        blockers.append("CROSS_TYPE_OR_ROW_GROUP_INCOMPLETE")
    if mask_result["status"] != "PASS_FROZEN_MINIMA":
        blockers.append(mask_result["reason_code"])
    if overlap["status"] in {
        "HOLD_MISSING_PARTITION_ROLES",
        "HOLD_MISSING_CORE_ROLES",
        "HOLD_UNATTESTED_OPTIONAL_ROLES",
        "HOLD_CORE_SCHEMA_OR_SCHEDULE",
        "HOLD_SCHEDULE_INPUT_IDENTITY",
    }:
        blockers.append(overlap["status"])
    if distinctive["status"] != "PASS_BOUNDED_DISTINCTIVE_INPUT_SCREEN":
        blockers.append(distinctive["status"])
    return {
        "schema": SCHEMA,
        "status": "HOLD" if blockers else "PASS_BOUNDED_CPU_ADMISSION",
        "blockers": blockers,
        "source_revision": SOURCE_REVISION,
        "input_sha256": {role: expected for role, (_, expected) in pinned.items()},
        "core_manifest_sha256": CORE_MANIFEST_SHA256,
        "core_role_sha256": core_hashes,
        "selected_rows": len(selected),
        "by_type": schedule["by_type"],
        "group_integrity": groups,
        "rights": rights,
        "teacher_mask": mask_result,
        "broad_input_span_overlap": overlap,
        "distinctive_input_overlap": distinctive,
        "gpu_hours": 0,
        "limits": "Bounded CPU checks only. Whole source/group integrity is required for this admission; the source/type-only planner does not establish it. A clean lexical overlap screen cannot establish semantic/source isolation, third-party pretraining exposure, or legal permission. No model parity, optimization, SELECT, CAL or formal/public inference was run.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "schedule",
        "train",
        "rights",
        "choice_teacher",
        "score_teacher",
        "core_directory",
        "tokenizer_dir",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, local_files_only=True
        )
        result = audit(
            schedule_path=args.schedule,
            train_path=args.train,
            rights_path=args.rights,
            choice_teacher_path=args.choice_teacher,
            score_teacher_path=args.score_teacher,
            core_directory=args.core_directory,
            tokenizer=tokenizer,
        )
    except (
        AdmissionHold,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
    ) as error:
        result = {
            "schema": SCHEMA,
            "status": "HOLD",
            "reason_code": (
                str(error)
                if isinstance(error, AdmissionHold)
                else f"PRIVATE_INPUT_OR_RUNTIME_{type(error).__name__.upper()}"
            ),
            "gpu_hours": 0,
        }
    write_once(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "reason_code": result.get("reason_code"),
                "receipt_sha256": file_sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
