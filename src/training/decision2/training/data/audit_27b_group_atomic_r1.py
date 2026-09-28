"""Independent CPU admission of the locked group-atomic 27B r1 witness.

Only TRAIN labels enter the teacher-mask check. Protected roles are the pinned
gold-free input projections. Exact private pair IDs are saved mode 0600;
stdout contains only status and receipt digest. This can hold a candidate but
cannot prove absence of semantic paraphrase or authorize GPU work.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from training.data.audit_27b_group_atomic_capacity import state_near_ids
from training.data.audit_27b_pooled_admission import (
    CORE_MANIFEST_SHA256,
    group_integrity,
    load_core_inventory,
    selected_rights,
)
from training.data.audit_27b_quota_envelope import load_gold_free_train, token_count
from training.data.audit_27b_teacher_admission import (
    CAL_SHA256,
    CHOICE_TEACHER_SHA256,
    RIGHTS_SHA256,
    SCORE_TEACHER_SHA256,
    SELECT_SHA256,
    TRAIN_SHA256,
    AdmissionHold,
    _check,
    teacher_mask,
    verify_teacher,
    write_once,
)
from training.data.build_pilot import _simhash
from training.data.plan_goldfree_inventory import project_partition_row
from training.model.data import canonical, digest, file_sha256, load_partition

SCHEMA = "decision2-27b-group-atomic-r1-admission/1"
CAPACITY_SHA256 = "809912b45c75aa7b630fd8fb40cafe898dc55a7201cc47cb9084484b0abb134d"
LOCKED_SCHEDULE_SHA256 = (
    "0743da3f1ed18cc8eb824c41c736d92b23fb3d8a738e1b3401ff830229424be5"
)
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
MASK_SHA256 = "778eb599aa5f442a62d241394c65fd3b86e5a4aa4aad9c200bd360da86c0ab82"
EXPECTED_TYPES = {"choice": 1201, "noul": 940, "score": 419}
EXPECTED_RAW_TOKENS = 1_244_573
EXPECTED_PADDED_TOKENS = 1_253_880
EXPECTED_ROWS = 2560
EXPECTED_UPDATES = 160


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _full_input(row: dict[str, Any]) -> str:
    """One canonical string containing every projected native input field."""
    return canonical(
        {
            field: row[field]
            for field in ("state", "instructions", "options")
            if field in row
        }
    )


def surface_pair_ids(
    selected: list[dict[str, Any]],
    protected: list[dict[str, Any]],
    *,
    surface: str,
) -> dict[str, list[tuple[str, str]]]:
    """Independently enumerate exact and bounded-near pair IDs for one surface."""
    if surface not in {"complete_input", "state_evidence"}:
        raise ValueError("Unknown native input surface")
    view = _full_input if surface == "complete_input" else lambda row: row["state"]
    raw_index: dict[str, set[str]] = collections.defaultdict(set)
    normalized_index: dict[str, set[str]] = collections.defaultdict(set)
    near_right: list[tuple[str, str, int]] = []
    bands: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    for row in protected:
        value = view(row)
        norm = _normalized(value)
        raw_index[hashlib.sha256(value.encode()).hexdigest()].add(row["id"])
        normalized_index[hashlib.sha256(norm.encode()).hexdigest()].add(row["id"])
        if len(norm) < 40:
            continue
        bits = _simhash(norm)
        index = len(near_right)
        near_right.append((row["id"], norm, bits))
        for band in range(8):
            bands[(band, (bits >> (band * 8)) & 255)].add(index)
    right_ids = {row["id"] for row in protected}
    pairs: dict[str, set[tuple[str, str]]] = {
        name: set()
        for name in ("same_row_ids", "exact_raw", "exact_normalized", "near")
    }
    for row in selected:
        identifier = row["id"]
        if identifier in right_ids:
            pairs["same_row_ids"].add((identifier, identifier))
        value = view(row)
        norm = _normalized(value)
        pairs["exact_raw"].update(
            (identifier, other)
            for other in raw_index.get(hashlib.sha256(value.encode()).hexdigest(), ())
        )
        pairs["exact_normalized"].update(
            (identifier, other)
            for other in normalized_index.get(
                hashlib.sha256(norm.encode()).hexdigest(), ()
            )
        )
        if len(norm) < 40:
            continue
        bits = _simhash(norm)
        candidates = set()
        for band in range(8):
            candidates.update(bands.get((band, (bits >> (band * 8)) & 255), ()))
        for index in candidates:
            other_id, other, other_bits = near_right[index]
            if abs(len(norm) - len(other)) > 0.08 * max(len(norm), len(other)):
                continue
            if (bits ^ other_bits).bit_count() > 8:
                continue
            if difflib.SequenceMatcher(None, norm, other).ratio() >= 0.94:
                pairs["near"].add((identifier, other_id))
    return {name: sorted(matches) for name, matches in pairs.items()}


def source_group_pairs(
    selected: list[dict[str, Any]],
    partitions: dict[str, list[dict[str, Any]]],
    *,
    match_source: bool = True,
) -> dict[str, list[tuple[str, str]]]:
    """Check both composite and source-agnostic original group identity."""
    selected_groups: dict[tuple[str, str] | str, set[str]] = collections.defaultdict(
        set
    )
    for row in selected:
        key = (row["source"], row["group_id"]) if match_source else row["group_id"]
        selected_groups[key].add(row["id"])
    return {
        role: sorted(
            (selected_id, row["id"])
            for row in rows
            for selected_id in selected_groups.get(
                (row["source"], row["group_id"]) if match_source else row["group_id"],
                (),
            )
        )
        for role, rows in partitions.items()
    }


def near_pair_profile(
    pair_ids: list[tuple[str, str]],
    selected: dict[str, dict[str, Any]],
    protected_metadata: dict[str, dict[str, Any]],
    protected_inputs: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Aggregate same-family and state-similarity signals without targets."""
    result = collections.Counter()
    for left_id, right_id in pair_ids:
        left = selected[left_id]
        metadata = protected_metadata[right_id]
        right = protected_inputs[right_id]
        first = _normalized(left["state"])
        second = _normalized(right["state"])
        first_terms = set(re.findall(r"\w+", first))
        second_terms = set(re.findall(r"\w+", second))
        jaccard = (
            len(first_terms & second_terms) / len(first_terms | second_terms)
            if first_terms | second_terms
            else 0.0
        )
        edit_ratio = difflib.SequenceMatcher(None, first, second).ratio()
        result["same_family"] += left["family"] == metadata["family"]
        result["same_source_group"] += (
            left["source"],
            left["group_id"],
        ) == (metadata["source"], metadata["group_id"])
        result["state_edit_ratio_ge_0_9"] += edit_ratio >= 0.9
        result["state_edit_ratio_ge_0_8"] += edit_ratio >= 0.8
        result["state_word_jaccard_ge_0_8"] += jaccard >= 0.8
    return {"pairs": len(pair_ids), **dict(sorted(result.items()))}


def _partition_metadata(path: Path, expected_sha: str, split: str) -> list[dict]:
    _check(path.is_file() and file_sha256(path) == expected_sha, "PARTITION_HASH")
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            _check(
                isinstance(source, dict)
                and source.get("split") == split
                and all(
                    isinstance(source.get(field), str)
                    for field in (
                        "id",
                        "source",
                        "group_id",
                        "family",
                        "input_sha256",
                    )
                ),
                "PARTITION_METADATA",
            )
            # No protected target field is selected or inspected.
            result.append(
                {
                    field: source[field]
                    for field in (
                        "id",
                        "source",
                        "group_id",
                        "family",
                        "input_sha256",
                    )
                }
            )
    _check(len(result) == 700, "PARTITION_ROW_COUNT")
    return result


def audit(
    *,
    capacity_path: Path,
    train_path: Path,
    rights_path: Path,
    choice_teacher_path: Path,
    score_teacher_path: Path,
    select_path: Path,
    cal_path: Path,
    core_directory: Path,
    tokenizer_dir: Path,
) -> dict[str, Any]:
    pinned = {
        "capacity": (capacity_path, CAPACITY_SHA256),
        "train": (train_path, TRAIN_SHA256),
        "rights": (rights_path, RIGHTS_SHA256),
        "choice_teacher": (choice_teacher_path, CHOICE_TEACHER_SHA256),
        "score_teacher": (score_teacher_path, SCORE_TEACHER_SHA256),
        "select": (select_path, SELECT_SHA256),
        "cal": (cal_path, CAL_SHA256),
        "core_manifest": (core_directory / "manifest.json", CORE_MANIFEST_SHA256),
        "tokenizer": (tokenizer_dir / "tokenizer.json", TOKENIZER_SHA256),
    }
    for name, (path, expected) in pinned.items():
        _check(
            path.is_file() and file_sha256(path) == expected,
            f"FROZEN_{name.upper()}_HASH",
        )
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    summary = capacity["summary"]
    entries = capacity["private_schedule"]
    _check(
        capacity.get("schema") == "decision2-27b-group-atomic-capacity/1"
        and capacity.get("status") == "PASS_CAPACITY_WITNESS_ONLY"
        and summary.get("seed") == 96
        and summary.get("minimum_score_rows") == 400
        and summary.get("rows") == EXPECTED_ROWS
        and summary.get("schedule_sha256") == LOCKED_SCHEDULE_SHA256
        and summary.get("by_type") == EXPECTED_TYPES
        and summary.get("raw_token_exposure") == EXPECTED_RAW_TOKENS
        and summary.get("padded_token_exposure") == EXPECTED_PADDED_TOKENS
        and digest(entries) == LOCKED_SCHEDULE_SHA256,
        "LOCKED_CANDIDATE_IDENTITY",
    )
    train_input = load_gold_free_train(train_path)
    by_id = {row["id"]: row for row in train_input}
    _check(len(by_id) == 7455 and len(entries) == EXPECTED_ROWS, "TRAIN_ID_COUNT")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    chosen = []
    seen = set()
    for item in entries:
        _check(
            isinstance(item, dict)
            and set(item) == {"id", "input_sha256", "tokens"}
            and item["id"] in by_id
            and item["id"] not in seen,
            "SELECTED_ROW_IDENTITY",
        )
        row = by_id[item["id"]]
        _check(
            item["input_sha256"] == row["input_sha256"]
            and type(item["tokens"]) is int
            and 0 < item["tokens"] <= 4096
            and item["tokens"] == token_count(row, tokenizer),
            "SELECTED_NATIVE_INPUT_OR_LENGTH",
        )
        chosen.append(row)
        seen.add(item["id"])
    _check(
        dict(sorted(collections.Counter(row["task_type"] for row in chosen).items()))
        == EXPECTED_TYPES
        and sum(item["tokens"] for item in entries) == EXPECTED_RAW_TOKENS
        and sum((item["tokens"] + 7) // 8 * 8 for item in entries)
        == EXPECTED_PADDED_TOKENS
        and EXPECTED_ROWS // 16 == EXPECTED_UPDATES,
        "LOCKED_EXPOSURE",
    )
    groups = group_integrity(train_input, chosen)
    _check(groups["missing_any_row_groups"] == 0, "WHOLE_SOURCE_GROUP")
    roles, role_hashes = load_core_inventory(core_directory)
    projected = [
        project_partition_row({**row, "split": "train"}, "train") for row in chosen
    ]
    train_role = {row["id"]: row for row in roles["rights_clean_train"]}
    _check(
        all(train_role.get(row["id"]) == row for row in projected),
        "CORE_TRAIN_INPUT_IDENTITY",
    )
    # Recompute saved bounded exclusions from the pinned input-only inventory.
    excluded = set()
    near_evidence = {}
    for role in ("rights_clean_select", "rights_clean_cal"):
        matched, pairs = state_near_ids(
            [
                project_partition_row({**row, "split": "train"}, "train")
                for row in train_input
            ],
            roles[role],
        )
        pair_ids = sorted(pairs)
        _check(
            pair_ids
            == [tuple(item) for item in capacity["private_near_pair_ids"][role]]
            and digest(pair_ids)
            == capacity["near_counts"][role]["pair_identity_sha256"],
            "NEAR_EXCLUSION_EVIDENCE",
        )
        excluded.update(matched)
        near_evidence[role] = {
            "pair_count": len(pair_ids),
            "matched_train_rows": len(matched),
            "pair_ids_sha256": digest(pair_ids),
        }
    _check(
        excluded == set(capacity["private_near_excluded_train_ids"])
        and len(excluded) == 83,
        "NEAR_EXCLUSION_IDENTITY",
    )
    excluded_groups = {
        (row["source"], row["group_id"]) for row in train_input if row["id"] in excluded
    }
    _check(
        not excluded_groups & {(row["source"], row["group_id"]) for row in chosen},
        "EXCLUDED_GROUP_SELECTED",
    )
    partitions = {
        "rights_clean_select": _partition_metadata(
            select_path, SELECT_SHA256, "select"
        ),
        "rights_clean_cal": _partition_metadata(cal_path, CAL_SHA256, "cal"),
    }
    partition_pairs = source_group_pairs(chosen, partitions)
    bare_group_pairs = source_group_pairs(chosen, partitions, match_source=False)
    train_full = load_partition(train_path, "train")
    _check(
        len(train_full) == len(train_input)
        and all(
            a["id"] == b["id"] and a["input_sha256"] == b["input_sha256"]
            for a, b in zip(train_full, train_input, strict=True)
        ),
        "TRAIN_FULL_INPUT_IDENTITY",
    )
    full_by_id = {row["id"]: row for row in train_full}
    selected_full = [full_by_id[row["id"]] for row in chosen]
    rights = selected_rights(
        json.loads(rights_path.read_text(encoding="utf-8")), train_full, selected_full
    )
    vectors = verify_teacher(
        json.loads(choice_teacher_path.read_text(encoding="utf-8")),
        [row for row in train_full if row["task_type"] != "score"],
        task="choice_noul",
    ) | verify_teacher(
        json.loads(score_teacher_path.read_text(encoding="utf-8")),
        [row for row in train_full if row["task_type"] == "score"],
        task="score",
    )
    _check(len(vectors) == 7455, "TEACHER_VECTOR_COVERAGE")
    mask = teacher_mask(selected_full, vectors)
    _check(
        summary["teacher_mask"]["mask_sha256"] == MASK_SHA256
        and summary["teacher_mask"]["by_type"] == mask["by_type"]
        and summary["teacher_mask"]["score_by_level_count"]
        == mask["score_by_level_count"]
        and summary["teacher_mask"]["masked_rows"] == mask["masked_rows"]
        and mask["masked_rows"] == 971
        and mask["by_type"] == {"choice": 429, "noul": 368, "score": 174}
        and mask["score_by_level_count"].get("3") == 63,
        "LOCKED_TEACHER_MASK",
    )
    overlap = {}
    private_pairs = {}
    for role, protected in sorted(roles.items()):
        if role == "rights_clean_train":
            continue
        surfaces = {
            surface: surface_pair_ids(projected, protected, surface=surface)
            for surface in ("complete_input", "state_evidence")
        }
        private_pairs[role] = surfaces
        overlap[role] = {
            surface: {name: len(pair_ids) for name, pair_ids in matches.items()}
            for surface, matches in surfaces.items()
        }
    hard_overlap = any(
        any(counts[name] for name in ("same_row_ids", "exact_raw", "exact_normalized"))
        for result in overlap.values()
        for counts in result.values()
    ) or any(result["state_evidence"]["near"] for result in overlap.values())
    near_complete = sum(result["complete_input"]["near"] for result in overlap.values())
    selected_by_id = {row["id"]: row for row in selected_full}
    near_pair_profiles = {}
    for role in ("rights_clean_select", "rights_clean_cal"):
        metadata = {row["id"]: row for row in partitions[role]}
        protected = {row["id"]: row for row in roles[role]}
        near_pair_profiles[role] = {
            surface: near_pair_profile(
                private_pairs[role][surface]["near"],
                selected_by_id,
                metadata,
                protected,
            )
            for surface in ("complete_input", "state_evidence")
        }
    blockers = []
    if any(partition_pairs.values()) or any(bare_group_pairs.values()):
        blockers.append("SOURCE_GROUP_CROSSES_SELECT_CAL")
    if hard_overlap:
        blockers.append("NATIVE_INPUT_EXACT_OR_STATE_NEAR_OVERLAP")
    if near_complete:
        blockers.append("COMPLETE_INPUT_NEAR_PAIRS_REQUIRE_SEMANTIC_REVIEW")
    # Native evaluator projections are intentionally input-only and cannot
    # attest original source record IDs; do not manufacture source clearance.
    blockers.append("NATIVE_EVAL_SOURCE_PROVENANCE_REVIEW_PENDING")
    return {
        "schema": SCHEMA,
        "status": "HOLD" if blockers else "PASS_CPU_ADMISSION_ONLY",
        "blockers": blockers,
        "locked_schedule_sha256": LOCKED_SCHEDULE_SHA256,
        "pinned_sha256": {name: expected for name, (_, expected) in pinned.items()},
        "core_role_sha256": role_hashes,
        "selected_rows": len(chosen),
        "selected_types": EXPECTED_TYPES,
        "raw_token_exposure": EXPECTED_RAW_TOKENS,
        "padded_token_exposure": EXPECTED_PADDED_TOKENS,
        "group_integrity": groups,
        "near_exclusion": near_evidence,
        "source_group_partition_pair_counts": {
            role: len(pairs) for role, pairs in partition_pairs.items()
        },
        "bare_group_partition_pair_counts": {
            role: len(pairs) for role, pairs in bare_group_pairs.items()
        },
        "rights": rights,
        "teacher_mask": mask,
        "overlap_pair_counts": overlap,
        "near_pair_provenance_profile": near_pair_profiles,
        "private_source_group_pairs": partition_pairs,
        "private_bare_group_pairs": bare_group_pairs,
        "private_overlap_pair_ids": private_pairs,
        "gpu_hours": 0,
        "limits": "Exact and bounded-near projected native input screens only; near complete-input pairs and original-source identity of native evaluator roles still need independent semantic/source review. No protected targets, model output, optimizer or GPU used. Ledger integrity is not a legal conclusion.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "capacity",
        "train",
        "rights",
        "choice_teacher",
        "score_teacher",
        "select",
        "cal",
        "core_directory",
        "tokenizer_dir",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(
            capacity_path=args.capacity,
            train_path=args.train,
            rights_path=args.rights,
            choice_teacher_path=args.choice_teacher,
            score_teacher_path=args.score_teacher,
            select_path=args.select,
            cal_path=args.cal,
            core_directory=args.core_directory,
            tokenizer_dir=args.tokenizer_dir,
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
                "receipt_sha256": file_sha256(args.output),
                "reason_code": result.get("reason_code"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
