"""CPU-only capacity witness for a *future* group-atomic 27B schedule.

This does not authorize training or amend the failed pooled schedule. It
excludes entire TRAIN source groups with bounded near-state SELECT/CAL hits,
then asks whether an exact 2,560-row whole-group witness can fit the earlier
raw and eight-token padded exposure bands and frozen teacher-mask minima.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

from training.data.audit_27b_full_input_overlap import _normalize, input_spans
from training.data.audit_27b_pooled_admission import (
    CORE_MANIFEST_SHA256,
    SCHEDULE_SHA256,
    load_core_inventory,
    selected_rights,
)
from training.data.audit_27b_quota_envelope import load_gold_free_train, token_count
from training.data.audit_27b_teacher_admission import (
    CHOICE_TEACHER_SHA256,
    RIGHTS_SHA256,
    SCORE_TEACHER_SHA256,
    TRAIN_SHA256,
    AdmissionHold,
    _check,
    teacher_mask,
    verify_teacher,
    write_once,
)
from training.data.build_pilot import _simhash
from training.data.plan_goldfree_inventory import project_partition_row
from training.model.data import digest, file_sha256, load_partition

SCHEMA = "decision2-27b-group-atomic-capacity/1"
SEEDS = 128


def state_near_ids(
    train: list[dict[str, Any]], protected: list[dict[str, Any]]
) -> tuple[set[str], set[tuple[str, str]]]:
    """Mirror the bounded state-only screen and retain private pair identities."""
    right = []
    bands: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    exact: dict[str, set[str]] = collections.defaultdict(set)
    for row in protected:
        for span in input_spans(
            {"id": row["id"], "state": "STATE EVIDENCE\n" + row["state"]}
        ):
            value = _normalize(span)
            exact[value].add(row["id"])
            if len(value) < 40:
                continue
            bits = _simhash(value)
            index = len(right)
            right.append((row["id"], value, bits))
            for band in range(8):
                bands[(band, (bits >> (8 * band)) & 255)].add(index)
    matched = set()
    pairs = set()
    for row in train:
        for span in input_spans(
            {"id": row["id"], "state": "STATE EVIDENCE\n" + row["state"]}
        ):
            value = _normalize(span)
            for protected_id in exact.get(value, ()):
                pairs.add((row["id"], protected_id))
                matched.add(row["id"])
            if len(value) < 40:
                continue
            bits = _simhash(value)
            candidates = set()
            for band in range(8):
                candidates.update(bands.get((band, (bits >> (8 * band)) & 255), ()))
            for index in candidates:
                protected_id, other, other_bits = right[index]
                if abs(len(value) - len(other)) > 0.08 * max(len(value), len(other)):
                    continue
                if (bits ^ other_bits).bit_count() > 8:
                    continue
                if difflib.SequenceMatcher(None, value, other).ratio() >= 0.94:
                    pairs.add((row["id"], protected_id))
                    matched.add(row["id"])
    return matched, pairs


def choose_group_atomic_witness(
    train: list[dict[str, Any]],
    lengths: dict[str, int],
    excluded_ids: set[str],
    vectors: dict[str, dict[str, float]],
    *,
    total: int,
    target_raw_tokens: int,
    target_padded_tokens: int,
    minimum_score_rows: int = 460,
    seeds: int = SEEDS,
) -> dict[str, Any]:
    """Return a deterministic feasible witness, or a bounded HOLD report."""
    _check(0 <= minimum_score_rows <= total and seeds > 0, "CAPACITY_PARAMETERS")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in train:
        groups[(row["source"], row["group_id"])].append(row)
    excluded_groups = {
        key
        for key, rows in groups.items()
        if any(row["id"] in excluded_ids for row in rows)
    }
    eligible = {
        key: rows
        for key, rows in groups.items()
        if key not in excluded_groups
        and all(lengths[row["id"]] <= 4096 for row in rows)
    }
    score_groups = {
        key: rows
        for key, rows in eligible.items()
        if any(row["task_type"] == "score" for row in rows)
    }
    other_groups = {
        key: rows for key, rows in eligible.items() if key not in score_groups
    }
    score_rows = [
        row
        for rows in score_groups.values()
        for row in rows
        if row["task_type"] == "score"
    ]
    score_mask = teacher_mask(
        score_rows,
        vectors,
        minimum_score=0,
        minimum_three_level=0,
        minimum_human=0,
    )
    summary = {
        "minimum_score_rows": minimum_score_rows,
        "train_groups": len(groups),
        "near_excluded_groups": len(excluded_groups),
        "context_and_near_eligible_groups": len(eligible),
        "context_and_near_eligible_rows": sum(len(rows) for rows in eligible.values()),
        "eligible_score_rows": len(score_rows),
        "eligible_score_teacher_mask": score_mask,
    }
    fixed = [row for rows in score_groups.values() for row in rows]
    if (
        len(score_rows) < minimum_score_rows
        or score_mask["score_by_level_count"].get("3", 0) < 40
        or len(fixed) > total
        or sum(len(rows) for rows in other_groups.values()) < total - len(fixed)
    ):
        return {"status": "HOLD_CAPACITY_UPPER_BOUND", "summary": summary}
    target_other = total - len(fixed)
    best = None
    for seed in range(seeds):
        ordered = sorted(
            other_groups,
            key=lambda key: hashlib.sha256(
                f"future-group-atomic/{seed}/{key[0]}/{key[1]}".encode()
            ).digest(),
        )
        chosen = list(fixed)
        remaining = target_other
        for key in ordered:
            rows = other_groups[key]
            if len(rows) <= remaining:
                chosen.extend(rows)
                remaining -= len(rows)
            if remaining == 0:
                break
        if remaining:
            continue
        raw = sum(lengths[row["id"]] for row in chosen)
        padded = sum((lengths[row["id"]] + 7) // 8 * 8 for row in chosen)
        distance = max(
            abs(raw - target_raw_tokens) / target_raw_tokens,
            abs(padded - target_padded_tokens) / target_padded_tokens,
        )
        identity = [
            {
                "id": row["id"],
                "input_sha256": row["input_sha256"],
                "tokens": lengths[row["id"]],
            }
            for row in sorted(chosen, key=lambda row: row["id"])
        ]
        candidate = {
            "seed": seed,
            "rows": len(chosen),
            "raw_token_exposure": raw,
            "padded_token_exposure": padded,
            "relative_exposure_max_delta": distance,
            "by_type": dict(
                sorted(collections.Counter(row["task_type"] for row in chosen).items())
            ),
            "schedule_sha256": digest(identity),
            "schedule": identity,
        }
        try:
            candidate["teacher_mask"] = teacher_mask(chosen, vectors)
        except AdmissionHold:
            continue
        if best is None or distance < best[0]:
            best = (distance, candidate)
    if best is None:
        return {"status": "HOLD_NO_EXACT_GROUP_WITNESS", "summary": summary}
    distance, candidate = best
    summary.update(
        {key: value for key, value in candidate.items() if key != "schedule"}
    )
    return {
        "status": (
            "PASS_CAPACITY_WITNESS_ONLY" if distance <= 0.01 else "HOLD_TOKEN_EXPOSURE"
        ),
        "summary": summary,
        "private_schedule": candidate["schedule"],
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
    parser.add_argument("--minimum-score-rows", type=int, default=460)
    args = parser.parse_args()
    try:
        inputs = {
            "schedule": (args.schedule, SCHEDULE_SHA256),
            "train": (args.train, TRAIN_SHA256),
            "rights": (args.rights, RIGHTS_SHA256),
            "choice_teacher": (args.choice_teacher, CHOICE_TEACHER_SHA256),
            "score_teacher": (args.score_teacher, SCORE_TEACHER_SHA256),
            "core_manifest": (
                args.core_directory / "manifest.json",
                CORE_MANIFEST_SHA256,
            ),
        }
        for name, (path, expected) in inputs.items():
            _check(
                path.is_file() and file_sha256(path) == expected,
                f"FROZEN_{name.upper()}_HASH",
            )
        schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
        train_input = load_gold_free_train(args.train)
        train = load_partition(args.train, "train")
        _check(
            len(train) == len(train_input)
            and all(
                a["id"] == b["id"] and a["input_sha256"] == b["input_sha256"]
                for a, b in zip(train, train_input, strict=True)
            ),
            "TRAIN_LABEL_INPUT_IDENTITY",
        )
        rights = selected_rights(
            json.loads(args.rights.read_text(encoding="utf-8")), train, train
        )
        choice_rows = [row for row in train if row["task_type"] != "score"]
        score_rows = [row for row in train if row["task_type"] == "score"]
        vectors = verify_teacher(
            json.loads(args.choice_teacher.read_text(encoding="utf-8")),
            choice_rows,
            task="choice_noul",
        ) | verify_teacher(
            json.loads(args.score_teacher.read_text(encoding="utf-8")),
            score_rows,
            task="score",
        )
        roles, role_hashes = load_core_inventory(args.core_directory)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, local_files_only=True
        )
        lengths = {row["id"]: token_count(row, tokenizer) for row in train_input}
        train_projected = [project_partition_row(row, "train") for row in train]
        prior_ids = {item["id"] for item in schedule["schedule"]}
        _check(len(prior_ids) == 2560, "PRIOR_SCHEDULE_IDS")
        prior_projected = [row for row in train_projected if row["id"] in prior_ids]
        _check(len(prior_projected) == 2560, "PRIOR_SCHEDULE_TRAIN_IDENTITY")
        excluded_ids = set()
        near_counts = {}
        prior_near_counts = {}
        private_near_pair_ids = {}
        for role in ("rights_clean_select", "rights_clean_cal"):
            matches, pairs = state_near_ids(train_projected, roles[role])
            excluded_ids.update(matches)
            private_near_pair_ids[role] = sorted(pairs)
            near_counts[role] = {
                "pairs": len(pairs),
                "train_rows": len(matches),
                "pair_identity_sha256": digest(private_near_pair_ids[role]),
            }
            prior_matches, prior_pairs = state_near_ids(prior_projected, roles[role])
            prior_near_counts[role] = {
                "pairs": len(prior_pairs),
                "train_rows": len(prior_matches),
            }
        result = choose_group_atomic_witness(
            train,
            lengths,
            excluded_ids,
            vectors,
            total=2560,
            target_raw_tokens=schedule["raw_token_exposure"],
            target_padded_tokens=schedule["padded_token_exposure"],
            minimum_score_rows=args.minimum_score_rows,
        )
        result.update(
            {
                "schema": SCHEMA,
                "rights": rights,
                "core_role_sha256": role_hashes,
                "near_counts": near_counts,
                "prior_schedule_near_counts": prior_near_counts,
                "near_excluded_train_rows": len(excluded_ids),
                "private_near_pair_ids": private_near_pair_ids,
                "private_near_excluded_train_ids": sorted(excluded_ids),
                "source_schedule_sha256": SCHEDULE_SHA256,
                "gpu_hours": 0,
                "limits": "Capacity only: finite hash-order search, bounded near state exclusion, eight-token rather than dynamic batch padding, no semantic/source clearance, no optimizer or model outputs. A PASS never admits training.",
            }
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
