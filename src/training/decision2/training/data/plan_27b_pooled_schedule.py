"""Materialize a distinct, label-free 27B whole-group pooled-quota schedule.

The old exact-quota admission remains unchanged and failed. This prospective
planner minimizes source-quota drift at fixed type totals. Its private output
is a schedule candidate, never a teacher/rights/overlap training admission.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from training.data.audit_27b_quota_envelope import (
    _best_source_counts,
    load_gold_free_train,
    token_count,
)
from training.data.audit_27b_teacher_admission import (
    SOURCE_REVISION,
    AdmissionHold,
    _key,
    choose_whole_groups,
    largest_remainder,
    write_once,
)
from training.model.data import digest, file_sha256

SCHEMA = "decision2-27b-pooled-train-schedule/1"


def plan(
    rows: list[dict[str, Any]],
    lengths: dict[str, int],
    *,
    total: int = 2560,
    max_length: int = 4096,
    min_score: int = 460,
) -> dict[str, Any]:
    """Select exact type totals with minimum L1 source drift, without labels."""
    if len(lengths) != len(rows) or any(lengths.get(row["id"], 0) <= 0 for row in rows):
        raise AdmissionHold("TOKEN_LENGTH_COVERAGE")
    admitted = [row for row in rows if lengths[row["id"]] <= max_length]
    score = [row for row in admitted if row["task_type"] == "score"]
    if not min_score <= len(score) <= total:
        raise AdmissionHold("SCORE_ADMITTED_COUNT")
    score_all = collections.Counter(
        (row["source"], row["group_id"]) for row in rows if row["task_type"] == "score"
    )
    score_kept = collections.Counter((row["source"], row["group_id"]) for row in score)
    if any(score_kept[group] not in (0, count) for group, count in score_all.items()):
        raise AdmissionHold("PARTIAL_SCORE_GROUP_AT_CONTEXT_LIMIT")

    remaining = total - len(score)
    targets = {"choice": remaining // 2, "noul": remaining - remaining // 2}
    selected = list(score)
    source_quotas: dict[str, dict[str, int]] = {}
    source_realized: dict[str, dict[str, int]] = {}
    source_l1: dict[str, int] = {}
    for task in ("choice", "noul"):
        pool = [row for row in admitted if row["task_type"] == task]
        counts = dict(collections.Counter(row["source"] for row in pool))
        quota = largest_remainder(counts, targets[task])
        groups: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        for row in pool:
            groups[row["source"]][row["group_id"]].append(row)
        ordered = {
            source: sorted(
                source_groups.values(),
                key=lambda group: _key(f"{task}/{source}/{group[0]['group_id']}"),
            )
            for source, source_groups in groups.items()
        }
        realized, l1 = _best_source_counts(ordered, quota, targets[task])
        for source in sorted(realized):
            selected.extend(choose_whole_groups(ordered[source], realized[source]))
        source_quotas[task] = dict(sorted(quota.items()))
        source_realized[task] = dict(sorted(realized.items()))
        source_l1[task] = l1

    if len(selected) != total or len({row["id"] for row in selected}) != total:
        raise AdmissionHold("SCHEDULE_ID_OR_TOTAL")
    selected.sort(key=lambda row: _key(f"row/{row['id']}/{row['input_sha256']}"))
    by_type = dict(
        sorted(collections.Counter(row["task_type"] for row in selected).items())
    )
    if by_type != {
        "choice": targets["choice"],
        "noul": targets["noul"],
        "score": len(score),
    }:
        raise AdmissionHold("SCHEDULE_TYPE_TOTAL")
    schedule = [
        {
            "id": row["id"],
            "input_sha256": row["input_sha256"],
            "tokens": lengths[row["id"]],
        }
        for row in selected
    ]
    source_type_groups = collections.defaultdict(set)
    for row in selected:
        source_type_groups[(row["source"], row["group_id"])].add(row["task_type"])
    all_source_type_groups = collections.defaultdict(set)
    for row in rows:
        all_source_type_groups[(row["source"], row["group_id"])].add(row["task_type"])
    partial_cross_type = sum(
        bool(source_type_groups[group]) and source_type_groups[group] != tasks
        for group, tasks in all_source_type_groups.items()
    )
    return {
        "schema": SCHEMA,
        "status": "SCHEDULE_CANDIDATE_ONLY",
        "source_revision": SOURCE_REVISION,
        "rows": total,
        "updates": total // 16,
        "max_length": max_length,
        "by_type": by_type,
        "by_language": dict(
            sorted(collections.Counter(row["language"] for row in selected).items())
        ),
        "source_quotas": source_quotas,
        "source_realized": source_realized,
        "source_l1": source_l1,
        "source_l1_total": sum(source_l1.values()),
        "partial_cross_type_source_groups": partial_cross_type,
        "raw_token_exposure": sum(item["tokens"] for item in schedule),
        "padded_token_exposure": sum(
            (item["tokens"] + 7) // 8 * 8 for item in schedule
        ),
        "schedule_sha256": digest(schedule),
        "schedule": schedule,
        "gpu_hours": 0,
        "limits": "Input-only whole groups are scoped within source/type. Cross-type variants can be partially selected. No teacher mask, rights review, protected overlap, model parity or quality gate is established.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        rows = load_gold_free_train(args.train)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, local_files_only=True
        )
        result = plan(rows, {row["id"]: token_count(row, tokenizer) for row in rows})
        result["train_sha256"] = file_sha256(args.train)
        result["tokenizer_sha256"] = {
            name: file_sha256(args.tokenizer_dir / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        }
    except AdmissionHold as error:
        result = {
            "schema": SCHEMA,
            "status": "HOLD",
            "reason_code": str(error),
            "gpu_hours": 0,
        }
    write_once(args.output, result)
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": file_sha256(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
