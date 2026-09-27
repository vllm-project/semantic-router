"""Gold-free TRAIN-group capacity envelope for a *new* 27B teacher arm.

This never reads labels, teacher outputs, or evaluation sets. Its receipt is
aggregate-only and cannot admit an experiment without separate rights, teacher
identity, and protected-inventory checks.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from training.data.audit_27b_teacher_admission import (
    SOURCE_REVISION,
    TRAIN_SHA256,
    AdmissionHold,
    _key,
    largest_remainder,
    write_once,
)
from training.model.data import INPUT_FIELDS, digest, file_sha256

SCHEMA = "decision2-27b-train-group-envelope/1"
TRAIN_FIELDS = {
    "id",
    "source",
    "group_id",
    "task_type",
    "language",
    "state",
    "instructions",
    "options",
    "input_sha256",
}


def load_gold_free_train(path: Path) -> list[dict[str, Any]]:
    if file_sha256(path) != TRAIN_SHA256:
        raise AdmissionHold("FROZEN_TRAIN_HASH")
    result = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            source_row = json.loads(line)
            if not isinstance(source_row, dict) or not TRAIN_FIELDS.issubset(
                source_row
            ):
                raise AdmissionHold("TRAIN_SCHEMA")
            # Construct the projection before downstream code sees the row.
            row = {field: source_row[field] for field in TRAIN_FIELDS}
            if (
                row["task_type"] not in {"choice", "noul", "score"}
                or row["id"] in seen
                or row["input_sha256"]
                != digest({field: row[field] for field in INPUT_FIELDS})
            ):
                raise AdmissionHold("TRAIN_ID_OR_INPUT")
            seen.add(row["id"])
            result.append(row)
    if len(result) != 7455:
        raise AdmissionHold("TRAIN_ROW_COUNT")
    return result


def token_count(row: dict[str, Any], tokenizer: Any) -> int:
    # The native System One segmented prompt; no target or gold is touched.
    from training.model.decision_model import segments

    prefix, options, suffix = segments(row)
    return sum(
        len(tokenizer.encode(part, add_special_tokens=False))
        for part in (prefix, *options, suffix)
    )


def _reachable(groups: list[list[dict[str, Any]]], ceiling: int) -> list[int]:
    bits = 1
    mask = (1 << (ceiling + 1)) - 1
    for group in groups:
        bits = (bits | (bits << len(group))) & mask
    return [count for count in range(ceiling + 1) if bits & (1 << count)]


def _best_source_counts(
    groups: dict[str, list[list[dict[str, Any]]]],
    quota: dict[str, int],
    target: int,
) -> tuple[dict[str, int], int]:
    """Minimize source-quota L1 subject to exact whole-group type count.

    Equal-cost solutions use a fixed source-hash order and lexicographic count
    vector. This is a prospective feasibility calculation, not a TRAIN schedule.
    """
    sources = sorted(groups, key=lambda source: _key(f"source/{source}"))
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for source in sources:
        possible = _reachable(groups[source], target)
        next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
        for subtotal, (cost, chosen) in states.items():
            for count in possible:
                candidate_total = subtotal + count
                if candidate_total > target:
                    break
                candidate = (cost + abs(count - quota[source]), chosen + (count,))
                current = next_states.get(candidate_total)
                if current is None or candidate < current:
                    next_states[candidate_total] = candidate
        states = next_states
        if not states:
            raise AdmissionHold("POOLED_WHOLE_GROUP_QUOTA")
    if target not in states:
        raise AdmissionHold("POOLED_WHOLE_GROUP_QUOTA")
    cost, counts = states[target]
    return dict(zip(sources, counts, strict=True)), cost


def evaluate(
    rows: list[dict[str, Any]],
    lengths: dict[str, int],
    *,
    total: int = 2560,
    max_length: int = 4096,
    min_score: int = 460,
) -> dict[str, Any]:
    if len(lengths) != len(rows) or any(lengths.get(row["id"], 0) <= 0 for row in rows):
        raise AdmissionHold("TOKEN_LENGTH_COVERAGE")
    admitted = [row for row in rows if lengths[row["id"]] <= max_length]
    score = [row for row in admitted if row["task_type"] == "score"]
    if len(score) < min_score or len(score) > total:
        raise AdmissionHold("SCORE_ADMITTED_COUNT")
    remaining = total - len(score)
    targets = {"choice": remaining // 2, "noul": remaining - remaining // 2}

    all_groups: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        all_groups[row["group_id"]].add(row["task_type"])
    partial_score_groups = 0
    for group_id in {row["group_id"] for row in rows if row["task_type"] == "score"}:
        group_rows = [
            row
            for row in rows
            if row["group_id"] == group_id and row["task_type"] == "score"
        ]
        included = sum(lengths[row["id"]] <= max_length for row in group_rows)
        partial_score_groups += 0 < included < len(group_rows)

    by_type = {}
    strict_infeasible = 0
    relaxed_l1 = 0
    max_source_shift = 0
    for task in ("choice", "noul"):
        pool = [row for row in admitted if row["task_type"] == task]
        counts = dict(collections.Counter(row["source"] for row in pool))
        quota = largest_remainder(counts, targets[task])
        groups: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        for row in pool:
            groups[row["source"]][row["group_id"]].append(row)
        ordered_groups = {
            source: sorted(
                source_groups.values(),
                key=lambda group: _key(f"{task}/{source}/{group[0]['group_id']}"),
            )
            for source, source_groups in groups.items()
        }
        impossible = sum(
            quota[source] not in _reachable(ordered_groups[source], quota[source])
            for source in quota
        )
        strict_infeasible += impossible
        alternative, l1 = _best_source_counts(ordered_groups, quota, targets[task])
        relaxed_l1 += l1
        max_source_shift = max(
            max_source_shift,
            *(abs(alternative[source] - quota[source]) for source in quota),
        )
        by_type[task] = {
            "admitted_rows": len(pool),
            "target": targets[task],
            "sources": len(counts),
            "whole_groups": sum(len(group) for group in ordered_groups.values()),
            "strict_infeasible_source_buckets": impossible,
            "minimum_l1_source_quota_shift": l1,
            "largest_source_quota_shift": max(
                abs(alternative[source] - quota[source]) for source in quota
            ),
        }
    return {
        "schema": SCHEMA,
        "status": "FEASIBLE_CAPACITY_ONLY",
        "train_sha256": TRAIN_SHA256,
        "source_revision": SOURCE_REVISION,
        "train_rows": len(rows),
        "admitted_rows": len(admitted),
        "score_admitted_rows": len(score),
        "score_partial_groups_at_context_limit": partial_score_groups,
        "groups_spanning_task_types": sum(
            len(tasks) > 1 for tasks in all_groups.values()
        ),
        "type_profile": by_type,
        "strict_infeasible_source_buckets": strict_infeasible,
        "minimum_l1_source_quota_shift": relaxed_l1,
        "largest_source_quota_shift": max_source_shift,
        "gpu_hours": 0,
        "limits": "TRAIN metadata and token lengths only. No labels, teacher vectors, rights ledger, protected prompts, weights or evaluation outcomes were read. Capacity does not authorize training.",
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
        result = evaluate(
            rows, {row["id"]: token_count(row, tokenizer) for row in rows}
        )
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
