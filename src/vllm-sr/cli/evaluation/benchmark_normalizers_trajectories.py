"""Strict parser for TwinRouterBench trajectory-prefix artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    boolean,
    exact_object,
    integer,
    iter_jsonl,
    load_json,
    require_array,
    required_file,
    string,
)
from cli.evaluation.benchmark_normalization_types import (
    BenchmarkNormalizerDescriptor,
    NormalizedAdapterPayload,
)
from cli.evaluation.benchmark_normalizers_common import (
    applicable_track_ids,
    messages,
    native_digest,
    opaque_id,
)
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.suite_contract import NormalizedTrajectoryStep

_BANK_KEYS = {
    "id",
    "benchmark",
    "scenario",
    "instance_id",
    "step_index",
    "total_steps",
    "messages",
    "target_tier",
    "target_tier_id",
    "benchmark_display",
    "benchmark_subset",
    "benchmark_version",
    "pipeline_stage",
    "collector",
    "collected_at",
    "notes",
}
_SUMMARY_KEYS = {
    "classifier",
    "shard",
    "sample_mode",
    "sampled",
    "seed",
    "proportional_quotas",
    "benchmark_counts",
    "exact_match",
    "tier_match_accuracy",
    "accuracy_excluding_errors",
    "api_errors",
    "valid_response_rate",
    "scores_v2",
    "section_11",
    "router_accounting",
    "by_benchmark",
    "errors",
    "rows",
}


def _summary_row(value: Any) -> dict[str, Any]:
    base = {
        "id",
        "benchmark",
        "gold_tier_id",
        "instance_id",
        "step_index",
        "total_steps",
        "messages",
    }
    row = exact_object(
        value,
        required=base,
        optional={"pred_tier_id", "match", "passed", "usage", "error"},
        label="TwinRouterBench summary row",
    )
    success_fields = {"pred_tier_id", "match", "passed"}
    if "error" in row:
        if success_fields & set(row):
            raise NormalizationError("TwinRouterBench error row mixes success fields")
        string(row["error"], "TwinRouterBench row error")
    elif not success_fields.issubset(row):
        raise NormalizationError("TwinRouterBench success row lacks prediction fields")
    return row


def _question_bank(path: Path) -> dict[str, dict[str, Any]]:
    bank: dict[str, dict[str, Any]] = {}
    for value in iter_jsonl(path):
        row = exact_object(value, required=_BANK_KEYS, label="TwinRouterBench bank row")
        row_id = string(row["id"], "TwinRouterBench bank id")
        if row_id in bank:
            raise NormalizationError("TwinRouterBench question bank repeats id")
        messages(row["messages"], "TwinRouterBench messages")
        integer(row["step_index"], "TwinRouterBench step_index", minimum=1)
        integer(row["total_steps"], "TwinRouterBench total_steps", minimum=1)
        bank[row_id] = row
    return bank


def _summary_export(
    path: Path, max_bytes: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], int]:
    summary = exact_object(
        load_json(path, max_bytes=max_bytes),
        required=_SUMMARY_KEYS,
        label="TwinRouterBench eval summary",
    )
    summary_rows = [
        _summary_row(item) for item in require_array(summary["rows"], "summary rows")
    ]
    by_id = {
        string(row["id"], "TwinRouterBench summary id"): row for row in summary_rows
    }
    return summary, by_id, len(summary_rows)


def normalize_twinrouterbench(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    artifacts = {item.id: item for item in descriptor.required_artifacts}
    bank = _question_bank(required_file(root, artifacts["question-bank"]))
    summary_req = artifacts["summary"]
    summary, by_id, summary_row_count = _summary_export(
        required_file(root, summary_req), summary_req.max_bytes
    )
    if len(by_id) != summary_row_count or set(by_id) != set(bank):
        raise NormalizationError(
            "TwinRouterBench question bank and summary do not align"
        )
    by_instance: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    for row_id in sorted(bank):
        source = bank[row_id]
        result = by_id[row_id]
        if (
            result["instance_id"] != source["instance_id"]
            or result["step_index"] != source["step_index"]
            or result["total_steps"] != source["total_steps"]
            or result["gold_tier_id"] != source["target_tier_id"]
            or result["messages"] != source["messages"]
        ):
            raise NormalizationError(
                "TwinRouterBench summary row drifted from question bank"
            )
        by_instance[
            string(source["instance_id"], "TwinRouterBench instance_id")
        ].append((source, result))
    visible = []
    grading = []
    trajectories = []
    for instance_id, pairs in sorted(by_instance.items()):
        pairs.sort(
            key=lambda pair: integer(pair[0]["step_index"], "step_index", minimum=1)
        )
        total_steps = integer(pairs[0][0]["total_steps"], "total_steps", minimum=1)
        if [pair[0]["step_index"] for pair in pairs] != list(range(1, total_steps + 1)):
            raise NormalizationError(
                "TwinRouterBench trajectory is incomplete or unordered"
            )
        trajectory_id = opaque_id("trajectory", "twinrouterbench", instance_id)
        case_id = opaque_id("case", "twinrouterbench", instance_id)
        first = pairs[0][0]
        visible.append(
            CaseVisible(
                id=case_id,
                track_ids=applicable_track_ids(
                    descriptor.track_ids,
                    modality="text",
                ),
                messages=messages(first["messages"], "TwinRouterBench messages"),
                tags=("twinrouterbench", string(first["benchmark"], "benchmark")),
                trajectory_id=trajectory_id,
            )
        )
        grading.append(CaseGrading(case_id=case_id))
        matches = []
        passes = []
        for sequence, (source, result) in enumerate(pairs):
            matched = boolean(result.get("match", False), "TwinRouterBench match")
            passed = boolean(result.get("passed", False), "TwinRouterBench passed")
            matches.append(matched)
            passes.append(passed)
            terminal = sequence == len(pairs) - 1
            trajectories.append(
                NormalizedTrajectoryStep(
                    trajectory_id=trajectory_id,
                    step_id=opaque_id("step", source["id"]),
                    sequence=sequence,
                    case_id=case_id,
                    selected_action_id=(
                        opaque_id("action", "tier", result["pred_tier_id"])
                        if "pred_tier_id" in result
                        else None
                    ),
                    terminal=terminal,
                    terminal_success=(all(passes) if terminal else None),
                    task_score=(sum(matches) / len(matches) if terminal else None),
                    source_record_digest=native_digest(result),
                )
            )
    if not visible:
        raise NormalizationError("TwinRouterBench export has no trajectories")
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        trajectories=tuple(trajectories),
        split_protocol=f"Native {summary['sample_mode']} sample with seed {summary['seed']}.",
    )
