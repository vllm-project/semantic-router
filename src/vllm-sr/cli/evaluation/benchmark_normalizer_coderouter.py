"""Strict normalization for CodeRouterBench decision artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    exact_object,
    integer,
    iter_csv,
    iter_jsonl,
    load_json,
    no_duplicate,
    number,
    require_array,
    required_file,
    string,
)
from cli.evaluation.benchmark_normalization_types import (
    BenchmarkNormalizerDescriptor,
    NormalizedAdapterPayload,
)
from cli.evaluation.benchmark_normalizers_common import (
    arm_map,
    native_digest,
    text_case,
)
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.suite_contract import NormalizedDecision, NormalizedOutcome

_CODEROUTER_RESULT_HEADER = (
    "task_id",
    "split",
    "source_split",
    "dimension",
    "model",
    "score",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_ms",
    "cost_source",
)


def _coderouter_models(path: Path, max_bytes: int) -> tuple[str, ...]:
    payload = exact_object(
        load_json(path, max_bytes=max_bytes),
        required={"models"},
        label="CodeRouterBench models",
    )
    models = []
    for index, value in enumerate(require_array(payload["models"], "models")):
        row = exact_object(
            value,
            required={"model", "provider", "tier", "input_per_1m", "output_per_1m"},
            optional={"_note"},
            label=f"CodeRouterBench models[{index}]",
        )
        models.append(string(row["model"], "CodeRouterBench model"))
        string(row["provider"], "CodeRouterBench provider")
        string(row["tier"], "CodeRouterBench tier")
        number(row["input_per_1m"], "CodeRouterBench input price")
        number(row["output_per_1m"], "CodeRouterBench output price")
        if "_note" in row:
            string(row["_note"], "CodeRouterBench note")
    no_duplicate(models, "CodeRouterBench model manifest")
    return tuple(models)


def _coderouter_sources(root: Path, artifacts: dict[str, Any]) -> tuple[
    tuple[str, ...],
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, str]]],
]:
    models = _coderouter_models(
        required_file(root, artifacts["models"]), artifacts["models"].max_bytes
    )
    arms = arm_map(models)
    tasks: dict[str, dict[str, Any]] = {}
    for value in iter_jsonl(required_file(root, artifacts["tasks"])):
        row = exact_object(
            value,
            required={"dimension", "source_split", "split", "task_id"},
            label="CodeRouterBench task",
        )
        task_id = string(row["task_id"], "CodeRouterBench task_id")
        if task_id in tasks:
            raise NormalizationError("CodeRouterBench tasks repeat task_id")
        tasks[task_id] = row
    decisions: dict[str, dict[str, Any]] = {}
    for value in iter_jsonl(required_file(root, artifacts["decisions"])):
        row = exact_object(
            value,
            required={
                "chosen_model",
                "dimension",
                "matched_key",
                "matched_mode",
                "task_id",
                "voter",
            },
            label="CodeRouterBench decision",
        )
        task_id = string(row["task_id"], "CodeRouterBench decision task_id")
        if task_id in decisions:
            raise NormalizationError("CodeRouterBench decisions repeat task_id")
        decisions[task_id] = row
    results: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in iter_csv(
        required_file(root, artifacts["results"]), _CODEROUTER_RESULT_HEADER
    ):
        results[string(row["task_id"], "result task_id")].append(row)
    if not tasks or set(tasks) != set(decisions) or set(tasks) != set(results):
        raise NormalizationError(
            "CodeRouterBench tasks, decisions, and outcomes do not align"
        )
    return models, arms, tasks, decisions, dict(results)


def _coderouter_case(
    task_id: str,
    task: dict[str, Any],
    decision: dict[str, Any],
    rows: list[dict[str, str]],
    models: tuple[str, ...],
    arms: dict[str, str],
    descriptor: BenchmarkNormalizerDescriptor,
) -> tuple[CaseVisible, CaseGrading, NormalizedDecision, list[NormalizedOutcome]]:
    chosen = string(decision["chosen_model"], "chosen_model")
    if chosen not in arms:
        raise NormalizationError("CodeRouterBench decision selects undeclared model")
    if {row["model"] for row in rows} != set(models) or len(rows) != len(models):
        raise NormalizationError(
            "CodeRouterBench outcomes are not a dense model matrix"
        )
    dimension = string(task["dimension"], "dimension")
    visible, grading = text_case(
        f"coderouterbench:{task_id}",
        f"CodeRouterBench task {task_id} ({dimension})",
        descriptor=descriptor,
        tags=("coderouterbench", dimension),
        expected_route=arms[chosen],
    )
    normalized_decision = NormalizedDecision(
        case_id=visible.id,
        selected_arm_id=arms[chosen],
        selection_status="selected",
        selection_method=string(decision["voter"], "voter"),
        success=True,
        source_record_digest=native_digest(decision),
    )
    outcomes = []
    for row in sorted(rows, key=lambda item: item["model"]):
        if row["split"] != task["split"] or row["dimension"] != dimension:
            raise NormalizationError(
                "CodeRouterBench task metadata drifted across files"
            )
        score = number(row["score"], "score", maximum=1)
        outcomes.append(
            NormalizedOutcome(
                case_id=visible.id,
                arm_id=arms[row["model"]],
                quality=score,
                success=score > 0,
                latency_ms=number(row["latency_ms"], "latency_ms"),
                input_tokens=integer(row["input_tokens"], "input_tokens"),
                output_tokens=integer(row["output_tokens"], "output_tokens"),
                runtime_cost_usd=number(row["cost_usd"], "cost_usd"),
                grader_id="coderouterbench.score",
                grader_revision=descriptor.export_schema_id,
                split=string(row["split"], "split"),
                source_record_digest=native_digest(row),
            )
        )
    return visible, grading, normalized_decision, outcomes


def normalize_coderouterbench(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    artifacts = {item.id: item for item in descriptor.required_artifacts}
    models, arms, tasks, decisions_by_task, results_by_task = _coderouter_sources(
        root, artifacts
    )
    visible = []
    grading = []
    decisions = []
    outcomes = []
    for task_id in sorted(tasks):
        case_visible, case_grading, decision, case_outcomes = _coderouter_case(
            task_id,
            tasks[task_id],
            decisions_by_task[task_id],
            results_by_task[task_id],
            models,
            arms,
            descriptor,
        )
        visible.append(case_visible)
        grading.append(case_grading)
        decisions.append(decision)
        outcomes.extend(case_outcomes)
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        outcomes=tuple(outcomes),
        decisions=tuple(decisions),
        arm_ids=tuple(arms.values()),
        split_protocol="Frozen ID test task order with complete declared-model outcomes.",
    )
