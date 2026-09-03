"""Strict parsers for scenario, fusion, and model-budget matrices."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    integer,
    iter_csv,
    number,
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
from cli.evaluation.suite_contract import NormalizedOutcome

_XROUTE_HEADER = (
    "task_name",
    "query",
    "ground_truth",
    "metric",
    "choices",
    "task_id",
    "model_name",
    "response",
    "token_num",
    "input_tokens",
    "output_tokens",
    "response_time",
    "performance",
    "embedding_id",
)


def normalize_xroutebench(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    rows = list(
        iter_csv(required_file(root, descriptor.required_artifacts[0]), _XROUTE_HEADER)
    )
    if not rows:
        raise NormalizationError("xRouteBench standardized CSV is empty")
    models = {string(row["model_name"], "xRouteBench model_name") for row in rows}
    arms = arm_map(models)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            string(row["task_name"], "xRouteBench task_name"),
            string(row["task_id"], "xRouteBench task_id"),
        )
        grouped[key].append(row)
    visible = []
    grading = []
    outcomes = []
    for (task_name, task_id), aligned in sorted(grouped.items()):
        if {row["model_name"] for row in aligned} != models or len(aligned) != len(
            models
        ):
            raise NormalizationError("xRouteBench case is not a dense model matrix")
        prompts = {string(row["query"], "xRouteBench query") for row in aligned}
        answers = {
            string(row["ground_truth"], "xRouteBench ground_truth", allow_empty=True)
            for row in aligned
        }
        metrics = {string(row["metric"], "xRouteBench metric") for row in aligned}
        if len(prompts) != 1 or len(answers) != 1 or len(metrics) != 1:
            raise NormalizationError("xRouteBench aligned rows disagree on task fields")
        case_visible, case_grading = text_case(
            f"xroutebench:{task_name}:{task_id}",
            prompts.pop(),
            descriptor=descriptor,
            tags=("xroutebench", task_name),
            expected_answer=answers.pop(),
        )
        visible.append(case_visible)
        grading.append(case_grading)
        metric_name = metrics.pop()
        for row in sorted(aligned, key=lambda item: item["model_name"]):
            quality = number(row["performance"], "xRouteBench performance", maximum=1)
            outcomes.append(
                NormalizedOutcome(
                    case_id=case_visible.id,
                    arm_id=arms[row["model_name"]],
                    success=quality > 0,
                    quality=quality,
                    latency_ms=number(row["response_time"], "response_time") * 1000,
                    input_tokens=integer(row["input_tokens"], "input_tokens"),
                    output_tokens=integer(row["output_tokens"], "output_tokens"),
                    grader_id=f"xroutebench.{metric_name}",
                    grader_revision=descriptor.export_schema_id,
                    split="test",
                    source_record_digest=native_digest(row),
                )
            )
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        outcomes=tuple(outcomes),
        arm_ids=tuple(arms.values()),
        split_protocol="One standardized xRouteBench scenario test export; dense candidate set.",
    )


_FUSION_HEADER = (
    "task_name",
    "task_id",
    "task_description",
    "task_description_embedding",
    "query",
    "query_embedding",
    "ground_truth",
    "metric",
    "llm",
    "input_price",
    "output_price",
    "input_tokens_num",
    "output_tokens_num",
    "performance",
    "cost",
    "response",
    "llm_description",
)


def normalize_fusionfactory(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    rows = list(
        iter_csv(required_file(root, descriptor.required_artifacts[0]), _FUSION_HEADER)
    )
    if not rows:
        raise NormalizationError("FusionFactory aligned CSV is empty")
    actions = {string(row["llm"], "FusionFactory llm") for row in rows}
    arms = arm_map(actions)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_name"], row["task_id"])].append(row)
    visible = []
    grading = []
    outcomes = []
    for (task_name, task_id), aligned in sorted(grouped.items()):
        if {row["llm"] for row in aligned} != actions or len(aligned) != len(actions):
            raise NormalizationError("FusionFactory case is not a dense action matrix")
        prompts = {string(row["query"], "FusionFactory query") for row in aligned}
        answers = {
            string(row["ground_truth"], "FusionFactory ground_truth", allow_empty=True)
            for row in aligned
        }
        metrics = {string(row["metric"], "FusionFactory metric") for row in aligned}
        if len(prompts) != 1 or len(answers) != 1 or len(metrics) != 1:
            raise NormalizationError(
                "FusionFactory aligned rows disagree on task fields"
            )
        case_visible, case_grading = text_case(
            f"fusionfactory:{task_name}:{task_id}",
            prompts.pop(),
            descriptor=descriptor,
            tags=("fusionfactory", task_name),
            expected_answer=answers.pop(),
        )
        visible.append(case_visible)
        grading.append(case_grading)
        metric_name = metrics.pop()
        for row in sorted(aligned, key=lambda item: item["llm"]):
            quality = number(row["performance"], "FusionFactory performance", maximum=1)
            input_tokens = integer(row["input_tokens_num"], "input_tokens_num")
            output_tokens = integer(row["output_tokens_num"], "output_tokens_num")
            runtime_cost = (
                input_tokens * number(row["input_price"], "input_price")
                + output_tokens * number(row["output_price"], "output_price")
            ) / 1_000_000
            outcomes.append(
                NormalizedOutcome(
                    case_id=case_visible.id,
                    arm_id=arms[row["llm"]],
                    action_id=arms[row["llm"]],
                    success=quality > 0,
                    quality=quality,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    runtime_cost_usd=runtime_cost,
                    grader_id=f"fusionfactory.{metric_name}",
                    grader_revision=descriptor.export_schema_id,
                    split="aligned",
                    source_record_digest=native_digest(row),
                )
            )
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        outcomes=tuple(outcomes),
        arm_ids=tuple(arms.values()),
        split_protocol="Aligned native query/action rows; full declared action matrix.",
    )


_R2_HEADER = (
    "case_id",
    "query",
    "model",
    "budget_tokens",
    "score",
    "token_count",
    "split",
)
_R2_BUDGETS = (10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000)


def normalize_r2_router(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    rows = list(
        iter_csv(required_file(root, descriptor.required_artifacts[0]), _R2_HEADER)
    )
    if not rows:
        raise NormalizationError("R2-Bench model-budget CSV is empty")
    models = {string(row["model"], "R2-Bench model") for row in rows}
    arms = arm_map(models)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[string(row["case_id"], "R2-Bench case_id")].append(row)
    visible = []
    grading = []
    outcomes = []
    expected_cells = {(model, budget) for model in models for budget in _R2_BUDGETS}
    for source_id, aligned in sorted(grouped.items()):
        cells = {
            (row["model"], integer(row["budget_tokens"], "budget_tokens", minimum=1))
            for row in aligned
        }
        if cells != expected_cells or len(aligned) != len(expected_cells):
            raise NormalizationError(
                "R2-Bench case lacks the fixed model-budget tensor"
            )
        prompts = {string(row["query"], "R2-Bench query") for row in aligned}
        splits = {string(row["split"], "R2-Bench split") for row in aligned}
        if len(prompts) != 1 or len(splits) != 1:
            raise NormalizationError("R2-Bench aligned rows disagree on case metadata")
        case_visible, case_grading = text_case(
            f"r2-router:{source_id}",
            prompts.pop(),
            descriptor=descriptor,
            tags=("r2-router",),
        )
        visible.append(case_visible)
        grading.append(case_grading)
        split = splits.pop()
        for row in sorted(
            aligned, key=lambda item: (item["model"], int(item["budget_tokens"]))
        ):
            budget = integer(row["budget_tokens"], "budget_tokens", minimum=1)
            quality = number(row["score"], "R2-Bench score", maximum=1)
            outcomes.append(
                NormalizedOutcome(
                    case_id=case_visible.id,
                    arm_id=arms[row["model"]],
                    action_id=arms[row["model"]],
                    budget_tokens=budget,
                    success=quality > 0,
                    quality=quality,
                    output_tokens=integer(row["token_count"], "token_count"),
                    grader_id="r2-router.score",
                    grader_revision=descriptor.export_schema_id,
                    split=split,
                    source_record_digest=native_digest(row),
                )
            )
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        outcomes=tuple(outcomes),
        arm_ids=tuple(arms.values()),
        split_protocol="Fixed 15-budget sweep for every declared model and case.",
    )
