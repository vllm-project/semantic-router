"""Strict parsers for native dense outcome matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    exact_object,
    integer,
    iter_csv,
    iter_jsonl,
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
    load_model_manifest,
    native_digest,
    text_case,
)
from cli.evaluation.suite_contract import NormalizedOutcome

_LLMROUTER_TOP_KEYS = {
    "performance",
    "time_taken",
    "prompt_tokens",
    "completion_tokens",
    "cost",
    "counts",
    "model_name",
    "dataset_name",
    "split",
    "demo",
    "extra_metrics",
    "data_fingerprint",
    "records",
}
_LLMROUTER_RECORD_KEYS = {
    "index",
    "origin_query",
    "prompt",
    "prompt_tokens",
    "completion_tokens",
    "cost",
    "score",
    "prediction",
    "ground_truth",
    "raw_output",
    "extra_fields",
}
_MIN_DENSE_POOL_MODELS = 2


def _llmrouter_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for value in iter_jsonl(path):
        row = exact_object(
            value,
            required=_LLMROUTER_TOP_KEYS,
            label="LLMRouterBench result document",
        )
        if row["demo"] is not False:
            raise NormalizationError(
                "LLMRouterBench demo results are not qualified data"
            )
        string(row["model_name"], "LLMRouterBench model_name")
        string(row["dataset_name"], "LLMRouterBench dataset_name")
        string(row["split"], "LLMRouterBench split")
        string(row["data_fingerprint"], "LLMRouterBench data_fingerprint")
        if not isinstance(row["records"], list) or not row["records"]:
            raise NormalizationError("LLMRouterBench result document has no records")
        documents.append(row)
    if len(documents) < _MIN_DENSE_POOL_MODELS:
        raise NormalizationError("LLMRouterBench export requires at least two models")
    return documents


def _llmrouter_record_matrix(
    documents: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    records_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for document in documents:
        by_index: dict[str, dict[str, Any]] = {}
        for position, value in enumerate(document["records"]):
            record = exact_object(
                value,
                required=_LLMROUTER_RECORD_KEYS,
                label=f"LLMRouterBench records[{position}]",
            )
            source_index = str(record["index"])
            if source_index in by_index:
                raise NormalizationError("LLMRouterBench repeats a record index")
            string(record["origin_query"], "LLMRouterBench origin_query")
            string(record["prompt"], "LLMRouterBench prompt")
            number(record["score"], "LLMRouterBench score", maximum=1)
            number(record["cost"], "LLMRouterBench record cost")
            integer(record["prompt_tokens"], "LLMRouterBench prompt tokens")
            integer(record["completion_tokens"], "LLMRouterBench completion tokens")
            by_index[source_index] = record
        records_by_model[document["model_name"]] = by_index
    return records_by_model


def normalize_llmrouterbench(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    requirement = descriptor.required_artifacts[0]
    documents = _llmrouter_documents(required_file(root, requirement))
    models = tuple(string(row["model_name"], "model_name") for row in documents)
    arms = arm_map(models)
    if len(models) != len(arms):
        raise NormalizationError("LLMRouterBench repeats a model result document")
    identities = {
        (row["dataset_name"], row["split"], row["data_fingerprint"])
        for row in documents
    }
    if len(identities) != 1:
        raise NormalizationError(
            "LLMRouterBench model documents do not share one split"
        )
    records_by_model = _llmrouter_record_matrix(documents)
    index_sets = {frozenset(rows) for rows in records_by_model.values()}
    if len(index_sets) != 1:
        raise NormalizationError("LLMRouterBench model records are not dense")
    source_indices = sorted(next(iter(index_sets)))
    visible = []
    grading = []
    outcomes = []
    dataset_name, split, _ = identities.pop()
    for source_index in source_indices:
        aligned = [records_by_model[model][source_index] for model in sorted(models)]
        prompts = {string(row["origin_query"], "origin_query") for row in aligned}
        ground_truths = {str(row["ground_truth"]) for row in aligned}
        if len(prompts) != 1 or len(ground_truths) != 1:
            raise NormalizationError(
                "LLMRouterBench aligned records disagree on labels"
            )
        case_visible, case_grading = text_case(
            f"llmrouterbench:{dataset_name}:{split}:{source_index}",
            prompts.pop(),
            descriptor=descriptor,
            tags=("llmrouterbench", str(dataset_name), str(split)),
            expected_answer=ground_truths.pop(),
        )
        visible.append(case_visible)
        grading.append(case_grading)
        for model in sorted(models):
            row = records_by_model[model][source_index]
            quality = number(row["score"], "score", maximum=1)
            outcomes.append(
                NormalizedOutcome(
                    case_id=case_visible.id,
                    arm_id=arms[model],
                    success=quality > 0,
                    quality=quality,
                    input_tokens=integer(row["prompt_tokens"], "prompt tokens"),
                    output_tokens=integer(
                        row["completion_tokens"], "completion tokens"
                    ),
                    runtime_cost_usd=number(row["cost"], "cost"),
                    grader_id="llmrouterbench.score",
                    grader_revision=descriptor.export_schema_id,
                    split=str(split),
                    source_record_digest=native_digest(row),
                )
            )
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        outcomes=tuple(outcomes),
        arm_ids=tuple(arms.values()),
        split_protocol="One fingerprinted dataset/split with aligned native result records.",
    )


def normalize_routerbench(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    artifacts = {item.id: item for item in descriptor.required_artifacts}
    model_requirement = artifacts["models"]
    models = load_model_manifest(
        required_file(root, model_requirement), max_bytes=model_requirement.max_bytes
    )
    arms = arm_map(models)
    header = ["sample_id", "prompt", "eval_name"]
    for model in models:
        header.extend((model, f"{model}|model_response", f"{model}|total_cost"))
    visible = []
    grading = []
    outcomes = []
    seen: set[str] = set()
    for row in iter_csv(required_file(root, artifacts["wide-data"]), header):
        sample_id = string(row["sample_id"], "RouterBench sample_id")
        if sample_id in seen:
            raise NormalizationError("RouterBench repeats sample_id")
        seen.add(sample_id)
        eval_name = string(row["eval_name"], "RouterBench eval_name")
        case_visible, case_grading = text_case(
            f"routerbench:{eval_name}:{sample_id}",
            string(row["prompt"], "RouterBench prompt"),
            descriptor=descriptor,
            tags=("routerbench", eval_name),
        )
        visible.append(case_visible)
        grading.append(case_grading)
        for model in models:
            quality = number(row[model], f"RouterBench {model} score", maximum=1)
            string(
                row[f"{model}|model_response"],
                f"RouterBench {model} response",
                allow_empty=True,
            )
            outcomes.append(
                NormalizedOutcome(
                    case_id=case_visible.id,
                    arm_id=arms[model],
                    success=quality > 0,
                    quality=quality,
                    runtime_cost_usd=number(
                        row[f"{model}|total_cost"], f"RouterBench {model} cost"
                    ),
                    grader_id=f"routerbench.{eval_name}",
                    grader_revision=descriptor.export_schema_id,
                    split=eval_name,
                    source_record_digest=native_digest(
                        {
                            "sample_id": sample_id,
                            "model": model,
                            "score": row[model],
                            "response": row[f"{model}|model_response"],
                            "cost": row[f"{model}|total_cost"],
                        }
                    ),
                )
            )
    if not visible:
        raise NormalizationError("RouterBench wide CSV is empty")
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        outcomes=tuple(outcomes),
        arm_ids=tuple(arms.values()),
        split_protocol="Native converted wide table; declared model columns are complete.",
    )
