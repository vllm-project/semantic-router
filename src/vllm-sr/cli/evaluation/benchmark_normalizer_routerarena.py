"""Strict normalization for RouterArena decision and robustness artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    boolean,
    exact_object,
    integer,
    load_json,
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
    opaque_id,
    text_case,
)
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.suite_contract import (
    NormalizedDecision,
    NormalizedOutcome,
    NormalizedPerturbation,
)


def _routerarena_result(value: Any, label: str) -> dict[str, Any]:
    row = exact_object(
        value,
        required={"generated_answer", "success", "token_usage", "provider", "error"},
        label=label,
    )
    answer = row["generated_answer"]
    if answer is not None:
        string(answer, f"{label}.generated_answer", allow_empty=True)
    boolean(row["success"], f"{label}.success")
    string(row["provider"], f"{label}.provider")
    if row["error"] is not None:
        string(row["error"], f"{label}.error")
    exact_object(
        row["token_usage"],
        required={"input_tokens", "output_tokens", "total_tokens"},
        label=f"{label}.token_usage",
    )
    return row


def _routerarena_predictions(
    path: Path, max_bytes: int
) -> dict[str, list[dict[str, Any]]]:
    payload = require_array(
        load_json(path, max_bytes=max_bytes),
        "RouterArena predictions",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, value in enumerate(payload):
        row = exact_object(
            value,
            required={
                "global index",
                "prompt",
                "prediction",
                "generated_result",
                "cost",
                "accuracy",
            },
            optional={"for_optimality"},
            label=f"RouterArena predictions[{index}]",
        )
        source_id = string(row["global index"], "RouterArena global index")
        string(row["prompt"], "RouterArena prompt")
        string(row["prediction"], "RouterArena prediction")
        number(row["accuracy"], "RouterArena accuracy", maximum=1)
        number(row["cost"], "RouterArena cost")
        _routerarena_result(row["generated_result"], "RouterArena generated_result")
        if "for_optimality" in row:
            boolean(row["for_optimality"], "RouterArena for_optimality")
        grouped[source_id].append(row)
    if not grouped:
        raise NormalizationError("RouterArena prediction file is empty")
    return dict(grouped)


def _routerarena_robustness_predictions(
    path: Path, max_bytes: int
) -> dict[str, dict[str, Any]]:
    payload = require_array(
        load_json(path, max_bytes=max_bytes),
        "RouterArena robustness predictions",
    )
    by_source: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(payload):
        row = exact_object(
            value,
            required={
                "global index",
                "prompt",
                "prediction",
                "generated_result",
                "cost",
                "accuracy",
            },
            optional={"for_optimality"},
            label=f"RouterArena robustness predictions[{index}]",
        )
        source_id = string(row["global index"], "RouterArena robustness index")
        if source_id in by_source:
            raise NormalizationError("RouterArena robustness repeats a global index")
        string(row["prompt"], "RouterArena robustness prompt")
        string(row["prediction"], "RouterArena robustness prediction")
        if row["generated_result"] is not None:
            _routerarena_result(
                row["generated_result"], "RouterArena robustness generated_result"
            )
        if row["accuracy"] is not None:
            number(row["accuracy"], "RouterArena robustness accuracy", maximum=1)
        if row["cost"] is not None:
            number(row["cost"], "RouterArena robustness cost")
        if bool(row.get("for_optimality", False)):
            raise NormalizationError(
                "RouterArena robustness cannot contain optimality-only rows"
            )
        by_source[source_id] = row
    if not by_source:
        raise NormalizationError("RouterArena robustness prediction file is empty")
    return by_source


def _routerarena_source_case(
    source_id: str,
    rows: list[dict[str, Any]],
    arms: dict[str, str],
    descriptor: BenchmarkNormalizerDescriptor,
) -> tuple[
    CaseVisible,
    CaseGrading,
    NormalizedDecision,
    list[NormalizedOutcome],
    tuple[str, str, dict[str, Any]],
]:
    if {string(row["prediction"], "prediction") for row in rows} != set(arms):
        raise NormalizationError(
            "RouterArena optimality export must contain every model for every case"
        )
    if len(rows) != len(arms):
        raise NormalizationError("RouterArena case repeats a model outcome")
    prompts = {string(row["prompt"], "prompt") for row in rows}
    if len(prompts) != 1:
        raise NormalizationError("RouterArena case has inconsistent prompts")
    selected = [row for row in rows if not bool(row.get("for_optimality", False))]
    if len(selected) != 1:
        raise NormalizationError(
            "RouterArena case requires exactly one non-optimality router decision"
        )
    selected_row = selected[0]
    selected_model = string(selected_row["prediction"], "prediction")
    visible, grading = text_case(
        f"routerarena:{source_id}",
        prompts.pop(),
        descriptor=descriptor,
        tags=("routerarena",),
    )
    decision = NormalizedDecision(
        case_id=visible.id,
        selected_arm_id=arms[selected_model],
        selection_status="selected",
        selection_method="routerarena.prediction",
        success=boolean(
            selected_row["generated_result"]["success"], "generated success"
        ),
        source_record_digest=native_digest(selected_row),
    )
    outcomes = []
    for row in sorted(rows, key=lambda item: string(item["prediction"], "prediction")):
        result = row["generated_result"]
        usage = result["token_usage"]
        outcomes.append(
            NormalizedOutcome(
                case_id=visible.id,
                arm_id=arms[string(row["prediction"], "prediction")],
                success=boolean(result["success"], "generated success"),
                quality=number(row["accuracy"], "accuracy", maximum=1),
                input_tokens=integer(usage["input_tokens"], "input tokens"),
                output_tokens=integer(usage["output_tokens"], "output tokens"),
                runtime_cost_usd=number(row["cost"], "cost"),
                grader_id="routerarena.accuracy",
                grader_revision=descriptor.export_schema_id,
                split="routerarena",
                source_record_digest=native_digest(row),
            )
        )
    source = (visible.id, arms[selected_model], selected_row)
    return visible, grading, decision, outcomes, source


def _routerarena_perturbation_case(
    source_id: str,
    row: dict[str, Any],
    source: tuple[str, str, dict[str, Any]],
    arms: dict[str, str],
    descriptor: BenchmarkNormalizerDescriptor,
    native_pair_count: int,
) -> tuple[CaseVisible, CaseGrading, NormalizedDecision, NormalizedPerturbation]:
    source_case_id, _, source_row = source
    prediction = string(row["prediction"], "RouterArena robustness prediction")
    if prediction not in arms:
        raise NormalizationError(
            "RouterArena robustness selects an undeclared full-split model"
        )
    prompt = string(row["prompt"], "RouterArena robustness prompt")
    source_prompt = string(source_row["prompt"], "RouterArena source prompt")
    if prompt == source_prompt:
        raise NormalizationError(
            "RouterArena robustness prompt must differ from its source prompt"
        )
    visible, grading = text_case(
        f"routerarena-robustness:{source_id}",
        prompt,
        descriptor=descriptor,
        tags=("routerarena", "routerarena-robustness"),
    )
    visible = visible.model_copy(update={"track_ids": ("routing",)})
    decision = NormalizedDecision(
        case_id=visible.id,
        selected_arm_id=arms[prediction],
        selection_status="selected",
        selection_method="routerarena.robustness-prediction",
        success=True,
        source_record_digest=native_digest(row),
    )
    dataset = source_id.rsplit("_", maxsplit=1)[0].casefold()
    perturbation = NormalizedPerturbation(
        pair_id=opaque_id("pair", "routerarena", source_id),
        source_case_id=source_case_id,
        perturbed_case_id=visible.id,
        relation="invariant",
        slice_ids=("routerarena:paraphrase", f"dataset:{dataset}"),
        native_pair_count=native_pair_count,
        source_record_digest=native_digest({"source": source_row, "perturbed": row}),
    )
    return visible, grading, decision, perturbation


def normalize_routerarena(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    artifacts = {item.id: item for item in descriptor.required_artifacts}
    prediction_requirement = artifacts["predictions"]
    grouped = _routerarena_predictions(
        required_file(root, prediction_requirement),
        prediction_requirement.max_bytes,
    )
    robustness_requirement = artifacts["robustness-predictions"]
    robustness_by_source = _routerarena_robustness_predictions(
        required_file(root, robustness_requirement),
        robustness_requirement.max_bytes,
    )
    if not set(robustness_by_source).issubset(grouped):
        raise NormalizationError(
            "RouterArena robustness references an unknown full-split index"
        )
    models = {
        string(row["prediction"], "prediction")
        for rows in grouped.values()
        for row in rows
    }
    arms = arm_map(models)
    visible = []
    grading = []
    decisions = []
    outcomes = []
    source_cases: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for source_id in sorted(grouped):
        case_visible, case_grading, decision, case_outcomes, source = (
            _routerarena_source_case(
                source_id,
                grouped[source_id],
                arms,
                descriptor,
            )
        )
        visible.append(case_visible)
        grading.append(case_grading)
        decisions.append(decision)
        outcomes.extend(case_outcomes)
        source_cases[source_id] = source
    perturbations = []
    for source_id in sorted(robustness_by_source):
        case_visible, case_grading, decision, perturbation = (
            _routerarena_perturbation_case(
                source_id,
                robustness_by_source[source_id],
                source_cases[source_id],
                arms,
                descriptor,
                len(robustness_by_source),
            )
        )
        visible.append(case_visible)
        grading.append(case_grading)
        decisions.append(decision)
        perturbations.append(perturbation)
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        outcomes=tuple(outcomes),
        decisions=tuple(decisions),
        perturbations=tuple(perturbations),
        arm_ids=tuple(arms.values()),
        split_protocol=(
            "RouterArena global-index order; complete full-split model matrix plus "
            "native global-index full/robustness invariant pairs."
        ),
    )
