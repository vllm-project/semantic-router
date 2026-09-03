from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.method_contract_v2 import (
    EVALUATION_METHOD_CONTRACT_VERSION,
    R2_COMPOUND_MODEL_BUDGET_PLUGIN,
    ActionRef,
    CompoundModelBudgetOutcome,
    EvaluationMethodPlugin,
    SliceRef,
    reduce_compound_model_budget,
)
from cli.evaluation.method_registry_v2 import (
    METHOD_PLUGINS,
    method_plugin_for_benchmark,
)
from cli.evaluation.metric_compound_model_budget import r2_compound_metrics
from cli.evaluation.research_benchmark_inventory import RESEARCH_BENCHMARKS
from pydantic import ValidationError

_METHOD_CONFORMANCE_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "evaluation_method_contract_v2_conformance.v1.json"
)


def _decode_json_pointer(pointer: str) -> tuple[str, ...]:
    if not pointer.startswith("/"):
        raise AssertionError(
            f"JSON Pointer must identify a descriptor field: {pointer!r}"
        )
    tokens: list[str] = []
    for encoded_token in pointer[1:].split("/"):
        index = 0
        while index < len(encoded_token):
            if encoded_token[index] != "~":
                index += 1
                continue
            if index + 1 >= len(encoded_token) or encoded_token[index + 1] not in "01":
                raise AssertionError(f"JSON Pointer has an invalid escape: {pointer!r}")
            index += 2
        tokens.append(encoded_token.replace("~1", "/").replace("~0", "~"))
    return tuple(tokens)


def _json_array_index(token: str, length: int, pointer: str) -> int:
    if (
        not token.isascii()
        or not token.isdecimal()
        or (len(token) > 1 and token.startswith("0"))
    ):
        raise AssertionError(f"JSON Pointer has an invalid array index: {pointer!r}")
    index = int(token)
    if index >= length:
        raise AssertionError(f"JSON Pointer array index is out of bounds: {pointer!r}")
    return index


def _remove_json_pointer(document: Any, pointer: str) -> None:
    tokens = _decode_json_pointer(pointer)
    parent = document
    for token in tokens[:-1]:
        if isinstance(parent, dict):
            if token not in parent:
                raise AssertionError(f"JSON Pointer field does not exist: {pointer!r}")
            parent = parent[token]
        elif isinstance(parent, list):
            parent = parent[_json_array_index(token, len(parent), pointer)]
        else:
            raise AssertionError(f"JSON Pointer traverses a scalar: {pointer!r}")

    final_token = tokens[-1]
    if isinstance(parent, dict):
        if final_token not in parent:
            raise AssertionError(f"JSON Pointer field does not exist: {pointer!r}")
        del parent[final_token]
    elif isinstance(parent, list):
        del parent[_json_array_index(final_token, len(parent), pointer)]
    else:
        raise AssertionError(f"JSON Pointer targets a scalar: {pointer!r}")


def _outcomes() -> tuple[CompoundModelBudgetOutcome, ...]:
    return (
        CompoundModelBudgetOutcome(
            case_id="case-a",
            action=ActionRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="small"
            ),
            budget=100,
            score=0.4,
            slice_refs=(
                SliceRef(schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="all"),
            ),
        ),
        CompoundModelBudgetOutcome(
            case_id="case-a",
            action=ActionRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="small"
            ),
            budget=200,
            score=0.6,
            slice_refs=(
                SliceRef(schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="all"),
            ),
        ),
        CompoundModelBudgetOutcome(
            case_id="case-a",
            action=ActionRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="large"
            ),
            budget=100,
            score=0.6,
            slice_refs=(
                SliceRef(schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="all"),
            ),
        ),
        CompoundModelBudgetOutcome(
            case_id="case-a",
            action=ActionRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="large"
            ),
            budget=200,
            score=0.8,
            slice_refs=(
                SliceRef(schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="all"),
            ),
        ),
        CompoundModelBudgetOutcome(
            case_id="case-b",
            action=ActionRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="small"
            ),
            budget=100,
            score=0.2,
            slice_refs=(
                SliceRef(schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="all"),
            ),
        ),
        CompoundModelBudgetOutcome(
            case_id="case-b",
            action=ActionRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="small"
            ),
            budget=200,
            score=0.4,
            slice_refs=(
                SliceRef(schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="all"),
            ),
        ),
        CompoundModelBudgetOutcome(
            case_id="case-b",
            action=ActionRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="large"
            ),
            budget=100,
            score=0.4,
            slice_refs=(
                SliceRef(schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="all"),
            ),
        ),
        CompoundModelBudgetOutcome(
            case_id="case-b",
            action=ActionRef(
                schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="large"
            ),
            budget=200,
            score=0.6,
            slice_refs=(
                SliceRef(schema_version=EVALUATION_METHOD_CONTRACT_VERSION, id="all"),
            ),
        ),
    )


def test_r2_compound_model_budget_preserves_action_identity_and_shared_curve() -> None:
    report = reduce_compound_model_budget(_outcomes())

    assert report.method == R2_COMPOUND_MODEL_BUDGET_PLUGIN
    assert tuple(action.id for action in report.action_refs) == ("large", "small")
    assert [
        (point.action.id, point.budget) for point in report.raw_shared_domain_curve
    ] == [
        ("large", 100),
        ("large", 200),
        ("small", 100),
        ("small", 200),
    ]
    assert [
        point.mean_score for point in report.raw_shared_domain_curve
    ] == pytest.approx([0.5, 0.7, 0.3, 0.5])
    assert report.audc == pytest.approx(100.0)
    assert report.nauc == pytest.approx(0.5)
    assert report.peak == pytest.approx(0.7)
    assert report.qnc == pytest.approx(0.6)
    assert report.missing_case_action_budget_cells == 0


def test_r2_compound_model_budget_fails_closed_on_missing_or_duplicate_cells() -> None:
    rows = _outcomes()
    with pytest.raises(ValueError, match="exact shared"):
        reduce_compound_model_budget(rows[:-1])
    with pytest.raises(ValueError, match="duplicate case x action x budget"):
        reduce_compound_model_budget((*rows, rows[0]))


def test_r2_execution_records_reduce_without_generic_model_pool_semantics() -> None:
    records = [
        ExecutionRecord(
            id=f"r2-{case}-{action}-{budget}",
            track_id="model_pool",
            case_id=case,
            attempt_id=f"attempt-{case}-{action}-{budget}",
            status="succeeded",
            method_id="r2.compound-model-budget.v2",
            action_id=action,
            budget_tokens=budget,
            slice_ids=("all",),
            success=True,
            quality=score,
        )
        for case, action, budget, score in (
            ("case-a", "small", 100, 0.4),
            ("case-a", "small", 200, 0.6),
            ("case-a", "large", 100, 0.6),
            ("case-a", "large", 200, 0.8),
        )
    ]
    metrics = {metric.id: metric.value for metric in r2_compound_metrics(records)}
    assert metrics["r2.compound_model_budget.audc"] == pytest.approx(120.0)
    assert metrics["r2.compound_model_budget.nauc"] == pytest.approx(0.6)


def test_all_benchmark_methods_have_one_explicit_v2_declaration() -> None:
    assert len(METHOD_PLUGINS) == 13
    assert len(RESEARCH_BENCHMARKS) == 13
    assert method_plugin_for_benchmark("routejudge-orbit").status == "blocked"
    assert method_plugin_for_benchmark("routereval").status == "blocked"
    assert sum(plugin.status == "exploratory-import" for plugin in METHOD_PLUGINS) == 8
    assert sum(plugin.status == "data-required" for plugin in METHOD_PLUGINS) == 3
    assert all(plugin.evidence_ceiling == "E0" for plugin in METHOD_PLUGINS)
    assert all(plugin.native_parity != "native" for plugin in METHOD_PLUGINS)
    assert method_plugin_for_benchmark("r2-router") == R2_COMPOUND_MODEL_BUDGET_PLUGIN
    assert R2_COMPOUND_MODEL_BUDGET_PLUGIN.applicable_tracks == (
        "routing",
        "model_pool",
        "joint",
        "capacity",
    )
    assert R2_COMPOUND_MODEL_BUDGET_PLUGIN.live_tracks == ()
    assert method_plugin_for_benchmark("routerarena").applicable_tracks == (
        "routing",
        "model_pool",
        "joint",
    )


def test_method_v2_admission_matches_shared_cross_language_conformance() -> None:
    corpus = json.loads(_METHOD_CONFORMANCE_FIXTURE.read_text(encoding="utf-8"))
    assert set(corpus) == {
        "schema_version",
        "method_contract_version",
        "base_descriptor",
        "cases",
    }
    assert corpus["schema_version"] == "evaluation-method-conformance.v1"
    assert corpus["method_contract_version"] == EVALUATION_METHOD_CONTRACT_VERSION
    assert isinstance(corpus["base_descriptor"], dict) and corpus["base_descriptor"]
    assert isinstance(corpus["cases"], list)
    case_ids = [case["id"] for case in corpus["cases"]]
    assert case_ids
    assert len(case_ids) == len(set(case_ids))

    for case in corpus["cases"]:
        assert set(case) == {"id", "expected_valid", "remove_fields", "overrides"}
        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["expected_valid"], bool)
        assert isinstance(case["remove_fields"], list)
        assert all(isinstance(pointer, str) for pointer in case["remove_fields"])
        assert isinstance(case["overrides"], dict)

        descriptor = deepcopy(corpus["base_descriptor"])
        for pointer in case["remove_fields"]:
            _remove_json_pointer(descriptor, pointer)
        descriptor.update(deepcopy(case["overrides"]))
        try:
            EvaluationMethodPlugin.model_validate(descriptor)
            accepted = True
        except ValidationError:
            accepted = False
        assert accepted is case["expected_valid"], case["id"]
