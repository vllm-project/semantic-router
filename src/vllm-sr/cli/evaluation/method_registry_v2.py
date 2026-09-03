"""v2 method declarations projected from the shared research inventory."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from cli.evaluation.method_contract_v2 import (
    EVALUATION_METHOD_CONTRACT_VERSION,
    AnalysisPlan,
    EvaluationMethodPlugin,
    SliceRef,
)
from cli.evaluation.research_benchmark_inventory import RESEARCH_BENCHMARKS


def _plugin(entry: Mapping[str, Any]) -> EvaluationMethodPlugin:
    return EvaluationMethodPlugin(
        schema_version=EVALUATION_METHOD_CONTRACT_VERSION,
        id=entry["method_id"],
        version=EVALUATION_METHOD_CONTRACT_VERSION,
        status=entry["status"],
        execution_owner=entry["execution_owner"],
        input_schema=entry["input_schema"],
        export_schema=entry["export_schema"],
        live_input_complete=False,
        live_grader=False,
        applicable_tracks=tuple(entry["applicable_tracks"]),
        live_tracks=(),
        produced_metric_ids=tuple(entry["produced_metric_ids"]),
        evidence_ceiling=entry["evidence_ceiling"],
        native_parity=entry["native_parity"],
        required_artifact_ids=tuple(entry["required_artifact_ids"]),
        analysis_plan=AnalysisPlan(
            schema_version=EVALUATION_METHOD_CONTRACT_VERSION,
            id=entry["analysis_plan_id"],
            analysis_unit=entry["analysis_unit"],
            cluster_unit="case",
            slices=(
                SliceRef(
                    schema_version=EVALUATION_METHOD_CONTRACT_VERSION,
                    id="all",
                ),
            ),
            curve_domain=entry["curve_domain"],
            missingness="fail_closed",
        ),
    )


METHOD_PLUGINS = tuple(_plugin(entry) for entry in RESEARCH_BENCHMARKS)

_BY_BENCHMARK = MappingProxyType(
    {
        entry["adapter_id"]: plugin
        for entry, plugin in zip(RESEARCH_BENCHMARKS, METHOD_PLUGINS, strict=True)
    }
)


def method_plugin_for_benchmark(adapter_id: str) -> EvaluationMethodPlugin:
    try:
        return _BY_BENCHMARK[adapter_id]
    except KeyError as exc:
        raise ValueError(
            f"benchmark has no v2 research method declaration: {adapter_id}"
        ) from exc
