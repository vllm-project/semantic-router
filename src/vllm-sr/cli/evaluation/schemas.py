"""Canonical generated JSON Schemas for cross-language contract checks."""

from __future__ import annotations

from typing import Any

from cli.evaluation.catalog import EvaluationCatalog
from cli.evaluation.contracts import RunManifest
from cli.evaluation.reporting import EvaluationReport

CONTRACT_MODELS = {
    "catalog": EvaluationCatalog,
    "manifest": RunManifest,
    "report": EvaluationReport,
}


def contract_schemas() -> dict[str, dict[str, Any]]:
    return {name: model.model_json_schema() for name, model in CONTRACT_MODELS.items()}
