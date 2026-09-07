"""Local-only comparison output without a Dashboard server attestation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.reporting import DecisionVerdict, EvaluationGate, EvaluationMetric


class StandaloneComparison(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    baseline_run_id: str
    candidate_run_id: str
    verdict: DecisionVerdict
    summary: str
    metrics: tuple[EvaluationMetric, ...]
    gates: tuple[EvaluationGate, ...]
    recommendations: tuple[str, ...]
    created_at: datetime | None = None
