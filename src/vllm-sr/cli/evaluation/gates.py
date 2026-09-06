"""Small dispatcher for the evidence-aware G0-G9 release gate contract."""

from __future__ import annotations

from datetime import datetime

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.gate_context import GateEvidenceContext, gate_metadata
from cli.evaluation.gate_contract import (
    DEFAULT_CHANGE_PROFILE,
    ChangeProfile,
    gate_applicability,
)
from cli.evaluation.gate_foundation import (
    evaluate_g0,
    evaluate_g1,
    evaluate_g2,
    evaluate_g3,
)
from cli.evaluation.gate_methods import (
    evaluate_g4,
    evaluate_g5,
    evaluate_g6,
    evaluate_g7,
)
from cli.evaluation.gate_production import evaluate_g8, evaluate_g9
from cli.evaluation.reporting import EvaluationGate, EvaluationMetric


def compute_gates(
    metrics: list[EvaluationMetric],
    *,
    has_records: bool,
    change_profile: ChangeProfile = DEFAULT_CHANGE_PROFILE,
    evidence: GateEvidenceContext | None = None,
    records: list[ExecutionRecord] | None = None,
    evaluated_at: datetime | None = None,
) -> list[EvaluationGate]:
    """Evaluate every gate under one explicit change-profile matrix."""

    context = evidence or GateEvidenceContext()
    gates: list[EvaluationGate] = []
    for definition, disposition in gate_applicability(change_profile):
        metadata = gate_metadata(definition, change_profile, records, evaluated_at)
        if definition.id == "G0":
            gate = evaluate_g0(
                definition,
                disposition,
                metadata,
                has_records=has_records,
                context=context,
            )
        elif definition.id == "G1":
            gate = evaluate_g1(definition, disposition, metadata, context=context)
        elif definition.id == "G2":
            gate = evaluate_g2(
                definition,
                disposition,
                metadata,
                metrics=metrics,
                context=context,
            )
        elif definition.id == "G3":
            gate = evaluate_g3(definition, disposition, metadata)
        elif definition.id == "G4":
            gate = evaluate_g4(definition, disposition, metadata, context)
        elif definition.id == "G5":
            gate = evaluate_g5(definition, disposition, metadata, context)
        elif definition.id == "G6":
            gate = evaluate_g6(definition, disposition, metadata, context)
        elif definition.id == "G7":
            gate = evaluate_g7(definition, disposition, metadata, metrics)
        elif definition.id == "G8":
            gate = evaluate_g8(definition, disposition, metadata, context)
        elif definition.id == "G9":
            gate = evaluate_g9(definition, disposition, metadata, context)
        else:  # pragma: no cover - gate_applicability owns the closed set
            raise ValueError(f"unsupported gate: {definition.id}")
        gates.append(gate)
    return gates
