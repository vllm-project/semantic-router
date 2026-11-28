"""Replay a promotion verdict from the selected ledger record and experiment spec."""

from __future__ import annotations

import contextlib
import math
import re

from .contract import SPLIT_NAMES, ContractError, ExperimentSpec
from .ledger import compute_record_id, validate_record
from .live import measurements_from_rows, paired_stats_from_comparison, resolve_gates
from .verdict import evaluate_verdict, evidence_quality_reasons


def _number(value, field):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ContractError(f"promotion evidence needs a finite {field}")
    return value


def verify_promotion_evidence(spec: ExperimentSpec, record: dict) -> None:
    validate_record(record)
    if record.get("record_id") != compute_record_id(record):
        raise ContractError("promotion record content differs from its record_id")
    if len(spec.workloads) > 1 and not record.get("workload"):
        raise ContractError("promotion record must name one declared workload")
    workload = spec.select_workload(record.get("workload"))
    split = record.get("split") or ("dev" if workload.legacy else None)
    if split not in SPLIT_NAMES:
        raise ContractError(f"unknown live split: {split!r}")
    if not split or workload.split_hash(split) != record.get("dataset_hash"):
        raise ContractError(
            "promotion record dataset_hash does not match its workload split"
        )

    evidence = record.get("verification", {})
    objective = spec.objective
    candidate = _number(
        record["metrics"].get(objective.metric_key), "candidate objective"
    )
    baseline = evidence.get("baseline")
    if baseline is None:
        # Both live CLI paths originally wrote the baseline in this notes field.
        match = re.search(
            r"(?:^|;\s*)baseline=([-+0-9.eE]+)(?:;|$)", record.get("notes", "")
        )
        if match:
            with contextlib.suppress(ValueError):
                baseline = float(match.group(1))
    baseline = _number(baseline, "baseline objective")

    measurements = evidence.get("guard_measurements", [])
    if not isinstance(measurements, list):
        raise ContractError("guard_measurements must be a list")
    guards, _ = resolve_gates(spec.gates, measurements_from_rows(measurements))
    if record.get("gates") != {guard.name: "pass" for guard in guards}:
        raise ContractError("promotion record must carry every declared passing guard")
    constraints = record.get("constraints") or {}
    if not isinstance(constraints, dict) or set(constraints) != {
        c.name for c in objective.constraints
    }:
        raise ContractError("recorded constraints differ from the experiment spec")
    inputs = []
    references = {}
    for declaration in objective.constraints:
        row = constraints[declaration.name]
        if not isinstance(row, dict) or row.get("status") != "pass":
            raise ContractError(
                f"constraint {declaration.name} has no passing measurement"
            )
        spec_fields = declaration.to_dict()
        for key, value in spec_fields.items():
            if row.get(key, 0 if key == "tolerance_pct" else None) != value:
                raise ContractError(
                    f"constraint {declaration.name}.{key} differs from the spec"
                )
        entry = dict(
            spec_fields, candidate=_number(row.get("candidate"), declaration.name)
        )
        if row.get("baseline") is not None:
            entry["baseline"] = _number(row["baseline"], declaration.name + " baseline")
        if declaration.metric == objective.metric_key and (
            entry["candidate"] != candidate
            or entry.get("baseline", baseline) != baseline
        ):
            raise ContractError(
                f"constraint {declaration.name} disagrees with the objective measurements"
            )
        if isinstance(declaration.bound, str) and declaration.bound.startswith("ref:"):
            role = declaration.bound.split(":", 1)[1]
            if row.get("ref_metric", objective.metric_key) != declaration.metric:
                raise ContractError(
                    f"constraint {declaration.name} needs its reference in {declaration.metric}"
                )
            ref = _number(row.get("ref_value"), declaration.name + " reference")
            metrics = references.setdefault(role, {})
            if metrics.setdefault(declaration.metric, ref) != ref:
                raise ContractError(f"conflicting reference measurements for {role}")
        inputs.append(entry)
    raw_paired = record.get("paired")
    paired = paired_stats_from_comparison({"comparisons": [raw_paired]})
    if raw_paired is not None and paired is None:
        raise ContractError("invalid paired evidence in promotion record")
    incomplete = []
    quality = evidence.get("evidence_quality") or {}
    if not isinstance(quality, dict):
        raise ContractError("evidence_quality must contain arm measurements")
    for arm, rates in quality.items():
        if not isinstance(rates, dict):
            raise ContractError(f"invalid {arm} evidence quality measurements")
        measured = {}
        for key in ("truncation_rate", "missing_result_rate", "no_answer_rate"):
            value = rates.get(key)
            if value is not None and not 0 <= _number(value, key) <= 1:
                raise ContractError(f"invalid {arm} {key}")
            measured[key] = value
        incomplete.extend(
            evidence_quality_reasons(
                arm,
                **measured,
                max_truncation_pct=spec.evidence_quality.max_truncation_pct,
                max_no_answer_pct=spec.evidence_quality.max_no_answer_pct,
            )
        )
    replayed = evaluate_verdict(
        baseline_metric=baseline,
        candidate_metric=candidate,
        direction=objective.direction,
        min_improvement_pct=objective.min_improvement_pct,
        min_improvement_pp=objective.min_improvement_pp,
        min_cases=objective.min_cases,
        stats=paired,
        guards=guards,
        constraints=inputs,
        references=references,
        incomplete_reasons=incomplete,
    )
    if replayed.verdict != "PASS":
        raise ContractError(
            "promotion measurements do not support PASS: " + "; ".join(replayed.reasons)
        )
    for row in replayed.constraints:
        if row["bound_value"] != constraints[row["name"]].get("bound_value"):
            raise ContractError(
                f"constraint {row['name']} recorded bound differs from its measured bound"
            )
