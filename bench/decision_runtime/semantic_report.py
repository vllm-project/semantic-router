"""Decision-workload receipts: complete workflows, decisions, and HTTP calls."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cases import MODELS
from .report import percentile
from .semantic_cases import WorkloadCase
from .semantic_transport import HttpSample

SCHEMA_VERSION = "decision-semantic-workload-v2"


@dataclass(frozen=True)
class WorkflowSample:
    arm: str
    phase: str
    round: int
    sequence: int
    case_id: str
    concurrency: int
    decisions: int
    http_calls: int
    started_ns: int
    ended_ns: int
    error_codes: tuple[str, ...]

    @property
    def success(self) -> bool:
        return not self.error_codes

    @property
    def latency_ms(self) -> float:
        return (self.ended_ns - self.started_ns) / 1_000_000

    def public_record(self, origin_ns: int) -> dict[str, object]:
        return {
            "arm": self.arm,
            "phase": self.phase,
            "round": self.round,
            "sequence": self.sequence,
            "case_id": self.case_id,
            "concurrency": self.concurrency,
            "decisions": self.decisions,
            "http_calls": self.http_calls,
            "started_offset_ms": (self.started_ns - origin_ns) / 1_000_000,
            "completed_offset_ms": (self.ended_ns - origin_ns) / 1_000_000,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error_codes": list(self.error_codes),
        }


def group_workflows(
    arm: str,
    phase: str,
    round_number: int,
    cases: list[WorkloadCase],
    samples: list[HttpSample],
    concurrency: int,
) -> list[WorkflowSample]:
    by_sequence: dict[int, list[HttpSample]] = {}
    for sample in samples:
        by_sequence.setdefault(sample.sequence, []).append(sample)
    workflows = []
    for sequence, case in enumerate(cases):
        calls = by_sequence.get(sequence, [])
        expected = case.state_count if arm == "old" else 1
        if len(calls) != expected:
            raise ValueError("HTTP call count does not match the logical workflow")
        errors = tuple(
            sample.error_code for sample in calls if sample.error_code is not None
        )
        workflows.append(
            WorkflowSample(
                arm=arm,
                phase=phase,
                round=round_number,
                sequence=sequence,
                case_id=case.id,
                concurrency=concurrency,
                decisions=case.decisions,
                http_calls=expected,
                started_ns=min(sample.started_ns for sample in calls),
                ended_ns=max(sample.ended_ns for sample in calls),
                error_codes=errors,
            )
        )
    return workflows


def _ranks(values: list[float]) -> dict[str, float | None]:
    return {
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def _phase_summary(
    workflows: list[WorkflowSample],
    samples: list[HttpSample],
    *,
    throughput: bool,
) -> dict[str, Any]:
    successful = [workflow for workflow in workflows if workflow.success]
    errors = Counter(
        sample.error_code for sample in samples if sample.error_code is not None
    )
    result: dict[str, Any] = {
        "attempted_workflows": len(workflows),
        "successful_workflows": len(successful),
        "failed_workflows": len(workflows) - len(successful),
        "attempted_decisions": sum(workflow.decisions for workflow in workflows),
        # A multi-state workflow counts only when every state has a conforming
        # answer. A failed batch and a partially failed fan-out are equivalent.
        "successful_decisions": sum(workflow.decisions for workflow in successful),
        "http_attempts": len(samples),
        "http_successes": sum(sample.success for sample in samples),
        "errors": dict(sorted(errors.items())),
        "workflow_latency": _ranks([item.latency_ms for item in successful]),
        "http_latency": _ranks(
            [sample.latency_ms for sample in samples if sample.success]
        ),
    }
    if throughput:
        windows = []
        for round_number in sorted({workflow.round for workflow in workflows}):
            wave = [item for item in workflows if item.round == round_number]
            seconds = (
                max(item.ended_ns for item in wave)
                - min(item.started_ns for item in wave)
            ) / 1_000_000_000
            windows.append(
                {
                    "round": round_number,
                    "window_seconds": seconds,
                    "attempted_workflows": len(wave),
                    "successful_workflows": sum(item.success for item in wave),
                    "attempted_decisions": sum(item.decisions for item in wave),
                    "successful_decisions": sum(
                        item.decisions for item in wave if item.success
                    ),
                }
            )
        total = sum(window["window_seconds"] for window in windows)
        result.update(
            {
                "rounds": windows,
                "total_window_seconds": total,
                "successful_decisions_per_second": (
                    result["successful_decisions"] / total if total > 0 else None
                ),
                "attempted_decisions_per_second": (
                    result["attempted_decisions"] / total if total > 0 else None
                ),
            }
        )
    return result


def summarize_shape(
    workflows: list[WorkflowSample],
    samples: list[HttpSample],
    old: dict[str, str | int],
    new: dict[str, str | int],
    *,
    state_count: int,
    same_wire_bytes: bool,
) -> dict[str, Any]:
    arms = {}
    for arm in ("old", "new"):
        arms[arm] = {}
        for phase in ("warmup", "latency", "throughput"):
            arms[arm][phase] = _phase_summary(
                [item for item in workflows if item.arm == arm and item.phase == phase],
                [item for item in samples if item.arm == arm and item.phase == phase],
                throughput=phase == "throughput",
            )

    reasons = [
        f"different_{key}"
        for key in ("model_revision", "hardware", "network_scope")
        if old[key] != new[key]
    ]
    for arm in ("old", "new"):
        for phase in ("warmup", "latency", "throughput"):
            if arms[arm][phase]["failed_workflows"]:
                reasons.append(f"{arm}_{phase}_errors")
        for phase in ("latency", "throughput"):
            if not arms[arm][phase]["successful_workflows"]:
                reasons.append(f"{arm}_no_{phase}_samples")
    eligible = not reasons
    old_p50 = arms["old"]["latency"]["workflow_latency"]["p50_ms"]
    new_p50 = arms["new"]["latency"]["workflow_latency"]["p50_ms"]
    old_dps = arms["old"]["throughput"]["successful_decisions_per_second"]
    new_dps = arms["new"]["throughput"]["successful_decisions_per_second"]
    return {
        "arms": arms,
        "comparison": {
            "type": (
                "identical_single_request_bytes"
                if same_wire_bytes
                else (
                    "single_fanout_vs_batch_protocol_workflow"
                    if state_count > 1
                    else "single_request_model_id_adapter_workflow"
                )
            ),
            "wire_bytes_identical": same_wire_bytes,
            "eligible": eligible,
            "reasons": reasons,
            "old_over_new_p50_workflow_latency": (
                old_p50 / new_p50 if eligible and new_p50 else None
            ),
            "new_over_old_successful_decisions_per_second": (
                new_dps / old_dps if eligible and old_dps else None
            ),
        },
    }


def build_semantic_matrix(receipt_paths: list[Path]) -> dict[str, Any]:
    if len(receipt_paths) != len(MODELS):
        raise ValueError("semantic matrix requires exactly six per-model receipts")
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in receipt_paths]
    indexed = {}
    for receipt in receipts:
        if receipt.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported semantic receipt schema")
        if receipt.get("status") != "measured" or not receipt.get("shapes"):
            raise ValueError("semantic matrix requires measured receipts")
        if receipt.get("audit", {}).get("status") not in {"passed", "failed"}:
            raise ValueError("semantic matrix requires completed parity audits")
        model = receipt.get("model")
        if model in indexed:
            raise ValueError("duplicate model receipt")
        indexed[model] = receipt
    if set(indexed) != set(MODELS):
        raise ValueError("semantic matrix requires all six Decision models")
    for key in ("settings", "harness_sha256", "case_ids"):
        values = {json.dumps(receipt[key], sort_keys=True) for receipt in receipts}
        if len(values) != 1:
            raise ValueError(f"semantic matrix has mismatched {key}")
    return {
        "schema_version": SCHEMA_VERSION,
        "settings": receipts[0]["settings"],
        "harness_sha256": receipts[0]["harness_sha256"],
        "models": [
            {
                "model": model,
                "old_provenance": indexed[model]["old"],
                "new_provenance": indexed[model]["new"],
                "cohort_sha256": indexed[model]["cohort_sha256"],
                "audit": indexed[model]["audit"],
                "shapes": indexed[model]["shapes"],
            }
            for model in MODELS
        ],
    }
