"""A release cannot pass with missing, stale, or altered paired measurements."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decision_perf_release_gate as gate

SOURCE = "a" * 40
OLD_CORE = "b" * 64
MODEL_REVISION = "c" * 40
NEW_IMAGE = "sha256:" + "d" * 64
OLD_IMAGE = "sha256:" + "e" * 64
HARNESS = "f" * 64


def _save(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm(q: int, s: int, seconds: float, concurrency: int) -> dict:
    count = gate.MIN_WORKFLOWS_PER_ROUND
    attempted = count * gate.ROUNDS
    latency = seconds * 1000 / math.ceil(count / concurrency)
    throughput = {
        "attempted_workflows": attempted,
        "successful_workflows": attempted,
        "failed_workflows": 0,
        "attempted_decisions": attempted * q * s,
        "successful_decisions": attempted * q * s,
        "successful_decisions_per_second": attempted * q * s / (seconds * gate.ROUNDS),
        "workflow_latency": dict.fromkeys(("p50_ms", "p95_ms", "p99_ms"), latency),
        "rounds": [
            {
                "round": round_number,
                "window_seconds": seconds,
                "attempted_workflows": count,
                "successful_workflows": count,
                "attempted_decisions": count * q * s,
                "successful_decisions": count * q * s,
            }
            for round_number in range(gate.ROUNDS)
        ],
    }
    return {
        "throughput": throughput,
        "latency": {
            "workflow_latency": dict.fromkeys(("p50_ms", "p95_ms", "p99_ms"), 10.0)
        },
    }


def _compact_arm(raw: dict) -> dict:
    measured = raw["throughput"]
    return {
        "throughput_decisions_per_second": measured["successful_decisions_per_second"],
        "throughput_window_workflow_ms": measured["workflow_latency"],
        "low_load_probe_workflow_ms": raw["latency"]["workflow_latency"],
        **{
            name: measured[name]
            for name in (
                "attempted_workflows",
                "successful_workflows",
                "failed_workflows",
                "attempted_decisions",
                "successful_decisions",
            )
        },
        "round_windows_seconds": [row["window_seconds"] for row in measured["rounds"]],
    }


def _fixture(root: Path, *, slowdown: str | None = None) -> tuple[Path, dict]:
    report = {
        "schema_version": gate.SCHEMA,
        "scope": "synthetic same-revision Decision HTTP performance, not task-quality evaluation",
        "source_sha": SOURCE,
        "new_runtime_image_source_sha": SOURCE,
        "source_image_match": True,
        "harness_sha256": HARNESS,
        "environment": {
            "hardware": "one isolated AMD Instinct MI300X GPU per paired comparison",
            "network": "both arms loopback HTTP on same validation host",
            "physical_batch_size": gate.PHYSICAL_BATCH,
            "throughput_rounds_per_cell": gate.ROUNDS,
            "concurrencies": list(gate.CONCURRENCIES),
            "workload_shapes": [{"questions": q, "states": s} for q, s in gate.SHAPES],
        },
        "models": [],
    }
    for model_id in sorted(gate.MODEL_IDS):
        model = {
            "model_id": model_id,
            "same_old_new_revision": MODEL_REVISION,
            "same_old_new_artifact_content_id": "1" * 64,
            "artifact_metadata_sha256": "2" * 64,
            "artifact_manifest_sha256": "3" * 64,
            "old_core_source_sha256": OLD_CORE,
            "old_arm_overlay": "none",
            "old_image_id": OLD_IMAGE,
            "new_image_id": NEW_IMAGE,
            "shapes": [],
        }
        slug = model_id.rsplit("/", 1)[-1].lower()
        for q, s in gate.SHAPES:
            directory = root / "raw" / f"{slug}-q{q}s{s}"
            workflows: list[dict] = []
            cells = []
            raw_cells = []
            offset_ms = 0.0
            for concurrency in gate.CONCURRENCIES:
                new_seconds = 0.625
                if model_id == sorted(gate.MODEL_IDS)[0]:
                    if slowdown == "no_gain":
                        new_seconds = 1.0
                    elif slowdown == "regression" and (q, s, concurrency) == (
                        32,
                        1,
                        8,
                    ):
                        new_seconds = 1.4
                old = _arm(q, s, 1.0, concurrency)
                new = _arm(q, s, new_seconds, concurrency)
                for round_number in range(gate.ROUNDS):
                    for arm in (
                        ("old", "new") if round_number % 2 == 0 else ("new", "old")
                    ):
                        seconds = 1.0 if arm == "old" else new_seconds
                        groups = math.ceil(gate.MIN_WORKFLOWS_PER_ROUND / concurrency)
                        duration_ms = seconds * 1000 / groups
                        for sequence in range(gate.MIN_WORKFLOWS_PER_ROUND):
                            start = offset_ms + (sequence // concurrency) * duration_ms
                            end = start + duration_ms
                            workflows.append(
                                {
                                    "arm": arm,
                                    "phase": "throughput",
                                    "round": round_number,
                                    "sequence": sequence,
                                    "concurrency": concurrency,
                                    "decisions": q * s,
                                    "http_calls": s if arm == "old" else 1,
                                    "started_offset_ms": start,
                                    "completed_offset_ms": end,
                                    "latency_ms": duration_ms,
                                    "success": True,
                                    "error_codes": [],
                                }
                            )
                        offset_ms += seconds * 1000
                summary = {
                    "comparison": {
                        "eligible": True,
                        "reasons": [],
                        "type": (
                            "single_fanout_vs_batch_protocol_workflow"
                            if s > 1
                            else "single_request_model_id_adapter_workflow"
                        ),
                        "new_over_old_successful_decisions_per_second": 1 / new_seconds,
                        "old_over_new_p50_workflow_latency": 1.0,
                    },
                    "arms": {"old": old, "new": new},
                    "telemetry": {
                        "arms": {
                            "new": {
                                "status": "complete",
                                "rounds": gate.ROUNDS,
                                "counter_deltas": {
                                    "physical_batches": 100,
                                    "physical_batch_rows": 400,
                                },
                                "observed_rows_per_physical_batch": 4.0,
                            }
                        }
                    },
                }
                raw_cells.append(
                    {
                        "question_count": q,
                        "state_count": s,
                        "concurrency": concurrency,
                        "summary": summary,
                    }
                )
                cells.append(
                    {
                        "concurrency": concurrency,
                        "old": _compact_arm(old),
                        "new": _compact_arm(new),
                        "new_over_old_decisions_per_second": 1 / new_seconds,
                        "old_over_new_low_load_p50_ms": 1.0,
                        "new_observed_rows_per_physical_batch": 4.0,
                        "new_physical_batches": 100,
                        "new_physical_batch_rows": 400,
                    }
                )
            raw = {
                "schema_version": gate.RAW_SCHEMA,
                "model": model_id,
                "status": "measured",
                "source_commit": SOURCE,
                "harness_sha256": HARNESS,
                "failed_workflows": 0,
                "failed_metrics_shapes": 0,
                "audit": {
                    "status": "passed",
                    "comparison_eligible": True,
                    "mismatch_counts": {},
                },
                "settings": {
                    "question_counts": [q],
                    "state_counts": [s],
                    "concurrencies": list(gate.CONCURRENCIES),
                    "throughput_rounds": gate.ROUNDS,
                    "throughput_workflows_per_round_per_arm": gate.MIN_WORKFLOWS_PER_ROUND,
                    "parity_policy": "require",
                },
                "old": {
                    "model_revision": MODEL_REVISION,
                    "source_ref": "sha256:" + OLD_CORE,
                    "declared_physical_batch_size": gate.PHYSICAL_BATCH,
                    "hardware": "MI300X GPU0",
                    "network_scope": "loopback",
                },
                "new": {
                    "model_revision": MODEL_REVISION,
                    "source_ref": SOURCE,
                    "declared_physical_batch_size": gate.PHYSICAL_BATCH,
                    "hardware": "MI300X GPU0",
                    "network_scope": "loopback",
                },
                "shapes": raw_cells,
            }
            preflight = {
                "status": "passed",
                "mismatch_counts": {},
            }
            raw_receipt = directory / "receipt.json"
            raw_preflight = directory / "preflight-summary.json"
            raw_workflows = directory / "workflows.jsonl"
            receipt_sha = _save(raw_receipt, raw)
            preflight_sha = _save(raw_preflight, preflight)
            raw_workflows.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in workflows) + "\n",
                encoding="utf-8",
            )
            workflow_sha = hashlib.sha256(raw_workflows.read_bytes()).hexdigest()
            model["shapes"].append(
                {
                    "question_count": q,
                    "state_count": s,
                    "preflight_status": "passed",
                    "preflight_mismatch_counts": {},
                    "preflight_max_absolute_probability_delta": 0.001,
                    "formal_audit_status": "passed",
                    "formal_audit_mismatch_counts": {},
                    "raw_receipt_path": str(raw_receipt.relative_to(root)),
                    "raw_receipt_sha256": receipt_sha,
                    "raw_preflight_path": str(raw_preflight.relative_to(root)),
                    "raw_preflight_sha256": preflight_sha,
                    "raw_workflows_path": str(raw_workflows.relative_to(root)),
                    "raw_workflows_sha256": workflow_sha,
                    "harness_sha256": HARNESS,
                    "benchmark_source_sha": SOURCE,
                    "new_runtime_source_sha": SOURCE,
                    "cells": cells,
                }
            )
        report["models"].append(model)
    path = root / "report.json"
    _save(path, report)
    return path, report


class DecisionPerformanceGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path, self.report = _fixture(self.root)

    def write_report(self) -> None:
        _save(self.path, self.report)

    def test_complete_six_model_paired_measurement_passes(self) -> None:
        result = gate.validate_report(self.path, source_sha=SOURCE)
        self.assertEqual(set(result["models"]), gate.MODEL_IDS)
        self.assertEqual(result["source_sha"], SOURCE)

    def test_requires_same_source_and_candidate_image(self) -> None:
        self.report["new_runtime_image_source_sha"] = "9" * 40
        self.write_report()
        with self.assertRaisesRegex(ValueError, "candidate image source"):
            gate.validate_report(self.path, source_sha=SOURCE)

    def test_requires_all_six_models_three_shapes_and_three_cells(self) -> None:
        self.report["models"].pop()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "exactly six"):
            gate.validate_report(self.path, source_sha=SOURCE)
        _, self.report = _fixture(self.root)
        self.report["models"][0]["shapes"][0]["cells"].pop()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "three concurrency cells"):
            gate.validate_report(self.path, source_sha=SOURCE)

    def test_raw_files_are_hashed_and_cannot_escape_bundle(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        raw = self.root / shape["raw_receipt_path"]
        raw.write_text("{}")
        with self.assertRaisesRegex(ValueError, "content changed"):
            gate.validate_report(self.path, source_sha=SOURCE)
        _, self.report = _fixture(self.root)
        self.report["models"][0]["shapes"][0]["raw_receipt_path"] = "../elsewhere.json"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "under raw"):
            gate.validate_report(self.path, source_sha=SOURCE)

    def test_semantic_mismatch_and_failures_block_release(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        shape["preflight_mismatch_counts"] = {"probability_tolerance": 1}
        self.write_report()
        with self.assertRaisesRegex(ValueError, "semantic mismatch"):
            gate.validate_report(self.path, source_sha=SOURCE)

    def test_recomputed_ratio_rejects_fabricated_gain(self) -> None:
        cell = self.report["models"][0]["shapes"][0]["cells"][0]
        cell["new_over_old_decisions_per_second"] = 100.0
        self.write_report()
        with self.assertRaisesRegex(ValueError, "throughput ratio"):
            gate.validate_report(self.path, source_sha=SOURCE)

    def test_multi_state_protocol_must_be_labeled_as_batch_comparison(self) -> None:
        shape = self.report["models"][0]["shapes"][1]
        raw_path = self.root / shape["raw_receipt_path"]
        raw = json.loads(raw_path.read_text())
        raw["shapes"][0]["summary"]["comparison"][
            "type"
        ] = "identical_single_request_bytes"
        shape["raw_receipt_sha256"] = _save(raw_path, raw)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "protocol is mislabeled"):
            gate.validate_report(self.path, source_sha=SOURCE)

    def test_physical_batch_telemetry_is_consistent(self) -> None:
        cell = self.report["models"][0]["shapes"][0]["cells"][0]
        cell["new_physical_batch_rows"] = 900
        self.write_report()
        with self.assertRaisesRegex(ValueError, "physical batch"):
            gate.validate_report(self.path, source_sha=SOURCE)

    def test_no_material_gain_blocks_release(self) -> None:
        self.path, self.report = _fixture(self.root, slowdown="no_gain")
        with self.assertRaisesRegex(
            ValueError, "lacks a material high-load performance gain"
        ):
            gate.validate_report(self.path, source_sha=SOURCE)

    def test_high_load_regression_blocks_release(self) -> None:
        self.path, self.report = _fixture(self.root, slowdown="regression")
        with self.assertRaisesRegex(
            ValueError, "material high-load throughput regression"
        ):
            gate.validate_report(self.path, source_sha=SOURCE)

    def test_interleaved_waves_must_match_raw_workflow_times(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_workflows_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["concurrency"] == 1 and row["round"] == 0 and row["arm"] == "new":
                row["started_offset_ms"] -= 1000
                row["completed_offset_ms"] -= 1000
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
        )
        shape["raw_workflows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "alternating waves"):
            gate.validate_report(self.path, source_sha=SOURCE)


if __name__ == "__main__":
    unittest.main()
