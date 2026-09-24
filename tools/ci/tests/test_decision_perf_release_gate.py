"""A release cannot pass with missing, stale, or altered paired measurements."""

from __future__ import annotations

import hashlib
import base64
import copy
import gzip
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decision_perf_release_gate as gate
import decision_timed_semantics as timed_semantics

SOURCE = "a" * 40
OLD_CORE = "b" * 64
MODEL_REVISION = "c" * 40
NEW_IMAGE = "sha256:" + "d" * 64
OLD_IMAGE = "sha256:" + "e" * 64
HARNESS = "f" * 64
CANDIDATE_REF = "ghcr.io/example/decision-runtime-rocm@sha256:" + "6" * 64


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _qualification() -> dict:
    return {
        "source_sha": SOURCE,
        "candidate_ref": CANDIDATE_REF,
        "models": [
            {
                "id": model_id,
                "revision": MODEL_REVISION,
                "artifact_content_id": "sha256:" + "1" * 64,
            }
            for model_id in sorted(gate.MODEL_IDS)
        ],
    }


def _save(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm(q: int, s: int, seconds: float, concurrency: int, arm: str) -> dict:
    count = gate.MIN_WORKFLOWS_PER_ROUND
    attempted = count * gate.ROUNDS
    if arm == "old":
        http_latency = seconds * 1000 / math.ceil(count * s / concurrency)
        latency = http_latency * math.ceil(s / concurrency)
    else:
        latency = seconds * 1000 / math.ceil(count / concurrency)
        http_latency = latency
    throughput = {
        "attempted_workflows": attempted,
        "successful_workflows": attempted,
        "failed_workflows": 0,
        "attempted_decisions": attempted * q * s,
        "successful_decisions": attempted * q * s,
        "successful_decisions_per_second": attempted * q * s / (seconds * gate.ROUNDS),
        "workflow_latency": dict.fromkeys(("p50_ms", "p95_ms", "p99_ms"), latency),
        "http_latency": dict.fromkeys(("p50_ms", "p95_ms", "p99_ms"), http_latency),
        "http_attempts": attempted * (s if arm == "old" else 1),
        "http_successes": attempted * (s if arm == "old" else 1),
        "errors": {},
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

    def low_load(count: int) -> dict:
        return {
            "attempted_workflows": count,
            "successful_workflows": count,
            "failed_workflows": 0,
            "attempted_decisions": count * q * s,
            "successful_decisions": count * q * s,
            "http_attempts": count * (s if arm == "old" else 1),
            "http_successes": count * (s if arm == "old" else 1),
            "errors": {},
            "workflow_latency": dict.fromkeys(("p50_ms", "p95_ms", "p99_ms"), 10.0),
            "http_latency": dict.fromkeys(
                ("p50_ms", "p95_ms", "p99_ms"),
                10.0 / math.ceil(s / concurrency) if arm == "old" else 10.0,
            ),
        }

    return {
        "throughput": throughput,
        "warmup": low_load(2),
        "latency": low_load(16),
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


def _request_body(
    case_id: str,
    arm: str,
    q: int,
    s: int,
    model_id: str,
    state: int | None = None,
) -> bytes:
    body = timed_semantics.canonical_request_bodies(model_id, q, s)[case_id]
    if arm == "new":
        return body
    payload = json.loads(body)
    payload["model"] = "old-decision-model"
    if s > 1:
        payload["state"] = payload.pop("states")[state]["state"]
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _request_hash(
    case_id: str,
    arm: str,
    q: int,
    s: int,
    model_id: str,
    state: int | None = None,
) -> str:
    return hashlib.sha256(
        _request_body(case_id, arm, q, s, model_id, state)
    ).hexdigest()


def _audit_record(case_id: str, q: int, s: int, model_id: str) -> dict:
    questions = json.loads(_request_body(case_id, "new", q, s, model_id))["questions"]

    def answer(question_id: str, question: dict) -> dict:
        kind = question["type"]
        if kind == "noul":
            return {
                "question_id": question_id,
                "type": kind,
                "old_probability": 0.5,
                "new_probability": 0.501,
                "absolute_probability_delta": 0.001,
            }
        keys = (
            list(question["criteria"])
            if kind == "choice"
            else [str(i) for i in range(len(question["criteria"]))]
        )
        probabilities = dict.fromkeys(keys, 1 / len(keys))
        row = {
            "question_id": question_id,
            "type": kind,
            "old_probabilities": probabilities,
            "new_probabilities": probabilities,
            "absolute_probability_deltas": dict.fromkeys(keys, 0.0),
        }
        if kind == "choice":
            row.update(old_outcome=keys[0], new_outcome=keys[0])
        else:
            row.update(old_score=1.0, new_score=1.0)
        return row

    return {
        "case_id": case_id,
        "question_count": q,
        "state_count": s,
        "request_semantics_identical_except_model_id_and_batch_envelope": True,
        "passed": True,
        "mismatches": [],
        "old_http": [
            {
                "request_sha256": _request_hash(case_id, "old", q, s, model_id, index),
                "response_sha256": "7" * 64,
                "status_code": 200,
                "error_code": None,
            }
            for index in range(s)
        ],
        "new_http": {
            "request_sha256": _request_hash(case_id, "new", q, s, model_id),
            "response_sha256": "8" * 64,
            "status_code": 200,
            "error_code": None,
        },
        "states": [
            {
                "state_id": f"s{index:04d}",
                "old_input_tokens": 10,
                "new_input_tokens": 10,
                "input_token_delta": 0,
                "new_output_tokens": 0,
                "answers": [
                    answer(question_id, question)
                    for question_id, question in questions.items()
                ],
            }
            for index in range(s)
        ],
    }


def _timed_response(model_id: str, q: int, s: int, case_id: str) -> bytes:
    questions = json.loads(_request_body(case_id, "new", q, s, model_id))["questions"]

    def answer(question: dict) -> dict:
        kind = question["type"]
        if kind == "noul":
            return {"type": kind, "noul": 0.501}
        keys = (
            list(question["criteria"])
            if kind == "choice"
            else [str(index) for index in range(len(question["criteria"]))]
        )
        probabilities = dict.fromkeys(keys, 1 / len(keys))
        if kind == "choice":
            return {
                "type": kind,
                "choice": keys[0],
                "confidence": 0.0,
                "probabilities": probabilities,
            }
        return {
            "type": kind,
            "score": 1.0,
            "confidence": 0.0,
            "legend": {
                str(index): criterion
                for index, criterion in enumerate(question["criteria"])
            },
            "probabilities": probabilities,
        }

    answers = {
        question_id: answer(question) for question_id, question in questions.items()
    }
    if s == 1:
        body = {
            "model": model_id,
            "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }
    else:
        body = {
            "model": model_id,
            "results": [
                {
                    "id": f"s{index:04d}",
                    "answers": answers,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                }
                for index in range(s)
            ],
            "usage": {"input_tokens": 10 * s, "output_tokens": 0},
        }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def _probability_count(model_id: str, q: int, s: int) -> int:
    count = 0
    for body in timed_semantics.canonical_request_bodies(model_id, q, s).values():
        questions = json.loads(body)["questions"]
        count += (
            sum(
                1 if question["type"] == "noul" else len(question["criteria"])
                for question in questions.values()
            )
            * s
        )
    return count


def _sample_records(
    workflow: dict,
    q: int,
    s: int,
    model_id: str,
    case_ids: list[str],
    intervals: list[tuple[float, float]],
) -> list[dict]:
    arm = workflow["arm"]
    selected = (
        [workflow["case_id"]]
        if workflow["phase"] != "throughput"
        else [
            case_ids[index % len(case_ids)]
            for index in range(gate.MIN_WORKFLOWS_PER_ROUND)
        ]
    )
    schedule = {
        "question_count": q,
        "state_count": s,
        "concurrency": workflow["concurrency"],
        "phase": workflow["phase"],
        "round": workflow["round"],
        "case_ids": selected,
    }
    schedule_hash = hashlib.sha256(
        json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return [
        {
            **{
                field: workflow[field]
                for field in (
                    "arm",
                    "phase",
                    "round",
                    "sequence",
                    "case_id",
                    "concurrency",
                )
            },
            "state_id": f"s{index:04d}" if arm == "old" else None,
            "request_kind": "single" if arm == "old" or s == 1 else "batch",
            "request_sha256": _request_hash(
                workflow["case_id"],
                arm,
                q,
                s,
                model_id,
                index if arm == "old" else None,
            ),
            "response_sha256": "9" * 64,
            "request_bytes": 100,
            "logical_schedule_sha256": schedule_hash,
            "decisions": q if arm == "old" else q * s,
            "started_offset_ms": intervals[index][0],
            "completed_offset_ms": intervals[index][1],
            "latency_ms": intervals[index][1] - intervals[index][0],
            "status_code": 200,
            "success": True,
            "error_code": None,
        }
        for index in range(s if arm == "old" else 1)
    ]


def _throughput_intervals(
    *,
    arm: str,
    sequence: int,
    s: int,
    concurrency: int,
    seconds: float,
    offset_ms: float,
    latency_only: bool,
) -> list[tuple[float, float]]:
    if latency_only:
        start = offset_ms + sequence * 500 / (gate.MIN_WORKFLOWS_PER_ROUND - 1)
        return [(start, start + 500)]
    calls_per_workflow = s if arm == "old" else 1
    total_calls = gate.MIN_WORKFLOWS_PER_ROUND * calls_per_workflow
    call_duration = seconds * 1000 / math.ceil(total_calls / concurrency)
    return [
        (
            offset_ms
            + ((sequence * calls_per_workflow + index) // concurrency) * call_duration,
            offset_ms
            + (((sequence * calls_per_workflow + index) // concurrency) + 1)
            * call_duration,
        )
        for index in range(calls_per_workflow)
    ]


def _refresh_metric_hash(snapshot: dict) -> None:
    content = {
        "counters": snapshot["counters"],
        "physical_batch_size_buckets": snapshot["physical_batch_size_buckets"],
    }
    snapshot["response_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _metric_record(
    q: int,
    s: int,
    concurrency: int,
    round_number: int,
    cumulative: dict[str, float],
) -> dict:
    rows_per_round = gate.MIN_WORKFLOWS_PER_ROUND * q * s
    batches = rows_per_round // 4
    previous_batches = int(cumulative["physical_batches"])
    values = {
        "row_preparation_seconds": 0.1,
        "row_preparations": gate.MIN_WORKFLOWS_PER_ROUND,
        "physical_batches": batches,
        "physical_batch_rows": rows_per_round,
    }
    before = dict(cumulative)
    after = {name: before[name] + value for name, value in values.items()}
    cumulative.update(after)

    def buckets(count: int) -> dict:
        return {
            **{
                str(bound): count if bound >= 4 else 0
                for bound in gate.PHYSICAL_BATCH_BUCKETS
            },
            "+Inf": count,
        }

    before_buckets = buckets(previous_batches)
    after_buckets = buckets(previous_batches + batches)
    delta_buckets = buckets(batches)

    def snapshot(counters: dict, histogram: dict) -> dict:
        # The fixture models one immutable /metrics response per hash. Reusing
        # one hash for changing cumulative counters would be impossible on wire.
        content = {
            "counters": counters,
            "physical_batch_size_buckets": histogram,
        }
        _refresh_metric_hash(content)
        return content

    return {
        "arm": "new",
        "question_count": q,
        "state_count": s,
        "concurrency": concurrency,
        "round": round_number,
        "before": snapshot(before, before_buckets),
        "after": snapshot(after, after_buckets),
        "delta": {
            "counters": values,
            "physical_batch_size_buckets": delta_buckets,
        },
        "error_code": None,
    }


def _fixture(
    root: Path,
    *,
    slowdown: str | None = None,
    throughput_ratios: dict[tuple[str, int, int, int], float] | None = None,
) -> tuple[Path, dict]:
    report = {
        "schema_version": gate.SCHEMA,
        "scope": "synthetic same-revision Decision HTTP performance, not task-quality evaluation",
        "source_sha": SOURCE,
        "candidate_ref": CANDIDATE_REF,
        "run_id": "42",
        "run_attempt": "1",
        "new_runtime_image_source_sha": SOURCE,
        "source_image_match": True,
        "harness_sha256": HARNESS,
        "environment": {
            "hardware": "one isolated AMD Instinct MI300X GPU per paired comparison",
            "network": "both arms loopback HTTP on same validation host",
            "physical_batch_policy": "per-model launch size; occupancy from per-round metrics",
            "throughput_rounds_per_cell": gate.ROUNDS,
            "concurrencies": list(gate.CONCURRENCIES),
            "workload_shapes": [{"questions": q, "states": s} for q, s in gate.SHAPES],
            "arrival_policy": gate.ARRIVAL_POLICY,
            "gpu_exclusivity": "dedicated_gpu_no_unrelated_compute; externally attested by protected runner",
            "gpu_clock_policy": "unobserved",
            "service_residency": "both service containers running during alternating waves; HBM residency unmeasured",
            "new_max_concurrency": 4,
            "new_max_queue": 32,
        },
        "models": [],
    }
    for model_id in sorted(gate.MODEL_IDS):
        metric_snapshot = {
            "row_preparation_seconds": 0.0,
            "row_preparations": 0,
            "physical_batches": 0,
            "physical_batch_rows": 0,
        }
        model = {
            "model_id": model_id,
            "old_core_source_kind": "mounted_adapter",
            "old_artifact_layout": "full_snapshot_selected_data_v1",
            "old_full_snapshot_sha256": "4" * 64,
            "old_snapshot_verifications": [
                {
                    "full_snapshot_sha256": "4" * 64,
                    "selected_data_content_id": "1" * 64,
                },
                {
                    "full_snapshot_sha256": "4" * 64,
                    "selected_data_content_id": "1" * 64,
                },
            ],
            "same_old_new_revision": MODEL_REVISION,
            "same_old_new_artifact_content_id": "1" * 64,
            "artifact_metadata_sha256": "2" * 64,
            "artifact_manifest_sha256": "3" * 64,
            "old_core_source_sha256": OLD_CORE,
            "old_adapter_source_sha256": "7" * 64,
            "old_source_declaration": {
                "adapter_path_sha256": _text_sha("/adapter.py"),
                "imported_core_path_sha256": _text_sha("/core.py"),
                "imported_module_sha256": _text_sha("old_core"),
                "artifact_mount_path_sha256": _text_sha("/artifact"),
            },
            "old_arm_overlay": "none",
            "old_image_id": OLD_IMAGE,
            "new_image_id": NEW_IMAGE,
            "old_physical_batch_size": gate.PHYSICAL_BATCH,
            "new_physical_batch_size": gate.PHYSICAL_BATCH,
            "new_scheduler": {"max_concurrency": 4, "max_queue": 32},
            "shapes": [],
        }
        slug = model_id.rsplit("/", 1)[-1].lower()
        old_observation = {
            "schema_version": "decision-old-baseline-attestation-v1",
            "challenge": "8" * 64,
            "pid": 1,
            "process_start_ticks": 12345,
            "adapter_path_sha256": _text_sha("/adapter.py"),
            "adapter_sha256": model["old_adapter_source_sha256"],
            "imported_module_sha256": _text_sha("old_core"),
            "imported_core_path_sha256": _text_sha("/core.py"),
            "core_mount_sha256": OLD_CORE,
            "loaded_artifact_root_sha256": _text_sha("/artifact"),
            "loaded_artifact_content_id": model["same_old_new_artifact_content_id"],
            "model_id": model_id,
            "revision": MODEL_REVISION,
        }
        old_attestation = root / "raw" / f"{slug}-old-attestation.json"
        model["old_attestation_path"] = str(old_attestation.relative_to(root))
        model["old_attestation_sha256"] = _save(
            old_attestation,
            {
                "schema_version": "decision-old-baseline-attestation-v1",
                "observations": [
                    old_observation,
                    {**old_observation, "challenge": "9" * 64},
                ],
            },
        )
        for q, s in gate.SHAPES:
            directory = root / "raw" / f"{slug}-q{q}s{s}"
            case_ids = list(timed_semantics.canonical_request_bodies(model_id, q, s))
            probability_count = _probability_count(model_id, q, s)
            workflows: list[dict] = []
            samples: list[dict] = []
            metrics: list[dict] = []
            cells = []
            raw_cells = []
            offset_ms = 0.0
            for concurrency in gate.CONCURRENCIES:
                new_seconds = 0.625
                if model_id == sorted(gate.MODEL_IDS)[0]:
                    if slowdown in ("no_gain", "latency_only"):
                        new_seconds = 1.0
                    elif slowdown == "regression" and (q, s, concurrency) == (
                        32,
                        1,
                        8,
                    ):
                        new_seconds = 1 / 0.94
                if throughput_ratios is not None:
                    ratio = throughput_ratios.get((model_id, q, s, concurrency))
                    if ratio is not None:
                        new_seconds = 1 / ratio
                old = _arm(q, s, 1.0, concurrency, "old")
                new = _arm(q, s, new_seconds, concurrency, "new")
                total_rows = gate.MIN_WORKFLOWS_PER_ROUND * gate.ROUNDS * q * s
                total_batches = total_rows // 4
                if slowdown == "latency_only" and (q, s, concurrency) == (32, 1, 32):
                    new["throughput"]["workflow_latency"] = dict.fromkeys(
                        ("p50_ms", "p95_ms", "p99_ms"), 500.0
                    )
                    new["throughput"]["http_latency"] = dict.fromkeys(
                        ("p50_ms", "p95_ms", "p99_ms"), 500.0
                    )
                for phase, count in (("warmup", 2), ("latency", 16)):
                    for sequence in range(count):
                        for arm in (
                            ("old", "new") if sequence % 2 == 0 else ("new", "old")
                        ):
                            call_duration = (
                                10 / math.ceil(s / concurrency)
                                if arm == "old"
                                else 10.0
                            )
                            intervals = [
                                (
                                    offset_ms + (index // concurrency) * call_duration,
                                    offset_ms
                                    + ((index // concurrency) + 1) * call_duration,
                                )
                                for index in range(s if arm == "old" else 1)
                            ]
                            workflow = {
                                "arm": arm,
                                "phase": phase,
                                "round": sequence,
                                "sequence": 0,
                                "case_id": case_ids[sequence % len(case_ids)],
                                "concurrency": concurrency,
                                "decisions": q * s,
                                "http_calls": s if arm == "old" else 1,
                                "started_offset_ms": min(
                                    start for start, _ in intervals
                                ),
                                "completed_offset_ms": max(end for _, end in intervals),
                                "latency_ms": max(end for _, end in intervals)
                                - min(start for start, _ in intervals),
                                "success": True,
                                "error_codes": [],
                            }
                            workflows.append(workflow)
                            samples.extend(
                                _sample_records(
                                    workflow, q, s, model_id, case_ids, intervals
                                )
                            )
                            offset_ms += 10
                for round_number in range(gate.ROUNDS):
                    for arm in (
                        ("old", "new") if round_number % 2 == 0 else ("new", "old")
                    ):
                        seconds = 1.0 if arm == "old" else new_seconds
                        latency_only = slowdown == "latency_only" and (
                            q,
                            s,
                            concurrency,
                            arm,
                        ) == (32, 1, 32, "new")
                        for sequence in range(gate.MIN_WORKFLOWS_PER_ROUND):
                            intervals = _throughput_intervals(
                                arm=arm,
                                sequence=sequence,
                                s=s,
                                concurrency=concurrency,
                                seconds=seconds,
                                offset_ms=offset_ms,
                                latency_only=latency_only,
                            )
                            start = min(begin for begin, _ in intervals)
                            end = max(finish for _, finish in intervals)
                            workflow = {
                                "arm": arm,
                                "phase": "throughput",
                                "round": round_number,
                                "sequence": sequence,
                                "case_id": case_ids[sequence % len(case_ids)],
                                "concurrency": concurrency,
                                "decisions": q * s,
                                "http_calls": s if arm == "old" else 1,
                                "started_offset_ms": start,
                                "completed_offset_ms": end,
                                "latency_ms": end - start,
                                "success": True,
                                "error_codes": [],
                            }
                            workflows.append(workflow)
                            samples.extend(
                                _sample_records(
                                    workflow, q, s, model_id, case_ids, intervals
                                )
                            )
                        offset_ms += seconds * 1000
                    metrics.append(
                        _metric_record(q, s, concurrency, round_number, metric_snapshot)
                    )
                summary = {
                    "comparison": {
                        "eligible": True,
                        "reasons": [],
                        "type": (
                            "single_fanout_vs_batch_protocol_workflow"
                            if s > 1
                            else "single_request_model_id_adapter_workflow"
                        ),
                        "wire_bytes_identical": False,
                        "new_over_old_successful_decisions_per_second": 1 / new_seconds,
                        "old_over_new_p50_workflow_latency": 1.0,
                    },
                    "arms": {"old": old, "new": new},
                    "telemetry": {
                        "arms": {
                            "old": {"status": "not_requested"},
                            "new": {
                                "status": "complete",
                                "rounds": gate.ROUNDS,
                                "counter_deltas": {
                                    "row_preparation_seconds": 0.30000000000000004,
                                    "row_preparations": 96,
                                    "physical_batches": total_batches,
                                    "physical_batch_rows": total_rows,
                                },
                                "physical_batch_size_bucket_deltas": {
                                    **{
                                        str(bound): total_batches if bound >= 4 else 0
                                        for bound in gate.PHYSICAL_BATCH_BUCKETS
                                    },
                                    "+Inf": total_batches,
                                },
                                "normalization_decisions": 96 * q * s,
                                "row_preparation_seconds_per_decision": 0.30000000000000004
                                / (96 * q * s),
                                "row_preparations_per_decision": 1 / (q * s),
                                "physical_batch_rows_per_decision": 1.0,
                                "observed_rows_per_physical_batch": 4.0,
                            },
                        }
                    },
                }
                raw_cells.append(
                    {
                        "question_count": q,
                        "state_count": s,
                        "concurrency": concurrency,
                        "case_ids": case_ids,
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
                        "new_physical_batches": total_batches,
                        "new_physical_batch_rows": total_rows,
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
                "case_ids": case_ids,
                "audit": {
                    "status": "passed",
                    "comparison_eligible": True,
                    "mismatch_counts": {},
                    "cases": len(case_ids),
                    "passed_cases": len(case_ids),
                    "failed_cases": 0,
                    "absolute_probability_delta": {
                        "count": probability_count,
                        "max": 0.001,
                    },
                    "probability_tolerance_absolute": 0.01,
                },
                "adapter": {
                    "kind": "model_id_only",
                    "old_model_id": "old-decision-model",
                    "envelope_transform": "none",
                    "applied_outside_timed_interval": True,
                },
                "settings": {
                    "question_counts": [q],
                    "state_counts": [s],
                    "concurrencies": list(gate.CONCURRENCIES),
                    "throughput_rounds": gate.ROUNDS,
                    "throughput_workflows_per_round_per_arm": gate.MIN_WORKFLOWS_PER_ROUND,
                    "variants_per_shape": 4,
                    "seed": 17,
                    "warmup_workflows_per_arm": 2,
                    "latency_workflows_per_arm": 16,
                    "parity_policy": "require",
                    "timed_semantic_evidence": True,
                    "arrival_policy": gate.ARRIVAL_POLICY,
                    "metrics_collection": {"old": False, "new": True},
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
                "cases": len(case_ids),
                "passed_cases": len(case_ids),
                "failed_cases": 0,
                "absolute_probability_delta": {
                    "count": probability_count,
                    "max": 0.001,
                },
                "probability_tolerance_absolute": 0.01,
            }
            raw_receipt = directory / "receipt.json"
            raw_preflight = directory / "preflight-summary.json"
            raw_workflows = directory / "workflows.jsonl"
            extra = {}
            audit_rows = [
                _audit_record(case_id, q, s, model_id) for case_id in case_ids
            ]
            timed_records = []
            for sample in samples:
                if (
                    sample["arm"] == "new"
                    and sample["phase"] == "throughput"
                    and sample["concurrency"] in (8, 32)
                ):
                    timed_body = _timed_response(model_id, q, s, sample["case_id"])
                    sample["response_sha256"] = hashlib.sha256(timed_body).hexdigest()
                    timed_records.append(
                        {
                            field: sample[field]
                            for field in (
                                "case_id",
                                "concurrency",
                                "round",
                                "sequence",
                                "request_sha256",
                                "response_sha256",
                            )
                        }
                        | {
                            "request_base64": base64.b64encode(
                                _request_body(sample["case_id"], "new", q, s, model_id)
                            ).decode("ascii"),
                            "response_base64": base64.b64encode(timed_body).decode(
                                "ascii"
                            ),
                        }
                    )
            timed_path = directory / "timed-semantic.jsonl.gz"
            timed_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(timed_path, "wt", encoding="utf-8") as handle:
                for record in timed_records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            extra["raw_timed_semantic_path"] = str(timed_path.relative_to(root))
            extra["raw_timed_semantic_sha256"] = hashlib.sha256(
                timed_path.read_bytes()
            ).hexdigest()
            for name, rows in (
                ("preflight_audit", audit_rows),
                ("audit", audit_rows),
                ("samples", samples),
                ("metrics", metrics),
            ):
                path = directory / (name.replace("_", "-") + ".jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                    encoding="utf-8",
                )
                extra[f"raw_{name}_path"] = str(path.relative_to(root))
                extra[f"raw_{name}_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
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
                    "arrival_policy": gate.ARRIVAL_POLICY,
                    "benchmark_source_sha": SOURCE,
                    "new_runtime_source_sha": SOURCE,
                    **extra,
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

    def timed_records(self, shape: dict) -> list[dict]:
        with gzip.open(self.root / shape["raw_timed_semantic_path"], "rt") as handle:
            return [json.loads(line) for line in handle]

    def write_timed_records(self, shape: dict, rows: list[dict]) -> None:
        path = self.root / shape["raw_timed_semantic_path"]
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        shape["raw_timed_semantic_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self.write_report()

    def rewrite_case_request(self, shape: dict, *, field: str) -> None:
        """Keep archive, samples and both audits mutually hash-consistent."""

        records = self.timed_records(shape)
        case_id = records[0]["case_id"]
        request = json.loads(base64.b64decode(records[0]["request_base64"]))
        if field == "state":
            request["state"]["message"] = "A substituted but valid message"
        else:
            request["questions"]["q0001"]["criteria"][
                "billing"
            ] = "A substituted but valid criterion"
        body = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(body).hexdigest()
        for record in records:
            if record["case_id"] == case_id:
                record["request_base64"] = base64.b64encode(body).decode("ascii")
                record["request_sha256"] = digest
        sample_path = self.root / shape["raw_samples_path"]
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        for sample in samples:
            if sample["arm"] == "new" and sample["case_id"] == case_id:
                sample["request_sha256"] = digest
                sample["request_bytes"] = len(body)
        sample_path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
        shape["raw_samples_sha256"] = hashlib.sha256(
            sample_path.read_bytes()
        ).hexdigest()
        for name in ("raw_preflight_audit_path", "raw_audit_path"):
            audit_path = self.root / shape[name]
            audit_rows = [
                json.loads(line) for line in audit_path.read_text().splitlines()
            ]
            target = next(row for row in audit_rows if row["case_id"] == case_id)
            target["new_http"]["request_sha256"] = digest
            audit_path.write_text(
                "\n".join(json.dumps(row) for row in audit_rows) + "\n"
            )
            shape[name.replace("_path", "_sha256")] = hashlib.sha256(
                audit_path.read_bytes()
            ).hexdigest()
        self.write_timed_records(shape, records)

    def validate(self, **kwargs: object) -> dict:
        return gate.validate_report(
            self.path,
            source_sha=SOURCE,
            qualification=_qualification(),
            candidate_ref=CANDIDATE_REF,
            **kwargs,
        )

    def test_complete_six_model_paired_measurement_passes(self) -> None:
        result = self.validate(run_id="42", run_attempt="1")
        self.assertEqual(set(result["models"]), gate.MODEL_IDS)
        self.assertEqual(result["source_sha"], SOURCE)
        self.assertAlmostEqual(result["high_load_throughput_geomean"], 1.6)
        for model in result["models"].values():
            self.assertEqual(
                set(model["high_load_shape_ratios"]),
                {"q8_s8_c32", "q32_s32_c32"},
            )
            self.assertAlmostEqual(model["high_load_throughput_geomean"], 1.6)

    def test_old_v4_report_without_live_process_proof_is_rejected(self) -> None:
        model = self.report["models"][0]
        for field in (
            "old_core_source_kind",
            "old_adapter_source_sha256",
            "old_source_declaration",
            "old_attestation_path",
            "old_attestation_sha256",
        ):
            model.pop(field)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "mandatory live old process proof"):
            self.validate()

    def test_old_v4_schema_is_rejected_even_with_current_proof_fields(self) -> None:
        self.report["schema_version"] = "decision-paired-release-v4"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "performance report schema"):
            self.validate()

    def test_old_v5_schema_is_rejected_even_with_full_snapshot_proof(self) -> None:
        self.report["schema_version"] = "decision-paired-release-v5"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "performance report schema"):
            self.validate()

    def test_full_snapshot_proof_is_required_and_bound_to_declaration(self) -> None:
        model = self.report["models"][0]
        self.validate()
        original = copy.deepcopy(model)
        model.pop("old_snapshot_verifications")
        self.write_report()
        with self.assertRaisesRegex(ValueError, "old snapshot verifications"):
            self.validate()
        model.update(copy.deepcopy(original))
        model["old_snapshot_verifications"][1]["full_snapshot_sha256"] = "5" * 64
        self.write_report()
        with self.assertRaisesRegex(ValueError, "declared old full snapshot"):
            self.validate()
        model.update(copy.deepcopy(original))
        model["old_snapshot_verifications"][1]["selected_data_content_id"] = "5" * 64
        self.write_report()
        with self.assertRaisesRegex(ValueError, "old selected-data content identity"):
            self.validate()
        model.update(copy.deepcopy(original))
        model["old_artifact_layout"] = "data_only"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "old artifact layout"):
            self.validate()

    def test_imported_old_core_attestations_are_bound_to_raw_evidence(self) -> None:
        model = self.report["models"][0]
        adapter_sha = "7" * 64
        observation = {
            "schema_version": "decision-old-baseline-attestation-v1",
            "challenge": "8" * 64,
            "pid": 1,
            "process_start_ticks": 12345,
            "adapter_path_sha256": _text_sha("/adapter.py"),
            "adapter_sha256": adapter_sha,
            "imported_module_sha256": _text_sha("old_core"),
            "imported_core_path_sha256": _text_sha("/core.py"),
            "core_mount_sha256": OLD_CORE,
            "loaded_artifact_root_sha256": _text_sha("/artifact"),
            "loaded_artifact_content_id": model["same_old_new_artifact_content_id"],
            "model_id": model["model_id"],
            "revision": MODEL_REVISION,
        }
        evidence_path = self.root / "raw" / "old-attestation.json"
        evidence = {
            "schema_version": "decision-old-baseline-attestation-v1",
            "observations": [observation, {**observation, "challenge": "9" * 64}],
        }
        model["old_adapter_source_sha256"] = adapter_sha
        model["old_attestation_path"] = "raw/old-attestation.json"
        model["old_attestation_sha256"] = _save(evidence_path, evidence)
        self.write_report()
        self.validate()
        evidence["observations"][1]["pid"] = True
        model["old_attestation_sha256"] = _save(evidence_path, evidence)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "old pid"):
            self.validate()
        evidence["observations"][1]["pid"] = 1
        evidence["observations"][1]["imported_core_path_sha256"] = _text_sha(
            "/unused.py"
        )
        model["old_attestation_sha256"] = _save(evidence_path, evidence)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "imported_core_path_sha256"):
            self.validate()
        evidence["observations"][1]["imported_core_path_sha256"] = _text_sha("/core.py")
        evidence["observations"][1]["loaded_artifact_content_id"] = "0" * 64
        model["old_attestation_sha256"] = _save(evidence_path, evidence)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "loaded_artifact_content_id"):
            self.validate()
        evidence["observations"][1] = {**observation, "challenge": "9" * 64}
        _save(evidence_path, evidence)
        with self.assertRaisesRegex(ValueError, "content changed"):
            self.validate()

    def test_timed_concurrent_evidence_cannot_omit_or_duplicate_a_workflow(
        self,
    ) -> None:
        shape = self.report["models"][0]["shapes"][2]
        records = self.timed_records(shape)
        self.assertEqual(len(records), 2 * gate.ROUNDS * gate.MIN_WORKFLOWS_PER_ROUND)
        for changed in (records[:-1], [*records, records[0]]):
            with self.subTest(records=len(changed)):
                self.write_timed_records(shape, changed)
                with self.assertRaisesRegex(
                    ValueError,
                    "timed semantic .*incomplete|timed semantic record is missing or duplicated",
                ):
                    self.validate()

    def test_timed_response_hash_must_bind_the_actual_body(self) -> None:
        shape = self.report["models"][0]["shapes"][2]
        records = self.timed_records(shape)
        body = json.loads(base64.b64decode(records[0]["response_base64"]))
        body["results"][0]["answers"]["q0000"]["noul"] = 0.75
        records[0]["response_base64"] = base64.b64encode(
            json.dumps(body).encode()
        ).decode()
        self.write_timed_records(shape, records)
        with self.assertRaisesRegex(ValueError, "timed semantic response hash"):
            self.validate()

    def test_timed_request_hash_must_bind_the_formal_case(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        records = self.timed_records(shape)
        request = json.loads(base64.b64decode(records[0]["request_base64"]))
        request["state"] = "a different logical state"
        records[0]["request_base64"] = base64.b64encode(
            json.dumps(request).encode()
        ).decode()
        self.write_timed_records(shape, records)
        with self.assertRaisesRegex(ValueError, "timed semantic request hash"):
            self.validate()

    def test_coherently_rehashed_state_cannot_replace_canonical_case(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        self.rewrite_case_request(shape, field="state")
        with self.assertRaisesRegex(
            ValueError, "timed semantic request differs from canonical case"
        ):
            self.validate()

    def test_coherently_rehashed_criteria_cannot_replace_canonical_case(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        self.rewrite_case_request(shape, field="criteria")
        with self.assertRaisesRegex(
            ValueError, "timed semantic request differs from canonical case"
        ):
            self.validate()

    def test_timed_probability_drift_fails_even_with_rehashed_evidence(self) -> None:
        shape = self.report["models"][0]["shapes"][2]
        records = self.timed_records(shape)
        record = next(
            row for row in records if row["concurrency"] == 32 and row["round"] == 2
        )
        body = json.loads(base64.b64decode(record["response_base64"]))
        body["results"][0]["answers"]["q0000"]["noul"] = 0.75
        wire = json.dumps(body).encode()
        record["response_base64"] = base64.b64encode(wire).decode()
        record["response_sha256"] = hashlib.sha256(wire).hexdigest()
        sample_path = self.root / shape["raw_samples_path"]
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        target = next(
            row
            for row in samples
            if row["arm"] == "new"
            and row["phase"] == "throughput"
            and row["round"] == record["round"]
            and row["concurrency"] == record["concurrency"]
            and row["sequence"] == record["sequence"]
        )
        target["response_sha256"] = record["response_sha256"]
        sample_path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
        shape["raw_samples_sha256"] = hashlib.sha256(
            sample_path.read_bytes()
        ).hexdigest()
        self.write_timed_records(shape, records)
        with self.assertRaisesRegex(ValueError, "timed semantic probability tolerance"):
            self.validate()

    def test_timed_response_token_usage_must_match_sealed_audit(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        records = self.timed_records(shape)
        body = json.loads(base64.b64decode(records[0]["response_base64"]))
        body["usage"]["input_tokens"] += 1
        wire = json.dumps(body).encode()
        records[0]["response_base64"] = base64.b64encode(wire).decode()
        records[0]["response_sha256"] = hashlib.sha256(wire).hexdigest()
        sample_path = self.root / shape["raw_samples_path"]
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        target = next(
            row
            for row in samples
            if row["arm"] == "new"
            and row["phase"] == "throughput"
            and row["concurrency"] == records[0]["concurrency"]
            and row["round"] == records[0]["round"]
            and row["sequence"] == records[0]["sequence"]
        )
        target["response_sha256"] = records[0]["response_sha256"]
        sample_path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
        shape["raw_samples_sha256"] = hashlib.sha256(
            sample_path.read_bytes()
        ).hexdigest()
        self.write_timed_records(shape, records)
        with self.assertRaisesRegex(ValueError, "timed semantic input token usage"):
            self.validate()

    def test_compressed_timed_archive_has_a_decompression_limit(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_timed_semantic_path"]
        with gzip.open(path, "wb") as handle:
            for _ in range(81):
                handle.write(b"x" * (1024 * 1024))
        shape["raw_timed_semantic_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "timed semantic archive is too large"):
            self.validate()

    def test_compressed_timed_archive_has_an_input_limit(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_timed_semantic_path"]
        path.write_bytes(b"x" * (timed_semantics.MAX_COMPRESSED_ARCHIVE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "is too large"):
            gate._evidence(
                self.root,
                shape["raw_timed_semantic_path"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
                "timed archive",
            )
        with self.assertRaisesRegex(ValueError, "compressed archive is too large"):
            timed_semantics._read_archive(path)

        class StaleStatPath:
            def stat(self):
                return SimpleNamespace(st_size=0)

            def open(self, *args):
                return path.open(*args)

        with self.assertRaisesRegex(ValueError, "compressed archive is too large"):
            timed_semantics._read_archive(StaleStatPath())

    def test_timed_archive_rejects_a_trailing_member_or_garbage(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_timed_semantic_path"]
        original = path.read_bytes()
        for suffix in (gzip.compress(b""), b"trailing-garbage"):
            with self.subTest(suffix=suffix[:8]):
                path.write_bytes(original + suffix)
                shape["raw_timed_semantic_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                self.write_report()
                with self.assertRaisesRegex(ValueError, "trailing or incomplete data"):
                    self.validate()

    def test_timed_response_body_has_a_per_request_limit(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        records = self.timed_records(shape)
        records[0]["response_base64"] = base64.b64encode(
            b"x" * (timed_semantics.MAX_RESPONSE_BYTES + 1)
        ).decode()
        self.write_timed_records(shape, records)
        with self.assertRaisesRegex(
            ValueError, "timed semantic response body is too large"
        ):
            self.validate()

    def test_an_older_attempt_cannot_qualify_the_current_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected run attempt"):
            self.validate(run_id="42", run_attempt="2")

    def test_report_must_join_candidate_and_every_qualified_model(self) -> None:
        self.report["candidate_ref"] = "ghcr.io/example/other@sha256:" + "5" * 64
        self.write_report()
        with self.assertRaisesRegex(ValueError, "performance candidate"):
            self.validate()
        self.report["candidate_ref"] = CANDIDATE_REF
        self.report["models"][0]["same_old_new_artifact_content_id"] = "5" * 64
        self.write_report()
        with self.assertRaisesRegex(ValueError, "qualified artifact"):
            self.validate()

    def test_contradictory_raw_sample_blocks_release_even_with_updated_hash(
        self,
    ) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_samples_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["status_code"] = 503
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_samples_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "raw sample contradicts"):
            self.validate()

    def test_sample_schedule_hash_must_match_paired_logical_cases(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_samples_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["logical_schedule_sha256"] = "0" * 64
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_samples_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "paired logical schedule identity"):
            self.validate()

    def test_different_old_new_case_ids_block_release(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_workflows_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if (
                row["arm"] == "new"
                and row["phase"] == "throughput"
                and row["concurrency"] == 1
                and row["round"] == 0
                and row["sequence"] == 0
            ):
                row["case_id"] = "q0032_s0001_v001"
                break
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_workflows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "old/new workflow case IDs differ"):
            self.validate()

    def test_contradictory_raw_audit_and_metric_capture_block_release(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        audit_path = self.root / shape["raw_audit_path"]
        audit_rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
        audit_rows[0]["passed"] = False
        audit_path.write_text("\n".join(json.dumps(row) for row in audit_rows) + "\n")
        shape["raw_audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "failed or different case"):
            self.validate()
        audit_rows[0]["passed"] = True
        audit_path.write_text("\n".join(json.dumps(row) for row in audit_rows) + "\n")
        shape["raw_audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        metrics_path = self.root / shape["raw_metrics_path"]
        metrics_rows = [
            json.loads(line) for line in metrics_path.read_text().splitlines()
        ]
        metrics_rows[0]["delta"]["counters"]["physical_batch_rows"] += 1
        metrics_path.write_text(
            "\n".join(json.dumps(row) for row in metrics_rows) + "\n"
        )
        shape["raw_metrics_sha256"] = hashlib.sha256(
            metrics_path.read_bytes()
        ).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "metric delta differs"):
            self.validate()

    def test_new_metric_rows_must_cover_every_successful_workflow(self) -> None:
        # At q32/s32, three rounds contain 98,304 logical rows; a plausible
        # looking 400-row metric receipt must not qualify the same workload.
        shape = self.report["models"][0]["shapes"][2]
        path = self.root / shape["raw_metrics_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        capture = next(
            row for row in rows if row["concurrency"] == 32 and row["round"] == 0
        )
        capture["delta"]["counters"]["physical_batch_rows"] = 400
        capture["after"]["counters"]["physical_batch_rows"] = (
            capture["before"]["counters"]["physical_batch_rows"] + 400
        )
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "new round physical row coverage"):
            self.validate()

    def test_new_metric_preparations_must_cover_workflows(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_metrics_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        capture = rows[0]
        capture["delta"]["counters"]["row_preparations"] = 1
        capture["after"]["counters"]["row_preparations"] = (
            capture["before"]["counters"]["row_preparations"] + 1
        )
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "new round row preparation coverage"):
            self.validate()

    def test_each_round_respects_configured_physical_batch_capacity(self) -> None:
        shape = self.report["models"][0]["shapes"][2]
        path = self.root / shape["raw_metrics_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        capture = next(
            row for row in rows if row["concurrency"] == 32 and row["round"] == 0
        )
        # 3,192 B8 forwards cannot contain 32,768 rows, even though the
        # unchanged shape-level total would still look plausible.
        capture["delta"]["counters"]["physical_batches"] = 3192
        capture["after"]["counters"]["physical_batches"] = (
            capture["before"]["counters"]["physical_batches"] + 3192
        )
        for bucket in (*map(str, gate.PHYSICAL_BATCH_BUCKETS), "+Inf"):
            value = 3192 if bucket not in ("1", "2") else 0
            capture["delta"]["physical_batch_size_buckets"][bucket] = value
            capture["after"]["physical_batch_size_buckets"][bucket] = (
                capture["before"]["physical_batch_size_buckets"][bucket] + value
            )
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "round physical batch capacity"):
            self.validate()

    def test_sample_peak_http_inflight_cannot_exceed_declared_concurrency(
        self,
    ) -> None:
        # Old q32/s32 fan-out has 1,024 calls per round. Declaring c32 does
        # not license sending all 1,024 simultaneously.
        shape = self.report["models"][0]["shapes"][2]
        path = self.root / shape["raw_samples_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        selected = [
            row
            for row in rows
            if row["arm"] == "old"
            and row["phase"] == "throughput"
            and row["concurrency"] == 32
            and row["round"] == 0
        ]
        self.assertEqual(len(selected), 1024)
        for row in selected:
            row["started_offset_ms"] = 0.0
            row["completed_offset_ms"] = 1000.0
            row["latency_ms"] = 1000.0
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_samples_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "peak in-flight HTTP"):
            self.validate()

    def test_low_load_sequences_share_the_same_inflight_limit(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        sample_path = self.root / shape["raw_samples_path"]
        workflow_path = self.root / shape["raw_workflows_path"]
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        workflows = [
            json.loads(line) for line in workflow_path.read_text().splitlines()
        ]

        def selected(row: dict) -> bool:
            return (
                row["arm"] == "new"
                and row["phase"] == "latency"
                and row["concurrency"] == 1
            )

        self.assertEqual(sum(selected(row) for row in samples), 16)
        for row in (*samples, *workflows):
            if selected(row):
                row["started_offset_ms"] = 100.0
                row["completed_offset_ms"] = 110.0
                row["latency_ms"] = 10.0
        sample_path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
        workflow_path.write_text("\n".join(json.dumps(row) for row in workflows) + "\n")
        shape["raw_samples_sha256"] = hashlib.sha256(
            sample_path.read_bytes()
        ).hexdigest()
        shape["raw_workflows_sha256"] = hashlib.sha256(
            workflow_path.read_bytes()
        ).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "low-load phases or arms overlap"):
            self.validate()

    def test_cumulative_metric_counters_cannot_reset_between_waves(self) -> None:
        shape = self.report["models"][0]["shapes"][2]
        path = self.root / shape["raw_metrics_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        first = next(
            row for row in rows if row["concurrency"] == 1 and row["round"] == 0
        )
        second = next(
            row for row in rows if row["concurrency"] == 1 and row["round"] == 1
        )
        for name, delta in second["delta"]["counters"].items():
            second["before"]["counters"][name] = 0
            second["after"]["counters"][name] = delta
        for name, delta in second["delta"]["physical_batch_size_buckets"].items():
            second["before"]["physical_batch_size_buckets"][name] = 0
            second["after"]["physical_batch_size_buckets"][name] = delta
        _refresh_metric_hash(second["before"])
        _refresh_metric_hash(second["after"])
        self.assertGreater(first["after"]["counters"]["physical_batches"], 0)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(
            ValueError, "metric counter reset between measured waves"
        ):
            self.validate()

    def test_metric_snapshots_allow_intervening_warmup_work(self) -> None:
        model = self.report["models"][0]
        for shape_index, shape in enumerate(model["shapes"]):
            path = self.root / shape["raw_metrics_path"]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                if shape_index == 0 and row["concurrency"] == 1:
                    continue
                for phase in ("before", "after"):
                    counters = row[phase]["counters"]
                    counters["row_preparation_seconds"] += 0.01
                    counters["row_preparations"] += 1
                    counters["physical_batches"] += 1
                    counters["physical_batch_rows"] += 4
                    buckets = row[phase]["physical_batch_size_buckets"]
                    for bucket in (*map(str, gate.PHYSICAL_BATCH_BUCKETS), "+Inf"):
                        if bucket not in ("1", "2"):
                            buckets[bucket] += 1
                    _refresh_metric_hash(row[phase])
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            shape["raw_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        self.assertEqual(set(self.validate()["models"]), gate.MODEL_IDS)

    def test_one_metrics_hash_cannot_name_different_cumulative_values(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_metrics_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertNotEqual(
            rows[0]["before"]["counters"], rows[1]["before"]["counters"]
        )
        rows[1]["before"]["response_sha256"] = rows[0]["before"]["response_sha256"]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(
            ValueError, "identical metrics response hash has different counters"
        ):
            self.validate()

    def test_warmup_cannot_overlap_measured_throughput(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        workflow_path = self.root / shape["raw_workflows_path"]
        sample_path = self.root / shape["raw_samples_path"]
        workflows = [
            json.loads(line) for line in workflow_path.read_text().splitlines()
        ]
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        target = next(
            row
            for row in workflows
            if row["concurrency"] == 1
            and row["arm"] == "old"
            and row["phase"] == "warmup"
            and row["round"] == 1
        )
        wave_start = min(
            row["started_offset_ms"]
            for row in workflows
            if row["concurrency"] == 1
            and row["arm"] == "old"
            and row["phase"] == "throughput"
            and row["round"] == 0
        )
        shift = wave_start + 1 - target["started_offset_ms"]
        for row in (
            target,
            *(
                sample
                for sample in samples
                if sample["concurrency"] == 1
                and sample["arm"] == "old"
                and sample["phase"] == "warmup"
                and sample["round"] == 1
            ),
        ):
            row["started_offset_ms"] += shift
            row["completed_offset_ms"] += shift
        workflow_path.write_text("\n".join(json.dumps(row) for row in workflows) + "\n")
        sample_path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
        shape["raw_workflows_sha256"] = hashlib.sha256(
            workflow_path.read_bytes()
        ).hexdigest()
        shape["raw_samples_sha256"] = hashlib.sha256(
            sample_path.read_bytes()
        ).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "low-load phases or arms overlap"):
            self.validate()

    def test_concurrency_cells_must_not_overlap(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        workflow_path = self.root / shape["raw_workflows_path"]
        sample_path = self.root / shape["raw_samples_path"]
        workflows = [
            json.loads(line) for line in workflow_path.read_text().splitlines()
        ]
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        c1_end = max(
            row["completed_offset_ms"] for row in workflows if row["concurrency"] == 1
        )
        c8_start = min(
            row["started_offset_ms"] for row in workflows if row["concurrency"] == 8
        )
        shift = c8_start - c1_end + 100
        self.assertGreater(shift, 0)
        for row in (*workflows, *samples):
            if row["concurrency"] == 8:
                row["started_offset_ms"] -= shift
                row["completed_offset_ms"] -= shift
        workflow_path.write_text("\n".join(json.dumps(row) for row in workflows) + "\n")
        sample_path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
        shape["raw_workflows_sha256"] = hashlib.sha256(
            workflow_path.read_bytes()
        ).hexdigest()
        shape["raw_samples_sha256"] = hashlib.sha256(
            sample_path.read_bytes()
        ).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "concurrency cells overlap"):
            self.validate()

    def test_metric_histogram_must_count_every_physical_batch(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_metrics_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        capture = rows[0]
        for bucket in ("4", "8", "16", "32", "64", "128", "256", "512", "1024", "+Inf"):
            capture["delta"]["physical_batch_size_buckets"][bucket] -= 1
            capture["after"]["physical_batch_size_buckets"][bucket] -= 1
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "histogram snapshot batch count"):
            self.validate()

    def test_metric_histogram_buckets_must_be_cumulative(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_metrics_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        capture = rows[0]
        capture["delta"]["physical_batch_size_buckets"]["8"] -= 1
        capture["after"]["physical_batch_size_buckets"]["8"] -= 1
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        shape["raw_metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "metric histogram is not cumulative"):
            self.validate()

    def test_requires_same_source_and_candidate_image(self) -> None:
        self.report["new_runtime_image_source_sha"] = "9" * 40
        self.write_report()
        with self.assertRaisesRegex(ValueError, "candidate image source"):
            self.validate()

    def test_requires_all_six_models_three_shapes_and_three_cells(self) -> None:
        self.report["models"].pop()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "exactly six"):
            self.validate()
        _, self.report = _fixture(self.root)
        self.report["models"][0]["shapes"][0]["cells"].pop()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "three concurrency cells"):
            self.validate()

    def test_raw_files_are_hashed_and_cannot_escape_bundle(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        raw = self.root / shape["raw_receipt_path"]
        raw.write_text("{}")
        with self.assertRaisesRegex(ValueError, "content changed"):
            self.validate()
        _, self.report = _fixture(self.root)
        self.report["models"][0]["shapes"][0]["raw_receipt_path"] = "../elsewhere.json"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "under raw"):
            self.validate()

    def test_semantic_mismatch_and_failures_block_release(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        shape["preflight_mismatch_counts"] = {"probability_tolerance": 1}
        self.write_report()
        with self.assertRaisesRegex(ValueError, "semantic mismatch"):
            self.validate()

    def test_recomputed_ratio_rejects_fabricated_gain(self) -> None:
        cell = self.report["models"][0]["shapes"][0]["cells"][0]
        cell["new_over_old_decisions_per_second"] = 100.0
        self.write_report()
        with self.assertRaisesRegex(ValueError, "throughput ratio"):
            self.validate()

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
            self.validate()

    def test_physical_batch_telemetry_is_consistent(self) -> None:
        cell = self.report["models"][0]["shapes"][0]["cells"][0]
        cell["new_physical_batch_rows"] = 900
        self.write_report()
        with self.assertRaisesRegex(ValueError, "physical batch"):
            self.validate()

    def test_larger_batch_needs_independent_tuning_qualification(self) -> None:
        self.report["models"][0]["new_physical_batch_size"] = 16
        self.write_report()
        with self.assertRaisesRegex(ValueError, "independent tuning qualification"):
            self.validate()

    def test_modest_gain_with_bounded_cell_regression_qualifies(self) -> None:
        ratios = {
            (model_id, q, s, 32): 1.06
            for model_id in gate.MODEL_IDS
            for q, s in gate.HIGH_LOAD_SHAPES
        }
        ratios[(sorted(gate.MODEL_IDS)[0], 32, 1, 8)] = 0.96
        self.path, self.report = _fixture(self.root, throughput_ratios=ratios)
        result = self.validate()
        self.assertAlmostEqual(result["high_load_throughput_geomean"], 1.06)
        for model in result["models"].values():
            self.assertAlmostEqual(model["high_load_throughput_geomean"], 1.06)
            for ratio in model["high_load_shape_ratios"].values():
                self.assertAlmostEqual(ratio, 1.06)

    def test_flat_model_qualifies_when_other_models_raise_overall_gain(self) -> None:
        self.path, self.report = _fixture(self.root, slowdown="no_gain")
        result = self.validate()
        self.assertAlmostEqual(
            result["models"][sorted(gate.MODEL_IDS)[0]][
                "high_load_throughput_geomean"
            ],
            1.0,
        )
        self.assertGreaterEqual(result["high_load_throughput_geomean"], 1.05)

    def test_latency_only_win_cannot_replace_unpaired_throughput_gain(self) -> None:
        ratios = {
            (model_id, q, s, 32): 1.0
            for model_id in gate.MODEL_IDS
            for q, s in gate.HIGH_LOAD_SHAPES
        }
        self.path, self.report = _fixture(
            self.root, slowdown="latency_only", throughput_ratios=ratios
        )
        with self.assertRaisesRegex(ValueError, "throughput gain is below 5%"):
            self.validate()

    def test_one_shape_win_cannot_hide_low_overall_gain(self) -> None:
        ratios = {
            (model_id, q, s, 32): 1.0
            for model_id in gate.MODEL_IDS
            for q, s in gate.HIGH_LOAD_SHAPES
        }
        first_model = sorted(gate.MODEL_IDS)[0]
        ratios[(first_model, 8, 8, 32)] = 1.20
        ratios[(first_model, 32, 32, 32)] = 0.96
        self.path, self.report = _fixture(self.root, throughput_ratios=ratios)
        with self.assertRaisesRegex(ValueError, "throughput gain is below 5%"):
            self.validate()

    def test_one_model_regression_blocks_aggregate_gain(self) -> None:
        ratios = {
            (sorted(gate.MODEL_IDS)[0], q, s, 32): 0.99
            for q, s in gate.HIGH_LOAD_SHAPES
        }
        self.path, self.report = _fixture(self.root, throughput_ratios=ratios)
        with self.assertRaisesRegex(ValueError, "material high-load model regression"):
            self.validate()

    def test_high_load_regression_blocks_release(self) -> None:
        self.path, self.report = _fixture(self.root, slowdown="regression")
        with self.assertRaisesRegex(
            ValueError, "material high-load throughput regression"
        ):
            self.validate()

    def test_interleaved_waves_must_match_raw_workflow_times(self) -> None:
        shape = self.report["models"][0]["shapes"][0]
        path = self.root / shape["raw_workflows_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if (
                row["concurrency"] == 1
                and row["phase"] == "throughput"
                and row["round"] == 0
                and row["arm"] == "new"
            ):
                row["started_offset_ms"] -= 1000
                row["completed_offset_ms"] -= 1000
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
        )
        shape["raw_workflows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_report()
        with self.assertRaisesRegex(ValueError, "alternating waves"):
            self.validate()


class TimedAnswerPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = {
            "n": {"type": "noul", "instructions": "Is action needed?"},
            "c": {
                "type": "choice",
                "instructions": "Choose a queue.",
                "criteria": {"billing": "Billing", "service": "Service"},
            },
            "s": {
                "type": "score",
                "instructions": "Rate urgency.",
                "criteria": ["Routine", "Same day", "Immediate"],
            },
        }
        self.audited = [
            {"question_id": "n", "type": "noul", "old_probability": 0.5},
            {
                "question_id": "c",
                "type": "choice",
                "old_outcome": "billing",
                "old_probabilities": {"billing": 0.501, "service": 0.499},
            },
            {
                "question_id": "s",
                "type": "score",
                "old_score": 1.0,
                "old_probabilities": {"0": 0.25, "1": 0.5, "2": 0.25},
            },
        ]
        self.actual = {
            "n": {"type": "noul", "noul": 0.501},
            "c": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.004,
                "probabilities": {"billing": 0.502, "service": 0.498},
            },
            "s": {
                "type": "score",
                "score": 1.01,
                "confidence": 0.25015,
                "legend": {"0": "Routine", "1": "Same day", "2": "Immediate"},
                "probabilities": {"0": 0.245, "1": 0.5, "2": 0.255},
            },
        }

    def test_mixed_answers_preserve_formal_audit_tolerances(self) -> None:
        timed_semantics._compare_answers(self.actual, self.audited, self.questions)
        self.actual["c"]["choice"] = "service"
        self.actual["c"]["probabilities"] = {"billing": 0.499, "service": 0.501}
        self.actual["c"]["confidence"] = 0.002
        with self.assertRaisesRegex(ValueError, "Choice outcome"):
            timed_semantics._compare_answers(self.actual, self.audited, self.questions)
        self.actual["c"]["choice"] = "billing"
        self.actual["c"]["probabilities"] = {"billing": 0.502, "service": 0.498}
        self.actual["c"]["confidence"] = 0.004
        self.actual["s"]["score"] = 1.03
        with self.assertRaisesRegex(ValueError, "Score weighted mean"):
            timed_semantics._compare_answers(self.actual, self.audited, self.questions)

    def test_duplicate_keys_and_nonfinite_values_are_rejected(self) -> None:
        for payload in (b'{"model":"a","model":"b"}', b'{"value":NaN}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                timed_semantics._json(payload)

    def test_independent_response_contract_rejects_invalid_math_and_fields(
        self,
    ) -> None:
        edits = (
            (
                "probability distribution",
                lambda row: row["c"]["probabilities"].update(service=0.4),
            ),
            ("Choice winner", lambda row: row["c"].update(choice="service")),
            ("confidence", lambda row: row["c"].update(confidence=0.8)),
            ("Score legend", lambda row: row["s"]["legend"].update({"1": "Wrong"})),
            ("response shape", lambda row: row["n"].update(extra=True)),
        )
        for message, edit in edits:
            with self.subTest(message=message):
                actual = copy.deepcopy(self.actual)
                edit(actual)
                with self.assertRaisesRegex(ValueError, message):
                    timed_semantics._compare_answers(
                        actual, self.audited, self.questions
                    )


if __name__ == "__main__":
    unittest.main()
