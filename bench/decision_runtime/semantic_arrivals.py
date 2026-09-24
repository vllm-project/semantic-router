"""Opt-in paired Decision workload with fixed logical-workflow arrivals.

This is exploratory evidence. The protected release gate still consumes the
existing ``semantic`` mode and does not qualify these receipts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .cases import MODELS
from .semantic_arrival_parity import TimedSemanticEvidence
from .semantic_audit import audit_cohorts
from .semantic_cases import WorkloadCase, cohort_sha256, generate_cases
from .semantic_report import percentile
from .semantic_runner import (
    MODEL_ID,
    _harness_digest,
    _metadata,
    _nonnegative_float,
    _nonnegative_int,
    _positive_csv,
    _positive_float,
    _positive_int,
    _schedule,
    _source_commit,
    _token,
)
from .semantic_transport import HttpSample, batch_url, measure_http
from .transport import Endpoint, validate_endpoint_url

SCHEMA = "decision-semantic-arrivals-v1"
TRACE_SCHEMA = "decision-logical-arrival-traces-v1"
Measure = Callable[..., HttpSample]
MAX_WAVE_CAPTURE_BYTES = 80 * 1024 * 1024


@dataclass(frozen=True)
class Arrival:
    sequence: int
    case: WorkloadCase
    offset_ns: int


def make_trace(
    cases: tuple[WorkloadCase, ...], count: int, seed: int, spacing_ns: int
) -> tuple[Arrival, ...]:
    """Select ordered cases before either arm runs; offsets never use responses."""

    return tuple(
        Arrival(sequence, case, sequence * spacing_ns)
        for sequence, case in enumerate(_schedule(cases, count, seed))
    )


def trace_record(
    trace: tuple[Arrival, ...], *, q: int, s: int, c: int, phase: str, round_: int
) -> dict[str, Any]:
    payload = {
        "question_count": q,
        "state_count": s,
        "http_concurrency_cap": c,
        "phase": phase,
        "round": round_,
        "arrivals": [
            {
                "sequence": item.sequence,
                "case_id": item.case.id,
                "offset_ns": item.offset_ns,
            }
            for item in trace
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": hashlib.sha256(canonical).hexdigest()}


def _arrival_harness_digest() -> str:
    digest = hashlib.sha256(bytes.fromhex(_harness_digest()))
    for path in (
        Path(__file__),
        Path(__file__).with_name("semantic_arrival_parity.py"),
        Path(__file__).resolve().parents[2] / "tools/ci/decision_timed_semantics.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _ranks(values: list[float]) -> dict[str, float | None]:
    return {
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def _failed_sample(
    arm: str,
    phase: str,
    round_: int,
    c: int,
    arrival: Arrival,
    spec: Any,
    started_ns: int,
) -> HttpSample:
    """Retain a bounded generic error when a client worker unexpectedly fails."""

    return HttpSample(
        arm=arm,
        phase=phase,
        round=round_,
        sequence=arrival.sequence,
        case_id=arrival.case.id,
        concurrency=c,
        state_id=spec.state_id,
        request_kind=(
            "single"
            if spec.state_id is not None or arrival.case.state_count == 1
            else "batch"
        ),
        request_sha256=spec.sha256,
        request_bytes=len(spec.body),
        decisions=spec.decisions,
        started_ns=started_ns,
        ended_ns=max(time.perf_counter_ns(), started_ns + 1),
        status_code=None,
        error_code="client_worker_error",
        response_sha256=None,
    )


def run_arrival_wave(
    trace: tuple[Arrival, ...],
    *,
    arm: str,
    phase: str,
    round_: int,
    c: int,
    old: Endpoint,
    new_single: Endpoint,
    new_batch: Endpoint,
    timeout: float,
    measure: Measure = measure_http,
) -> tuple[list[HttpSample], list[dict[str, Any]], dict[str, Any]]:
    """Release workflows on a fixed clock, then dispatch HTTP calls fairly.

    A single bounded HTTP pool is shared by all arrived workflows. At each free
    slot the dispatcher takes one pending state per workflow in round-robin
    order, so an old multi-state workflow cannot monopolize a burst.
    """

    if not trace or c < 1 or arm not in {"old", "new"}:
        raise ValueError(
            "arrival wave requires a trace, an arm, and a positive HTTP cap"
        )
    if [item.sequence for item in trace] != list(range(len(trace))):
        raise ValueError("arrival sequence must be contiguous")
    if any(item.offset_ns < 0 for item in trace) or any(
        left.offset_ns > right.offset_ns for left, right in zip(trace, trace[1:])
    ):
        raise ValueError("arrival offsets must be nonnegative and ordered")

    def endpoint_for(case: WorkloadCase) -> Endpoint:
        if arm == "old":
            return old
        return new_single if case.state_count == 1 else new_batch

    def specs_for(case: WorkloadCase) -> tuple[Any, ...]:
        return case.old_singles if arm == "old" else (case.new_request,)

    wave_start_ns = time.perf_counter_ns()
    arrived: dict[int, int] = {}
    pending: dict[int, deque[Any]] = {}
    ready: deque[int] = deque()
    active: dict[Future[HttpSample], tuple[Arrival, Any, int]] = {}
    samples: list[HttpSample] = []
    captured_body_bytes = 0
    client_completed_ns: dict[tuple[int, str | None], int] = {}
    next_arrival = 0
    peak_dispatched = 0
    peak_pending_http = 0

    with ThreadPoolExecutor(max_workers=c) as pool:
        while next_arrival < len(trace) or ready or active:
            now_ns = time.perf_counter_ns()
            while (
                next_arrival < len(trace)
                and wave_start_ns + trace[next_arrival].offset_ns <= now_ns
            ):
                item = trace[next_arrival]
                arrived[item.sequence] = time.perf_counter_ns()
                pending[item.sequence] = deque(specs_for(item.case))
                ready.append(item.sequence)
                next_arrival += 1
            peak_pending_http = max(
                peak_pending_http, sum(len(queue) for queue in pending.values())
            )
            while ready and len(active) < c:
                sequence = ready.popleft()
                spec = pending[sequence].popleft()
                if pending[sequence]:
                    ready.append(sequence)
                item = trace[sequence]
                submitted_ns = time.perf_counter_ns()
                future = pool.submit(
                    measure,
                    endpoint_for(item.case),
                    spec,
                    case_id=item.case.id,
                    concurrency=c,
                    phase=phase,
                    round_number=round_,
                    sequence=sequence,
                    timeout_seconds=timeout,
                    capture_response=phase == "throughput",
                )
                active[future] = (item, spec, submitted_ns)
                peak_dispatched = max(peak_dispatched, len(active))
            if active:
                next_deadline_ns = (
                    wave_start_ns + trace[next_arrival].offset_ns
                    if next_arrival < len(trace)
                    else None
                )
                timeout_seconds = (
                    max(0, (next_deadline_ns - time.perf_counter_ns()) / 1e9)
                    if next_deadline_ns is not None
                    else None
                )
                completed, _ = wait(
                    active, timeout=timeout_seconds, return_when=FIRST_COMPLETED
                )
                for future in completed:
                    item, spec, submitted_ns = active.pop(future)
                    try:
                        sample = future.result()
                    except Exception:
                        sample = _failed_sample(
                            arm, phase, round_, c, item, spec, submitted_ns
                        )
                    if (
                        sample.request_body is not None
                        and sample.response_body is not None
                    ):
                        body_bytes = len(sample.request_body) + len(
                            sample.response_body
                        )
                        if captured_body_bytes + body_bytes <= MAX_WAVE_CAPTURE_BYTES:
                            captured_body_bytes += body_bytes
                        else:
                            sample = replace(
                                sample, request_body=None, response_body=None
                            )
                    samples.append(sample)
                    client_completed_ns[item.sequence, spec.state_id] = max(
                        time.perf_counter_ns(), sample.ended_ns
                    )
            elif next_arrival < len(trace):
                delay = (
                    wave_start_ns
                    + trace[next_arrival].offset_ns
                    - time.perf_counter_ns()
                ) / 1e9
                if delay > 0:
                    time.sleep(delay)

    by_sequence: dict[int, list[HttpSample]] = {}
    for sample in samples:
        by_sequence.setdefault(sample.sequence, []).append(sample)
    workflows = []
    for item in trace:
        calls = by_sequence.get(item.sequence, [])
        expected = item.case.state_count if arm == "old" else 1
        if len(calls) != expected:
            raise ValueError("arrival workflow HTTP inventory is incomplete")
        planned_ns = wave_start_ns + item.offset_ns
        arrived_ns = arrived[item.sequence]
        first_send_ns = min(call.started_ns for call in calls)
        last_body_ns = max(call.ended_ns for call in calls)
        completed_ns = max(
            client_completed_ns[call.sequence, call.state_id] for call in calls
        )
        errors = sorted(
            call.error_code for call in calls if call.error_code is not None
        )
        workflows.append(
            {
                "arm": arm,
                "phase": phase,
                "round": round_,
                "sequence": item.sequence,
                "case_id": item.case.id,
                "decisions": item.case.decisions,
                "http_calls": expected,
                "scheduled_ns": planned_ns,
                "arrived_ns": arrived_ns,
                "first_send_ns": first_send_ns,
                "last_body_ns": last_body_ns,
                "completed_ns": completed_ns,
                "arrival_jitter_ms": (arrived_ns - planned_ns) / 1e6,
                "client_queue_ms": (first_send_ns - arrived_ns) / 1e6,
                "latency_ms": (completed_ns - planned_ns) / 1e6,
                "post_body_validation_ms": (completed_ns - last_body_ns) / 1e6,
                "success": not errors,
                "error_codes": errors,
            }
        )
    samples.sort(key=lambda item: (item.sequence, item.state_id or ""))
    last_completed_ns = max(item["completed_ns"] for item in workflows)
    window_seconds = (last_completed_ns - wave_start_ns) / 1e9
    wave = {
        "arm": arm,
        "phase": phase,
        "round": round_,
        "wave_start_ns": wave_start_ns,
        "window_seconds": window_seconds,
        "attempted_workflows": len(workflows),
        "successful_workflows": sum(item["success"] for item in workflows),
        "attempted_decisions": sum(item["decisions"] for item in workflows),
        "successful_decisions": sum(
            item["decisions"] for item in workflows if item["success"]
        ),
        "peak_dispatched_http": peak_dispatched,
        "peak_pending_http": peak_pending_http,
        "captured_timed_body_bytes": captured_body_bytes,
        "max_arrival_jitter_ms": max(item["arrival_jitter_ms"] for item in workflows),
    }
    return samples, workflows, wave


def summarize_arrivals(
    workflows: list[dict[str, Any]],
    waves: list[dict[str, Any]],
    *,
    max_jitter_ms: float,
    extra_reasons: tuple[str, ...] = (),
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    reasons = list(extra_reasons)
    for arm in ("old", "new"):
        arm_workflows = [item for item in workflows if item["arm"] == arm]
        arm_waves = [item for item in waves if item["arm"] == arm]
        successes = [item for item in arm_workflows if item["success"]]
        seconds = sum(item["window_seconds"] for item in arm_waves)
        failed = len(arm_workflows) - len(successes)
        if failed:
            reasons.append(f"{arm}_failed_workflows")
        if any(item["max_arrival_jitter_ms"] > max_jitter_ms for item in arm_waves):
            reasons.append(f"{arm}_arrival_jitter_exceeded")
        arms[arm] = {
            "attempted_workflows": len(arm_workflows),
            "successful_workflows": len(successes),
            "failed_workflows": failed,
            "successful_decisions": sum(item["decisions"] for item in successes),
            "total_window_seconds": seconds,
            "successful_decisions_per_second": (
                sum(item["decisions"] for item in successes) / seconds
                if seconds > 0
                else None
            ),
            "scheduled_arrival_to_completion": _ranks(
                [item["latency_ms"] for item in successes]
            ),
            "client_queue": _ranks([item["client_queue_ms"] for item in successes]),
            "arrival_jitter": _ranks(
                [item["arrival_jitter_ms"] for item in arm_workflows]
            ),
            "rounds": arm_waves,
        }
    eligible = not reasons
    old_dps = arms["old"]["successful_decisions_per_second"]
    new_dps = arms["new"]["successful_decisions_per_second"]
    old_p50 = arms["old"]["scheduled_arrival_to_completion"]["p50_ms"]
    new_p50 = arms["new"]["scheduled_arrival_to_completion"]["p50_ms"]
    return {
        "arms": arms,
        "comparison": {
            "eligible": eligible,
            "reasons": reasons,
            "new_over_old_successful_decisions_per_second": (
                new_dps / old_dps if eligible and old_dps else None
            ),
            "old_over_new_p50_scheduled_arrival_latency": (
                old_p50 / new_p50 if eligible and new_p50 else None
            ),
        },
    }


def _record_wave(
    samples_handle: Any,
    workflows_handle: Any,
    samples: list[HttpSample],
    workflows: list[dict[str, Any]],
    *,
    trace_sha256: str,
    origin_ns: int,
    timed_semantic_statuses: dict[tuple[int, str | None], str] | None = None,
) -> None:
    arrivals = {item["sequence"]: item for item in workflows}
    for sample in samples:
        record = sample.public_record(origin_ns)
        record["arrival_trace_sha256"] = trace_sha256
        workflow = arrivals[sample.sequence]
        record["scheduled_offset_ms"] = (workflow["scheduled_ns"] - origin_ns) / 1e6
        record["arrived_offset_ms"] = (workflow["arrived_ns"] - origin_ns) / 1e6
        record["http_wait_from_arrival_ms"] = (
            sample.started_ns - workflow["arrived_ns"]
        ) / 1e6
        if timed_semantic_statuses is not None:
            record["timed_semantic_status"] = timed_semantic_statuses[
                sample.sequence, sample.state_id
            ]
        samples_handle.write(json.dumps(record, sort_keys=True) + "\n")
    for workflow in workflows:
        record = {
            key: value for key, value in workflow.items() if not key.endswith("_ns")
        }
        record.update(
            {
                "arrival_trace_sha256": trace_sha256,
                "scheduled_offset_ms": (workflow["scheduled_ns"] - origin_ns) / 1e6,
                "arrived_offset_ms": (workflow["arrived_ns"] - origin_ns) / 1e6,
                "first_send_offset_ms": (workflow["first_send_ns"] - origin_ns) / 1e6,
                "last_body_offset_ms": (workflow["last_body_ns"] - origin_ns) / 1e6,
                "completed_offset_ms": (workflow["completed_ns"] - origin_ns) / 1e6,
            }
        )
        workflows_handle.write(json.dumps(record, sort_keys=True) + "\n")
    samples_handle.flush()
    workflows_handle.flush()


def _apply_timed_semantics(
    workflows: list[dict[str, Any]],
    wave: dict[str, Any],
    statuses: dict[tuple[int, str | None], str],
    evidence_summary: dict[str, Any],
) -> None:
    """Count a workflow only when every timed HTTP body matches its audit."""

    for workflow in workflows:
        call_statuses = [
            status
            for (sequence, _state_id), status in statuses.items()
            if sequence == workflow["sequence"]
        ]
        if len(call_statuses) != workflow["http_calls"]:
            call_statuses.append("timed_evidence_identity_mismatch")
        failures = sorted(
            {
                status
                for status in call_statuses
                if status != "passed" and status != "http_failure"
            }
        )
        if failures:
            workflow["success"] = False
            workflow["error_codes"] = sorted(set(workflow["error_codes"] + failures))
    wave["successful_workflows"] = sum(item["success"] for item in workflows)
    wave["successful_decisions"] = sum(
        item["decisions"] for item in workflows if item["success"]
    )
    wave["timed_semantics"] = evidence_summary


def run_semantic_arrivals(args: argparse.Namespace) -> int:
    old_url = validate_endpoint_url(args.old_url)
    new_url = validate_endpoint_url(args.new_url)
    old_meta = _metadata(args, "old")
    new_meta = _metadata(args, "new")
    old_model_id = args.old_model_id or args.model
    if not MODEL_ID.fullmatch(old_model_id):
        raise ValueError("old model ID is not a public-safe model slug")
    old = Endpoint("old", old_url, _token(args.old_token_env), args.old_response_mode)
    new_single = Endpoint("new", new_url, _token(args.new_token_env))
    new_batch = Endpoint("new", batch_url(new_url), new_single.token)
    scope_reasons = tuple(
        f"different_{key}"
        for key in ("model_revision", "hardware", "network_scope")
        if old_meta[key] != new_meta[key]
    )
    cohorts = {
        (q, s): generate_cases(
            args.model,
            old_model_id,
            question_count=q,
            state_count=s,
            variants=args.variants,
            seed=args.seed,
        )
        for q in args.question_counts
        for s in args.state_counts
    }
    spacing_ns = round(args.arrival_spacing_ms * 1e6)
    if args.arrival_spacing_ms > 0 and spacing_ns == 0:
        raise ValueError("positive arrival spacing must be at least one nanosecond")
    trace_rows = []
    trace_index: dict[tuple[int, int, int, str, int], tuple[Arrival, ...]] = {}
    for (q, s), cases in cohorts.items():
        for c in args.concurrencies:
            for phase, rounds, count in (
                ("warmup", 1, args.warmup),
                ("throughput", args.rounds, args.workflows),
            ):
                for round_ in range(rounds):
                    if count == 0:
                        continue
                    trace = make_trace(cases, count, args.seed + 1 + round_, spacing_ns)
                    trace_index[q, s, c, phase, round_] = trace
                    trace_rows.append(
                        trace_record(trace, q=q, s=s, c=c, phase=phase, round_=round_)
                    )
    trace_document = {
        "schema_version": TRACE_SCHEMA,
        "model": args.model,
        "arrival_spacing_ns": spacing_ns,
        "traces": trace_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = args.output_dir / "arrival-traces.json"
    trace_path.write_text(
        json.dumps(trace_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    audit_path = args.output_dir / "audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        audit = audit_cohorts(
            cohorts,
            old,
            new_single,
            new_batch,
            timeout=args.timeout,
            probability_tolerance=0.01,
            output_handle=handle,
        )
    audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    receipt_base = {
        "schema_version": SCHEMA,
        "scope": "exploratory paired fixed-arrival Decision HTTP workflow performance; not release evidence",
        "model": args.model,
        "old": old_meta,
        "new": new_meta,
        "adapter": {
            "old_model_id": old_model_id,
            "old_response_mode": args.old_response_mode,
        },
        "source_commit": _source_commit(),
        "harness_sha256": _arrival_harness_digest(),
        "cohort_sha256": hashlib.sha256(
            b"\x00".join(cohort_sha256(cases).encode() for cases in cohorts.values())
        ).hexdigest(),
        "arrival_traces_sha256": trace_sha256,
        "audit_sha256": audit_sha256,
        "settings": {
            "question_counts": args.question_counts,
            "state_counts": args.state_counts,
            "concurrencies": args.concurrencies,
            "variants_per_shape": args.variants,
            "warmup_workflows_per_arm": args.warmup,
            "throughput_workflows_per_round_per_arm": args.workflows,
            "throughput_rounds": args.rounds,
            "arrival_spacing_ns": spacing_ns,
            "max_arrival_jitter_ms": args.max_arrival_jitter_ms,
            "http_concurrency_unit": "maximum dispatched HTTP calls per arm",
            "arrival_policy": "fixed logical workflow releases, identical sealed trace per arm; old state singles use round-robin dispatch",
            "latency_boundary": "scheduled logical arrival through request-relative HTTP response contract validation; audit parity is checked afterwards",
            "timeout_seconds": args.timeout,
            "seed": args.seed,
            "timed_semantic_policy": "both arms, every throughput response, against the sealed untimed audit; fixed absolute probability tolerance 0.01",
            "timed_semantic_archive": "bounded private-review gzip sidecar; not safe to publish without inspection",
        },
        "audit": audit,
    }
    if audit["status"] != "passed":
        receipt = {**receipt_base, "status": "audit_failed", "shapes": []}
        (args.output_dir / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 1

    audit_rows = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    audited_cases = {row["case_id"]: row for row in audit_rows}
    if len(audited_cases) != len(audit_rows):
        raise ValueError("audited arrival cases are duplicated")

    origin_ns = time.perf_counter_ns()
    shapes = []
    with (
        (args.output_dir / "samples.jsonl").open(
            "w", encoding="utf-8"
        ) as samples_handle,
        (args.output_dir / "workflows.jsonl").open(
            "w", encoding="utf-8"
        ) as workflows_handle,
        gzip.open(
            args.output_dir / "timed-semantic.jsonl.gz", "wt", encoding="utf-8"
        ) as evidence_handle,
    ):
        timed_evidence = TimedSemanticEvidence(evidence_handle)
        for (q, s), _cases in cohorts.items():
            for c in args.concurrencies:
                workflows: list[dict[str, Any]] = []
                waves: list[dict[str, Any]] = []
                warmup_reasons: list[str] = []
                for phase, rounds in (("warmup", 1), ("throughput", args.rounds)):
                    for round_ in range(rounds):
                        trace = trace_index.get((q, s, c, phase, round_))
                        if trace is None:
                            continue
                        trace_hash = next(
                            row["sha256"]
                            for row in trace_rows
                            if (
                                row["question_count"],
                                row["state_count"],
                                row["http_concurrency_cap"],
                                row["phase"],
                                row["round"],
                            )
                            == (q, s, c, phase, round_)
                        )
                        order = ("old", "new") if round_ % 2 == 0 else ("new", "old")
                        for arm in order:
                            samples, wave_workflows, wave = run_arrival_wave(
                                trace,
                                arm=arm,
                                phase=phase,
                                round_=round_,
                                c=c,
                                old=old,
                                new_single=new_single,
                                new_batch=new_batch,
                                timeout=args.timeout,
                            )
                            semantic_statuses = None
                            if phase == "throughput":
                                semantic_statuses, evidence_summary = (
                                    timed_evidence.validate_wave(
                                        samples,
                                        trace,
                                        audited_cases,
                                        arm=arm,
                                        model_id=args.model,
                                        old_model_id=old_model_id,
                                        old_response_mode=args.old_response_mode,
                                    )
                                )
                                _apply_timed_semantics(
                                    wave_workflows,
                                    wave,
                                    semantic_statuses,
                                    evidence_summary,
                                )
                            _record_wave(
                                samples_handle,
                                workflows_handle,
                                samples,
                                wave_workflows,
                                trace_sha256=trace_hash,
                                origin_ns=origin_ns,
                                timed_semantic_statuses=semantic_statuses,
                            )
                            wave["wave_start_offset_ms"] = (
                                wave.pop("wave_start_ns") - origin_ns
                            ) / 1e6
                            if phase == "throughput":
                                workflows.extend(wave_workflows)
                                waves.append(wave)
                            else:
                                if any(not item["success"] for item in wave_workflows):
                                    warmup_reasons.append(f"{arm}_warmup_failed")
                                if (
                                    wave["max_arrival_jitter_ms"]
                                    > args.max_arrival_jitter_ms
                                ):
                                    warmup_reasons.append(
                                        f"{arm}_warmup_arrival_jitter_exceeded"
                                    )
                shapes.append(
                    {
                        "question_count": q,
                        "state_count": s,
                        "concurrency": c,
                        "trace_sha256_by_round": [
                            row["sha256"]
                            for row in trace_rows
                            if (
                                row["question_count"],
                                row["state_count"],
                                row["http_concurrency_cap"],
                                row["phase"],
                            )
                            == (q, s, c, "throughput")
                        ],
                        "summary": summarize_arrivals(
                            workflows,
                            waves,
                            max_jitter_ms=args.max_arrival_jitter_ms,
                            extra_reasons=(*scope_reasons, *warmup_reasons),
                        ),
                    }
                )
    if hashlib.sha256(trace_path.read_bytes()).hexdigest() != trace_sha256:
        raise ValueError("sealed arrival trace changed during measurement")
    if hashlib.sha256(audit_path.read_bytes()).hexdigest() != audit_sha256:
        raise ValueError("sealed semantic audit changed during measurement")
    receipt = {
        **receipt_base,
        "status": "measured",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "shapes": shapes,
        "timed_semantic_evidence": {
            "archive": "timed-semantic.jsonl.gz",
            "sha256": hashlib.sha256(
                (args.output_dir / "timed-semantic.jsonl.gz").read_bytes()
            ).hexdigest(),
            "archived_http": timed_evidence.archived,
            "uncompressed_bytes": timed_evidence.uncompressed_bytes,
            "max_uncompressed_bytes": 80 * 1024 * 1024,
        },
    }
    receipt_path = args.output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"receipt": str(receipt_path), "status": "measured"}))
    return int(any(not row["summary"]["comparison"]["eligible"] for row in shapes))


def add_arrival_parser(commands: argparse._SubParsersAction) -> None:
    run = commands.add_parser(
        "semantic-arrivals", help="measure fixed logical-workflow arrival traces"
    )
    run.add_argument("--model", required=True, choices=MODELS)
    run.add_argument("--old-url", required=True)
    run.add_argument("--new-url", required=True)
    run.add_argument("--old-token-env")
    run.add_argument("--new-token-env")
    run.add_argument("--old-model-id")
    run.add_argument(
        "--old-response-mode",
        choices=("decision_v1", "legacy_preview"),
        default="decision_v1",
    )
    for arm in ("old", "new"):
        run.add_argument(f"--{arm}-source-ref", required=True)
        run.add_argument(f"--{arm}-model-revision", required=True)
        run.add_argument(f"--{arm}-hardware", required=True)
        run.add_argument(
            f"--{arm}-network-scope",
            required=True,
            choices=("loopback", "private", "public"),
        )
        run.add_argument(
            f"--{arm}-physical-batch-size", required=True, type=_positive_int
        )
    run.add_argument("--question-counts", type=_positive_csv, default=(1, 8, 32))
    run.add_argument("--state-counts", type=_positive_csv, default=(1, 8, 32))
    run.add_argument("--concurrencies", type=_positive_csv, default=(1, 8, 32))
    run.add_argument("--variants", type=_positive_int, default=4)
    run.add_argument("--warmup", type=_nonnegative_int, default=2)
    run.add_argument("--workflows", type=_positive_int, default=32)
    run.add_argument("--rounds", type=_positive_int, default=3)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--timeout", type=_positive_float, default=60.0)
    run.add_argument(
        "--arrival-spacing-ms",
        type=_nonnegative_float,
        required=True,
        help="Fixed interval between logical arrivals; 0 is a synchronized burst.",
    )
    run.add_argument(
        "--max-arrival-jitter-ms",
        type=_positive_float,
        default=50.0,
        help="Mark ratios ineligible if the client misses an arrival by more than this.",
    )
    run.add_argument("--output-dir", required=True, type=Path)
    run.set_defaults(handler=run_semantic_arrivals)
