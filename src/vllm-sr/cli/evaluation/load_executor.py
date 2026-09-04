"""Frozen repeated closed-loop execution for promotion-grade capacity evidence."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Literal

from cli.evaluation.contracts import CapacityLoadProtocol, CaseVisible
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_source_ids import LIVE_CAPACITY_EVIDENCE_SOURCE_ID
from cli.evaluation.http_client import EvaluationHTTPClient, HTTPResult

ChatCall = Callable[
    [EvaluationHTTPClient, str, CaseVisible, str, str, str],
    HTTPResult,
]
ResultAccounting = Callable[
    [HTTPResult],
    tuple[int | None, int | None, float | None],
]
LoadPhase = Literal["warmup", "measurement"]


def _attempt_id(
    concurrency: int,
    phase: LoadPhase,
    repetition: int,
    request_index: int,
) -> str:
    phase_id = "w" if phase == "warmup" else "m"
    return f"capacity-c{concurrency}-{phase_id}{repetition}-q{request_index}"


def _record(
    *,
    concurrency: int,
    phase: LoadPhase,
    repetition: int,
    request_index: int,
    case: CaseVisible,
    result: HTTPResult,
    elapsed: float,
    throughput: float,
    accounting: ResultAccounting,
) -> ExecutionRecord:
    input_tokens, output_tokens, runtime_cost = accounting(result)
    attempt_id = _attempt_id(concurrency, phase, repetition, request_index)
    return ExecutionRecord(
        id=attempt_id,
        track_id="capacity",
        case_id=case.id,
        attempt_id=attempt_id,
        status="succeeded" if result.success else "failed",
        success=result.success,
        latency_ms=result.latency_ms,
        concurrency=concurrency,
        throughput_rps=throughput,
        load_elapsed_seconds=elapsed,
        load_phase=phase,
        load_repetition=repetition,
        load_request_index=request_index,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        runtime_cost=runtime_cost,
        evidence_kind=LIVE_CAPACITY_EVIDENCE_SOURCE_ID,
        broker_receipt=result.broker_receipt,
        error=result.error,
    )


def _batch_records(
    *,
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    concurrency: int,
    phase: LoadPhase,
    repetition: int,
    request_count: int,
    entrypoint_model: str,
    chat: ChatCall,
    accounting: ResultAccounting,
) -> list[ExecutionRecord]:
    started = perf_counter()
    results: list[tuple[int, CaseVisible, HTTPResult]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for request_index in range(request_count):
            case = cases[request_index % len(cases)]
            attempt_id = _attempt_id(
                concurrency,
                phase,
                repetition,
                request_index,
            )
            future = executor.submit(
                chat,
                client,
                endpoint,
                case,
                entrypoint_model,
                "capacity",
                attempt_id,
            )
            futures[future] = (request_index, case)
        for future in as_completed(futures):
            request_index, case = futures[future]
            results.append((request_index, case, future.result()))
    elapsed = max(perf_counter() - started, 1e-9)
    throughput = len(results) / elapsed
    return [
        _record(
            concurrency=concurrency,
            phase=phase,
            repetition=repetition,
            request_index=request_index,
            case=case,
            result=result,
            elapsed=elapsed,
            throughput=throughput,
            accounting=accounting,
        )
        for request_index, case, result in sorted(results)
    ]


def run_capacity_sweep(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    protocol: CapacityLoadProtocol,
    entrypoint_model: str,
    chat: ChatCall,
    accounting: ResultAccounting,
) -> list[ExecutionRecord]:
    """Execute every frozen level with one warmup and repeated measurement windows."""

    if not cases:
        raise ValueError("capacity execution requires at least one visible case")
    records: list[ExecutionRecord] = []
    for concurrency in protocol.concurrency_levels:
        records.extend(
            _batch_records(
                client=client,
                endpoint=endpoint,
                cases=cases,
                concurrency=concurrency,
                phase="warmup",
                repetition=0,
                request_count=(concurrency * protocol.warmup_request_multiplier),
                entrypoint_model=entrypoint_model,
                chat=chat,
                accounting=accounting,
            )
        )
        for repetition in range(1, protocol.repetitions_per_level + 1):
            records.extend(
                _batch_records(
                    client=client,
                    endpoint=endpoint,
                    cases=cases,
                    concurrency=concurrency,
                    phase="measurement",
                    repetition=repetition,
                    request_count=(protocol.measurement_requests_per_repetition),
                    entrypoint_model=entrypoint_model,
                    chat=chat,
                    accounting=accounting,
                )
            )
    return records
