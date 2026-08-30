"""Bounded multi-level concurrency sweep for live capacity evidence."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from cli.evaluation.contracts import CaseVisible
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.http_client import EvaluationHTTPClient, HTTPResult

ChatCall = Callable[
    [EvaluationHTTPClient, str, CaseVisible, str],
    HTTPResult,
]
ResultAccounting = Callable[
    [HTTPResult],
    tuple[int | None, int | None, float | None],
]


def _record(
    concurrency: int,
    index: int,
    case: CaseVisible,
    result: HTTPResult,
    elapsed: float,
    throughput: float,
    accounting: ResultAccounting,
) -> ExecutionRecord:
    input_tokens, output_tokens, runtime_cost = accounting(result)
    return ExecutionRecord(
        id=f"capacity-c{concurrency}-{case.id}-{index}",
        track_id="capacity",
        case_id=case.id,
        attempt_id=f"attempt-c{concurrency}-{case.id}-{index}",
        status="succeeded" if result.success else "failed",
        success=result.success,
        latency_ms=result.latency_ms,
        concurrency=concurrency,
        throughput_rps=throughput,
        load_elapsed_seconds=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        runtime_cost=runtime_cost,
        error=result.error,
    )


def _level_records(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    concurrency: int,
    entrypoint_model: str,
    chat: ChatCall,
    accounting: ResultAccounting,
) -> list[ExecutionRecord]:
    request_count = max(len(cases), concurrency)
    started = time.perf_counter()
    results: list[tuple[int, CaseVisible, HTTPResult]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                chat,
                client,
                endpoint,
                case,
                entrypoint_model,
            ): (index, case)
            for index in range(request_count)
            for case in (cases[index % len(cases)],)
        }
        for future in as_completed(futures):
            index, case = futures[future]
            results.append((index, case, future.result()))
    elapsed = max(time.perf_counter() - started, 1e-9)
    throughput = len(results) / elapsed
    return [
        _record(
            concurrency,
            index,
            case,
            result,
            elapsed,
            throughput,
            accounting,
        )
        for index, case, result in sorted(results)
    ]


def run_capacity_sweep(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    requested_concurrency: int,
    entrypoint_model: str,
    chat: ChatCall,
    accounting: ResultAccounting,
) -> list[ExecutionRecord]:
    records: list[ExecutionRecord] = []
    for concurrency in sorted({1, 2, requested_concurrency}):
        records.extend(
            _level_records(
                client,
                endpoint,
                cases,
                concurrency,
                entrypoint_model,
                chat,
                accounting,
            )
        )
    return records
