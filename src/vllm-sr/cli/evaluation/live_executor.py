"""Bounded live executor for the runtime tracks in the current catalog."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from cli.evaluation.contract_validation import derived_portable_id
from cli.evaluation.contracts import (
    CapacityLoadProtocol,
    CaseGrading,
    CaseVisible,
    GradingCaseSet,
    VisibleCaseSet,
)
from cli.evaluation.dense_pool_oracle import grade_routing_with_dense_pool_oracle
from cli.evaluation.evidence import ExecutionRecord, RoutingDiagnostic
from cli.evaluation.evidence_source_ids import (
    LIVE_JOINT_EVIDENCE_SOURCE_ID,
    LIVE_MODEL_POOL_EVIDENCE_SOURCE_ID,
    LIVE_ROUTING_EVIDENCE_SOURCE_ID,
)
from cli.evaluation.http_client import EvaluationHTTPClient, HTTPResult
from cli.evaluation.live_chat import (
    arm_accounting,
    chat_request,
    execute_chat_cases,
    grade_response,
    message_payloads,
)
from cli.evaluation.load_executor import run_capacity_sweep
from cli.evaluation.routing_trace import (
    normalize_routing_diagnostic,
    routing_trace_digest,
)
from cli.evaluation.target_arm_resolution import resolve_target_arm
from cli.evaluation.target_contracts import EvaluationTargetArm, ManifestMixture

LIVE_RUNTIME_TRACKS = frozenset(
    {"routing", "model_pool", "joint", "multimodal", "capacity"}
)
_MOM_TRACKS = ("routing", "model_pool", "joint")
_MIN_MODEL_POOL_ARMS = 2
_MIN_SHARED_MOM_TRACKS = 2


def _bounded_map(
    items: tuple[Any, ...],
    max_workers: int,
    worker: Callable[[Any], Any],
) -> list[Any]:
    """Execute one phase concurrently and return results in frozen input order."""

    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, item) for item in items]
        wait(futures)
        errors = [future.exception() for future in futures if future.exception()]
        if errors:
            for future in futures:
                future.cancel()
            raise errors[0]
        return [future.result() for future in futures]


@dataclass(frozen=True)
class LiveExecutionResult:
    records: list[ExecutionRecord]
    discovered_entrypoints: tuple[str, ...]
    routing_traces: tuple[RoutingDiagnostic, ...]


@dataclass(frozen=True)
class LiveRawResult:
    records: list[ExecutionRecord]
    discovered_entrypoints: tuple[str, ...]
    routing_traces: tuple[RoutingDiagnostic, ...]
    chat_results: dict[str, HTTPResult]
    model_pool_results: dict[tuple[str, str], HTTPResult]
    model_pool_arm_ids: tuple[str, ...]
    joint_results: dict[str, HTTPResult]


def _endpoint(base: str, suffix: str) -> str:
    return base.rstrip("/") + suffix


def _route_url(base: str) -> str:
    return _endpoint(base, "/api/v1/eval") + "?trace=true"


def _accounting_for_arm(
    result: HTTPResult,
    arm: EvaluationTargetArm,
) -> tuple[int | None, int | None, float | None]:
    accounting = arm_accounting(arm, result)
    if result.success and any(value is None for value in accounting):
        raise ValueError(
            "successful live chat response lacks complete token accounting"
        )
    return accounting


def _routed_accounting(
    result: HTTPResult,
    arms: tuple[EvaluationTargetArm, ...],
) -> tuple[EvaluationTargetArm | None, int | None, int | None, float | None]:
    if not result.success:
        return None, None, None, None
    selected_selector = result.headers.get("x-vsr-selected-model")
    arm = resolve_target_arm(selected_selector, arms)
    if selected_selector is not None and arm is None:
        raise ValueError("routed chat selected a model outside the frozen pool")
    if result.success and arm is None:
        raise ValueError("successful routed chat lacks a frozen selected arm")
    if arm is None:
        return None, None, None, None
    input_tokens, output_tokens, runtime_cost = _accounting_for_arm(result, arm)
    return arm, input_tokens, output_tokens, runtime_cost


def _discover_entrypoints(
    client: EvaluationHTTPClient,
    envoy_url: str,
    mixture: ManifestMixture,
) -> tuple[str, ...]:
    result = client.get(_endpoint(envoy_url, "/v1/models"))
    rows = result.payload.get("data") if result.payload else None
    if not result.success or not isinstance(rows, list):
        raise ValueError("runtime model catalog could not be read")
    virtual = _collect_virtual_entrypoints(rows)
    _validate_frozen_entrypoints(virtual, mixture)
    return mixture.aliases


def _collect_virtual_entrypoints(rows: list[Any]) -> dict[str, dict[str, Any]]:
    virtual: dict[str, dict[str, Any]] = {}
    for row in rows:
        entrypoint = _parse_virtual_entrypoint(row)
        if entrypoint is None:
            continue
        model_id, routing = entrypoint
        if model_id in virtual:
            raise ValueError(
                "runtime model catalog contains duplicate virtual entrypoints"
            )
        virtual[model_id] = routing
    if not virtual:
        raise ValueError(
            "runtime model catalog exposes no selectable virtual entrypoint"
        )
    return virtual


def _parse_virtual_entrypoint(
    row: Any,
) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(row, dict):
        return None
    routing = row.get("routing")
    if not isinstance(routing, dict) or routing.get("resolution") != "virtual":
        return None
    model_id = row.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("virtual model catalog entry requires a non-empty id")
    selectable = routing.get("selectable")
    default_route = routing.get("default_route", False)
    if not isinstance(selectable, bool) or not isinstance(default_route, bool):
        raise ValueError("virtual model catalog routing flags must be boolean")
    return (model_id, routing) if selectable else None


def _validate_frozen_entrypoints(
    virtual: dict[str, dict[str, Any]],
    mixture: ManifestMixture,
) -> None:
    for alias in mixture.aliases:
        routing = virtual.get(alias)
        if routing is None:
            raise ValueError(
                f"runtime model catalog does not expose frozen mixture alias {alias!r}"
            )
        if routing.get("recipe") != mixture.recipe_name:
            raise ValueError(
                f"runtime mixture alias {alias!r} does not bind recipe {mixture.recipe_name!r}"
            )
    if mixture.entrypoint_model not in virtual:
        raise ValueError("runtime model catalog does not expose the frozen entrypoint")


def discover_live_entrypoints(
    client: EvaluationHTTPClient,
    envoy_url: str,
    mixture: ManifestMixture,
) -> tuple[str, ...]:
    """Discover the frozen runtime entrypoint order for non-chat evidence paths."""

    return _discover_entrypoints(client, envoy_url, mixture)


def _route_records(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    mixture: ManifestMixture,
    concurrency: int,
) -> tuple[list[ExecutionRecord], tuple[RoutingDiagnostic, ...]]:
    def evaluate(case: CaseVisible) -> tuple[ExecutionRecord, RoutingDiagnostic | None]:
        attempt_id = derived_portable_id("attempt", case.id)
        result = client.post(
            endpoint,
            {
                "model": mixture.entrypoint_model,
                "messages": message_payloads(case),
                "evaluate_all_signals": True,
            },
            track_id="routing",
            case_id=case.id,
            attempt_id=attempt_id,
        )
        payload = result.payload or {}
        diagnostic = normalize_routing_diagnostic(case.id, payload)
        if result.success and diagnostic.recipe != mixture.recipe_name:
            raise ValueError("routing diagnostic does not bind the frozen recipe")
        selected = resolve_target_arm(diagnostic.selected_model, mixture.model_arms)
        if (
            result.success
            and diagnostic.selection_status in {"selected", "fallback"}
            and selected is None
        ):
            raise ValueError("routing selected a model outside the frozen mixture pool")
        return (
            ExecutionRecord(
                id=derived_portable_id("routing", case.id),
                track_id="routing",
                case_id=case.id,
                attempt_id=attempt_id,
                status="succeeded" if result.success else "failed",
                selected_arm_id=selected.id if selected is not None else None,
                selection_status=diagnostic.selection_status or "unknown",
                selection_method=diagnostic.selection_method,
                recipe=diagnostic.recipe,
                decision_name=diagnostic.decision_name,
                # Records attest the realized selector. The configured
                # decision algorithm remains available in routing traces.
                algorithm=diagnostic.selection_method,
                trace_digest=(
                    routing_trace_digest(diagnostic)
                    if result.payload is not None
                    else None
                ),
                success=result.success,
                fallback=diagnostic.selection_status == "fallback",
                latency_ms=result.latency_ms,
                evidence_kind=LIVE_ROUTING_EVIDENCE_SOURCE_ID,
                broker_receipt=result.broker_receipt,
                error=result.error,
            ),
            diagnostic if result.payload is not None else None,
        )

    outcomes = _bounded_map(cases, concurrency, evaluate)
    return (
        [record for record, _ in outcomes],
        tuple(trace for _, trace in outcomes if trace is not None),
    )


def _model_pool_records(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    arms: tuple[EvaluationTargetArm, ...],
    concurrency: int,
) -> tuple[list[ExecutionRecord], dict[tuple[str, str], HTTPResult]]:
    if len(arms) < _MIN_MODEL_POOL_ARMS:
        raise ValueError("live model_pool evaluation requires at least two arms")
    tasks = tuple((case, arm) for case in cases for arm in arms)

    def evaluate(
        task: tuple[CaseVisible, EvaluationTargetArm],
    ) -> tuple[ExecutionRecord, HTTPResult]:
        case, arm = task
        attempt_id = derived_portable_id("attempt-model-pool", case.id, arm.id)
        result = chat_request(
            client,
            endpoint,
            case,
            arm.model,
            "model_pool",
            attempt_id,
        )
        input_tokens, output_tokens, runtime_cost = _accounting_for_arm(result, arm)
        return (
            ExecutionRecord(
                id=derived_portable_id("model-pool", case.id, arm.id),
                track_id="model_pool",
                case_id=case.id,
                attempt_id=attempt_id,
                status="succeeded" if result.success else "failed",
                arm_id=arm.id,
                success=result.success,
                latency_ms=result.latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                runtime_cost=runtime_cost,
                evidence_kind=LIVE_MODEL_POOL_EVIDENCE_SOURCE_ID,
                broker_receipt=result.broker_receipt,
                error=result.error,
            ),
            result,
        )

    outcomes = _bounded_map(tasks, concurrency, evaluate)
    return (
        [record for record, _ in outcomes],
        {
            (case.id, arm.id): result
            for (case, arm), (_, result) in zip(tasks, outcomes, strict=True)
        },
    )


def _joint_records(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    mixture: ManifestMixture,
    concurrency: int,
) -> tuple[list[ExecutionRecord], dict[str, HTTPResult]]:
    def evaluate(case: CaseVisible) -> tuple[ExecutionRecord, HTTPResult]:
        attempt_id = derived_portable_id("attempt-joint", case.id)
        result = chat_request(
            client,
            endpoint,
            case,
            mixture.entrypoint_model,
            "joint",
            attempt_id,
        )
        arm, input_tokens, output_tokens, runtime_cost = _routed_accounting(
            result, mixture.model_arms
        )
        selection_method = (
            result.headers.get("x-vsr-selected-algorithm") if arm is not None else None
        )
        if arm is not None and not selection_method:
            raise ValueError("successful routed chat lacks its selection algorithm")
        selected_recipe = result.headers.get("x-vsr-selected-recipe")
        if selected_recipe is not None and selected_recipe != mixture.recipe_name:
            raise ValueError("routed chat does not bind the frozen recipe")
        decision_name = (
            result.headers.get("x-vsr-selected-decision") if arm is not None else None
        )
        return (
            ExecutionRecord(
                id=derived_portable_id("joint", case.id),
                track_id="joint",
                case_id=case.id,
                attempt_id=attempt_id,
                status="succeeded" if result.success else "failed",
                selected_arm_id=arm.id if arm is not None else None,
                selection_status="selected" if arm is not None else None,
                selection_method=selection_method,
                algorithm=selection_method,
                recipe=(
                    (selected_recipe or mixture.recipe_name)
                    if arm is not None
                    else None
                ),
                decision_name=decision_name,
                success=result.success,
                latency_ms=result.latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                runtime_cost=runtime_cost,
                evidence_kind=LIVE_JOINT_EVIDENCE_SOURCE_ID,
                broker_receipt=result.broker_receipt,
                error=result.error,
            ),
            result,
        )

    outcomes = _bounded_map(cases, concurrency, evaluate)
    return (
        [record for record, _ in outcomes],
        {case.id: result for case, (_, result) in zip(cases, outcomes, strict=True)},
    )


def _multimodal_records(
    cases: tuple[CaseVisible, ...],
    results: dict[str, HTTPResult],
    arms: tuple[EvaluationTargetArm, ...],
) -> list[ExecutionRecord]:
    records: list[ExecutionRecord] = []
    for case in cases:
        result = results[case.id]
        arm, input_tokens, output_tokens, runtime_cost = _routed_accounting(
            result, arms
        )
        records.append(
            ExecutionRecord(
                id=derived_portable_id("multimodal", case.id),
                track_id="multimodal",
                case_id=case.id,
                attempt_id=derived_portable_id("attempt", case.id),
                status="succeeded" if result.success else "failed",
                selected_arm_id=arm.id if arm is not None else None,
                success=result.success,
                modality=case.modality,
                latency_ms=result.latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                runtime_cost=runtime_cost,
                broker_receipt=result.broker_receipt,
                error=result.error,
            )
        )
    return records


def _planned_cases(
    cases: tuple[CaseVisible, ...], track_id: str
) -> tuple[CaseVisible, ...]:
    return tuple(case for case in cases if track_id in case.track_ids)


def _validate_mom_cohort(
    cases: tuple[CaseVisible, ...], track_ids: tuple[str, ...]
) -> None:
    selected = tuple(track_id for track_id in _MOM_TRACKS if track_id in track_ids)
    if len(selected) < _MIN_SHARED_MOM_TRACKS:
        return
    cohorts = {
        tuple(case.id for case in _planned_cases(cases, track_id))
        for track_id in selected
    }
    if len(cohorts) != 1:
        raise ValueError(
            "routing, model_pool, and joint must use the same visible case cohort"
        )


def _execute_multimodal_track(
    client: EvaluationHTTPClient,
    envoy_url: str,
    cases: tuple[CaseVisible, ...],
    entrypoint_model: str,
    model_arms: tuple[EvaluationTargetArm, ...],
) -> tuple[dict[str, HTTPResult], list[ExecutionRecord]]:
    multimodal_cases = tuple(
        case
        for case in cases
        if case.modality != "text" and "multimodal" in case.track_ids
    )
    if not multimodal_cases:
        raise ValueError("multimodal evaluation requires non-text cases")
    results = execute_chat_cases(
        client,
        _endpoint(envoy_url, "/v1/chat/completions"),
        multimodal_cases,
        entrypoint_model,
    )
    return results, _multimodal_records(multimodal_cases, results, model_arms)


def _execute_capacity_track(
    client: EvaluationHTTPClient,
    envoy_url: str,
    cases: tuple[CaseVisible, ...],
    protocol: CapacityLoadProtocol | None,
    concurrency: int,
    entrypoint_model: str,
    model_arms: tuple[EvaluationTargetArm, ...],
) -> list[ExecutionRecord]:
    if protocol is None:
        raise ValueError("live capacity execution requires capacity_load_protocol")
    if protocol.concurrency_levels[-1] != concurrency:
        raise ValueError("capacity load protocol does not match run concurrency")
    capacity_cases = tuple(case for case in cases if "capacity" in case.track_ids)
    return run_capacity_sweep(
        client,
        _endpoint(envoy_url, "/v1/chat/completions"),
        capacity_cases,
        protocol,
        entrypoint_model,
        chat_request,
        lambda result: _routed_accounting(result, model_arms)[1:],
    )


def execute_live_raw(
    visible: VisibleCaseSet,
    *,
    track_ids: tuple[str, ...],
    router_api_url: str | None,
    envoy_url: str,
    concurrency: int,
    capacity_load_protocol: CapacityLoadProtocol | None,
    mixture: ManifestMixture,
    client: EvaluationHTTPClient | None = None,
) -> LiveRawResult:
    if not envoy_url:
        raise ValueError("live runtime executor requires envoy_url")
    unsupported = sorted(set(track_ids) - LIVE_RUNTIME_TRACKS)
    if unsupported:
        raise ValueError(
            "live runtime executor does not own tracks: " + ", ".join(unsupported)
        )
    router = client or EvaluationHTTPClient()
    envoy = client or EvaluationHTTPClient()
    cases = visible.cases
    _validate_mom_cohort(cases, track_ids)
    entrypoints = _discover_entrypoints(envoy, envoy_url, mixture)
    entrypoint_model = mixture.entrypoint_model
    model_arms = mixture.model_arms
    records: list[ExecutionRecord] = []
    traces: tuple[RoutingDiagnostic, ...] = ()
    if "routing" in track_ids:
        if router_api_url is None:
            raise ValueError("routing evaluation requires router_api_url")
        routing_cases = _planned_cases(cases, "routing")
        route_records, traces = _route_records(
            router,
            _route_url(router_api_url),
            routing_cases,
            mixture,
            concurrency,
        )
        records.extend(route_records)
    model_pool_results: dict[tuple[str, str], HTTPResult] = {}
    if "model_pool" in track_ids:
        pool_records, model_pool_results = _model_pool_records(
            envoy,
            _endpoint(envoy_url, "/v1/chat/completions"),
            _planned_cases(cases, "model_pool"),
            model_arms,
            concurrency,
        )
        records.extend(pool_records)
    joint_results: dict[str, HTTPResult] = {}
    if "joint" in track_ids:
        joint_records, joint_results = _joint_records(
            envoy,
            _endpoint(envoy_url, "/v1/chat/completions"),
            _planned_cases(cases, "joint"),
            mixture,
            concurrency,
        )
        records.extend(joint_records)
    chat_results: dict[str, HTTPResult] = {}
    if "multimodal" in track_ids:
        chat_results, multimodal_records = _execute_multimodal_track(
            envoy,
            envoy_url,
            cases,
            entrypoint_model,
            model_arms,
        )
        records.extend(multimodal_records)
    if "capacity" in track_ids:
        records.extend(
            _execute_capacity_track(
                envoy,
                envoy_url,
                cases,
                capacity_load_protocol,
                concurrency,
                entrypoint_model,
                model_arms,
            )
        )
    return LiveRawResult(
        records=records,
        discovered_entrypoints=entrypoints,
        routing_traces=traces,
        chat_results=chat_results,
        model_pool_results=model_pool_results,
        model_pool_arm_ids=(
            tuple(arm.id for arm in model_arms) if "model_pool" in track_ids else ()
        ),
        joint_results=joint_results,
    )


def _grade_record(
    row: ExecutionRecord,
    labels: dict[str, CaseGrading],
    chat_results: dict[str, HTTPResult],
    model_pool_results: dict[tuple[str, str], HTTPResult],
    joint_results: dict[str, HTTPResult],
) -> ExecutionRecord:
    label = labels.get(row.case_id)
    if label is None:
        raise ValueError(f"live grading lacks hidden labels for case {row.case_id!r}")
    quality: float | None
    grader: str | None
    if row.track_id == "routing":
        quality = (
            float(row.selected_arm_id == label.expected_route)
            if row.success
            and row.selected_arm_id is not None
            and label.expected_route is not None
            else None
        )
        grader = "live-hidden-route-label.v1" if quality is not None else None
    elif row.track_id == "model_pool":
        if row.arm_id is None:
            raise ValueError("live model_pool record lacks its frozen arm id")
        result = model_pool_results[(row.case_id, row.arm_id)]
        quality = grade_response(result, label) if result.success else None
        grader = "live-hidden-answer-exact.v1" if quality is not None else None
    elif row.track_id == "joint":
        result = joint_results[row.case_id]
        quality = grade_response(result, label) if result.success else None
        grader = "live-hidden-answer-exact.v1" if quality is not None else None
    elif row.track_id == "multimodal":
        result = chat_results[row.case_id]
        quality = grade_response(result, label) if result.success else None
        grader = "live-hidden-answer-exact.v1" if quality is not None else None
    else:
        return row
    return row.model_copy(update={"quality": quality, "grader": grader})


def grade_live_execution(
    raw: LiveRawResult,
    grading: GradingCaseSet,
) -> LiveExecutionResult:
    labels = {case.case_id: case for case in grading.cases}
    graded = [
        _grade_record(
            row,
            labels,
            raw.chat_results,
            raw.model_pool_results,
            raw.joint_results,
        )
        for row in raw.records
    ]
    oracle_graded = (
        grade_routing_with_dense_pool_oracle(graded, raw.model_pool_arm_ids)
        if raw.model_pool_arm_ids
        else graded
    )
    return LiveExecutionResult(
        records=oracle_graded,
        discovered_entrypoints=raw.discovered_entrypoints,
        routing_traces=raw.routing_traces,
    )
