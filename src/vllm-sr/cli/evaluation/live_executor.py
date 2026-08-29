"""Truthful E0 live diagnostics with promotion-only seams kept fail-closed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cli.evaluation.contracts import (
    CaseGrading,
    CaseVisible,
    EvaluationTargetArm,
    GradingCaseSet,
    VisibleCaseSet,
)
from cli.evaluation.evidence import ExecutionRecord, RoutingDiagnostic
from cli.evaluation.http_client import EvaluationHTTPClient, HTTPResult
from cli.evaluation.live_chat import (
    chat_request,
    direct_arm_evidence,
    grade_response,
    result_accounting,
    routed_chat_records,
)
from cli.evaluation.live_outcomes import (
    offline_preference_records,
    safety_proxy_records,
)
from cli.evaluation.load_executor import run_capacity_sweep
from cli.evaluation.routing_trace import (
    normalize_routing_diagnostic,
    routing_trace_digest,
)


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
    arm_results: dict[tuple[str, str], HTTPResult]
    pool_records: list[ExecutionRecord]
    chat_results: dict[str, HTTPResult]
    joint_records: list[ExecutionRecord]


@dataclass(frozen=True)
class _PoolCollection:
    records: list[ExecutionRecord]
    results: dict[tuple[str, str], HTTPResult]


@dataclass(frozen=True)
class _ChatCollection:
    records: list[ExecutionRecord]
    joint_records: list[ExecutionRecord]
    results: dict[str, HTTPResult]


def _url(base: str, suffix: str) -> str:
    normalized = base.rstrip("/")
    if normalized.endswith(suffix):
        return normalized
    if suffix == "/api/v1/eval":
        if normalized.endswith("/api/v1"):
            return normalized + "/eval"
        if normalized.endswith("/api"):
            return normalized + "/v1/eval"
    if suffix.startswith("/v1/") and normalized.endswith("/v1"):
        return normalized + suffix.removeprefix("/v1")
    return normalized + suffix


def _route_url(base: str) -> str:
    return _url(base, "/api/v1/eval") + "?trace=true"


def _messages(case: CaseVisible) -> list[dict[str, Any]]:
    return [
        message.model_dump(mode="json", exclude_none=True) for message in case.messages
    ]


def _unavailable(track: str, case: CaseVisible, reason: str) -> ExecutionRecord:
    return ExecutionRecord(
        id=f"{track.replace('_', '-')}-{case.id}",
        track_id=track,
        case_id=case.id,
        attempt_id=f"attempt-{case.id}",
        status="unavailable",
        error=reason,
    )


def _discover_entrypoints(
    client: EvaluationHTTPClient, envoy_url: str
) -> tuple[str, ...]:
    result = client.get(_url(envoy_url, "/v1/models"))
    rows = result.payload.get("data") if result.payload else None
    if not result.success or not isinstance(rows, list):
        return ()
    candidates = [
        row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]
    ]
    virtual = [
        row
        for row in candidates
        if isinstance(row.get("routing"), dict)
        and row["routing"].get("resolution") == "virtual"
        and row["routing"].get("selectable") is not False
    ]
    selected = virtual or candidates
    default_ids = {
        row["id"]
        for row in selected
        if isinstance(row.get("routing"), dict)
        and row["routing"].get("default_route") is True
    }
    models = sorted(
        {row["id"] for row in selected},
        key=lambda model: (model not in default_ids, model),
    )
    return tuple(models[:8])


def _selected_entrypoint(discovered_entrypoints: tuple[str, ...]) -> str:
    """Choose the runtime-advertised virtual model used by every live probe.

    The OpenAI model catalog is the server-owned source of truth for the
    request-facing Entrypoint.  ``vllm-sr/auto`` is retained only as the
    canonical compatibility fallback for routing-only targets that do not
    expose an Envoy model catalog.
    """

    if discovered_entrypoints:
        return discovered_entrypoints[0]
    return "vllm-sr/auto"


def _route_records(
    client: EvaluationHTTPClient,
    endpoint: str,
    cases: tuple[CaseVisible, ...],
    entrypoint_model: str,
) -> tuple[list[ExecutionRecord], tuple[RoutingDiagnostic, ...]]:
    records: list[ExecutionRecord] = []
    traces: list[RoutingDiagnostic] = []
    for case in cases:
        result = client.post(
            endpoint,
            {
                "model": entrypoint_model,
                "messages": _messages(case),
                "evaluate_all_signals": True,
            },
        )
        payload = result.payload or {}
        diagnostic = normalize_routing_diagnostic(case.id, payload)
        if result.payload is not None:
            traces.append(diagnostic)
        records.append(
            ExecutionRecord(
                id=f"routing-{case.id}",
                track_id="routing",
                case_id=case.id,
                attempt_id=f"attempt-{case.id}",
                status="succeeded" if result.success else "failed",
                selected_arm_id=diagnostic.selected_model,
                selection_status=diagnostic.selection_status or "unknown",
                selection_method=diagnostic.selection_method,
                recipe=diagnostic.recipe,
                decision_name=diagnostic.decision_name,
                algorithm=diagnostic.algorithm,
                trace_digest=(
                    routing_trace_digest(diagnostic)
                    if result.payload is not None
                    else None
                ),
                success=result.success,
                fallback=diagnostic.selection_status == "fallback",
                latency_ms=result.latency_ms,
                evidence_kind="live-routing-diagnostic-smoke",
                error=result.error,
            )
        )
    return records, tuple(traces)


def _collect_routes(
    client: EvaluationHTTPClient,
    cases: tuple[CaseVisible, ...],
    track_ids: tuple[str, ...],
    router_api_url: str | None,
    entrypoint_model: str,
) -> tuple[list[ExecutionRecord], tuple[RoutingDiagnostic, ...]]:
    if "routing" not in track_ids:
        return [], ()
    if router_api_url:
        return _route_records(
            client,
            _route_url(router_api_url),
            cases,
            entrypoint_model,
        )
    return [
        _unavailable("routing", case, "router_api_url is not configured")
        for case in cases
    ], ()


def _collect_pool(
    client: EvaluationHTTPClient,
    cases: tuple[CaseVisible, ...],
    track_ids: tuple[str, ...],
    envoy_url: str | None,
    arms: tuple[EvaluationTargetArm, ...],
) -> _PoolCollection:
    if "model_pool" not in track_ids and "preference" not in track_ids:
        return _PoolCollection(records=[], results={})
    if envoy_url and arms:
        evidence = direct_arm_evidence(
            client, _url(envoy_url, "/v1/chat/completions"), cases, arms
        )
        return _PoolCollection(records=evidence.records, results=evidence.results)
    reason = (
        "envoy_url is not configured"
        if not envoy_url
        else "target has no server-owned physical model arms"
    )
    return _PoolCollection(
        records=[_unavailable("model_pool", case, reason) for case in cases],
        results={},
    )


def _multimodal_records(
    cases: tuple[CaseVisible, ...],
    results: dict[str, HTTPResult],
) -> list[ExecutionRecord]:
    records: list[ExecutionRecord] = []
    for case in (case for case in cases if case.modality != "text"):
        result = results.get(case.id)
        if result is None:
            records.append(
                _unavailable("multimodal", case, "envoy_url is not configured")
            )
            continue
        records.append(
            ExecutionRecord(
                id=f"multimodal-{case.id}",
                track_id="multimodal",
                case_id=case.id,
                attempt_id=f"attempt-{case.id}",
                status="succeeded" if result.success else "failed",
                success=result.success,
                modality=case.modality,
                latency_ms=result.latency_ms,
                error=result.error,
            )
        )
    return records


def _collect_chat(
    client: EvaluationHTTPClient,
    cases: tuple[CaseVisible, ...],
    track_ids: tuple[str, ...],
    envoy_url: str | None,
    arms: tuple[EvaluationTargetArm, ...],
    entrypoint_model: str,
) -> _ChatCollection:
    chat_tracks = {"joint", "multimodal", "preference", "safety"}
    if not set(track_ids) & chat_tracks:
        return _ChatCollection(records=[], joint_records=[], results={})
    if not envoy_url:
        records: list[ExecutionRecord] = []
        if "joint" in track_ids:
            records.extend(
                _unavailable("joint", case, "envoy_url is not configured")
                for case in cases
            )
        if "multimodal" in track_ids:
            records.extend(
                _unavailable("multimodal", case, "envoy_url is not configured")
                for case in cases
                if case.modality != "text"
            )
        return _ChatCollection(records=records, joint_records=[], results={})
    routed = routed_chat_records(
        client,
        _url(envoy_url, "/v1/chat/completions"),
        cases,
        arms,
        entrypoint_model,
    )
    records = list(routed.records) if "joint" in track_ids else []
    if "multimodal" in track_ids:
        records.extend(_multimodal_records(cases, routed.results))
    return _ChatCollection(
        records=records,
        joint_records=routed.records,
        results=routed.results,
    )


def _collect_agentic(
    cases: tuple[CaseVisible, ...],
    track_ids: tuple[str, ...],
) -> list[ExecutionRecord]:
    records: list[ExecutionRecord] = []
    if "agentic" in track_ids:
        records.extend(
            _unavailable(
                "agentic",
                case,
                "live agentic requires trajectory evidence not exposed by this target",
            )
            for case in cases
        )
    return records


def _collect_capacity(
    client: EvaluationHTTPClient,
    cases: tuple[CaseVisible, ...],
    track_ids: tuple[str, ...],
    envoy_url: str | None,
    concurrency: int,
    arms: tuple[EvaluationTargetArm, ...],
    entrypoint_model: str,
) -> list[ExecutionRecord]:
    if "capacity" not in track_ids:
        return []
    if envoy_url:
        return run_capacity_sweep(
            client,
            _url(envoy_url, "/v1/chat/completions"),
            cases,
            concurrency,
            entrypoint_model,
            chat_request,
            lambda result: result_accounting(result, arms),
        )
    return [
        _unavailable("capacity", case, "envoy_url is not configured") for case in cases
    ]


def execute_live_raw(
    visible: VisibleCaseSet,
    *,
    track_ids: tuple[str, ...],
    router_api_url: str | None,
    envoy_url: str | None,
    concurrency: int,
    model_arms: tuple[EvaluationTargetArm, ...] = (),
    router_api_key_env: str | None = None,
    envoy_api_key_env: str | None = None,
    client: EvaluationHTTPClient | None = None,
) -> LiveRawResult:
    unattested = sorted(set(track_ids).intersection({"model_pool", "joint"}))
    if unattested:
        raise ValueError(
            "live tracks require an attested direct-arm execution seam: "
            + ", ".join(unattested)
        )
    unsupported = sorted(
        set(track_ids) - {"routing", "multimodal", "capacity", "model_pool", "joint"}
    )
    if unsupported:
        raise ValueError(
            "generic live target cannot produce qualified track evidence: "
            + ", ".join(unsupported)
        )
    router = client or EvaluationHTTPClient(credential_env=router_api_key_env)
    envoy = client or EvaluationHTTPClient(credential_env=envoy_api_key_env)
    cases = visible.cases
    entrypoints = _discover_entrypoints(envoy, envoy_url) if envoy_url else ()
    entrypoint_model = _selected_entrypoint(entrypoints)
    route_records, traces = _collect_routes(
        router,
        cases,
        track_ids,
        router_api_url,
        entrypoint_model,
    )
    pool = _collect_pool(envoy, cases, track_ids, envoy_url, model_arms)
    chat = _collect_chat(
        envoy,
        cases,
        track_ids,
        envoy_url,
        model_arms,
        entrypoint_model,
    )
    records = route_records + chat.records
    if "model_pool" in track_ids:
        records.extend(pool.records)
    records.extend(_collect_agentic(cases, track_ids))
    records.extend(
        _collect_capacity(
            envoy,
            cases,
            track_ids,
            envoy_url,
            concurrency,
            model_arms,
            entrypoint_model,
        )
    )
    return LiveRawResult(
        records=records,
        discovered_entrypoints=entrypoints,
        routing_traces=traces,
        arm_results=pool.results,
        pool_records=pool.records,
        chat_results=chat.results,
        joint_records=chat.joint_records,
    )


def _grade_record(
    row: ExecutionRecord,
    labels: dict[str, CaseGrading],
    raw: LiveRawResult,
) -> ExecutionRecord:
    grading = labels[row.case_id]
    quality: float | None = None
    if row.track_id == "model_pool" and row.arm_id:
        result = raw.arm_results.get((row.case_id, row.arm_id))
        quality = grade_response(result, grading) if result else None
    elif row.track_id in {"joint", "multimodal"}:
        result = raw.chat_results.get(row.case_id)
        quality = grade_response(result, grading) if result else None
    return row.model_copy(update={"quality": quality})


def grade_live_execution(
    raw: LiveRawResult,
    visible: VisibleCaseSet,
    grading: GradingCaseSet,
    *,
    track_ids: tuple[str, ...],
    model_arms: tuple[EvaluationTargetArm, ...],
) -> LiveExecutionResult:
    labels = {case.case_id: case for case in grading.cases}
    records = [_grade_record(row, labels, raw) for row in raw.records]
    pool_records = [_grade_record(row, labels, raw) for row in raw.pool_records]
    joint_records = [_grade_record(row, labels, raw) for row in raw.joint_records]
    if "preference" in track_ids:
        records.extend(
            offline_preference_records(
                visible.cases,
                labels,
                tuple(arm.id for arm in model_arms),
                pool_records,
                joint_records,
            )
        )
    if "safety" in track_ids:
        records.extend(safety_proxy_records(visible.cases, labels, raw.chat_results))
    return LiveExecutionResult(
        records=records,
        discovered_entrypoints=raw.discovered_entrypoints,
        routing_traces=raw.routing_traces,
    )
