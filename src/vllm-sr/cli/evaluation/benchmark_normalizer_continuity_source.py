"""Schema validation and cohort alignment for continuity-bench artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    boolean,
    exact_object,
    integer,
    iter_csv,
    iter_jsonl,
    load_json,
    number,
    require_array,
    required_file,
    string,
)
from cli.evaluation.benchmark_normalizers_common import messages
from cli.evaluation.contract_primitives import Message

_CONTINUITY_HEADER = (
    "conversation_id",
    "system",
    "failed_over",
    "latency_ms",
    "queue_wait_ms",
    "preserved",
    "expected_fact",
    "reasoning",
    "provider",
    "turn_count",
    "concurrency",
)

CONTINUITY_FAULT_KINDS = MappingProxyType(
    {
        "timeout": "timeout",
        "rate_limit": "rate_limit",
        "api_error": "server_error",
    }
)


@dataclass(frozen=True)
class ContinuityContext:
    row: dict[str, str]
    source: dict[str, Any]
    plan: dict[str, Any]
    log: dict[str, Any]
    counterpart_row: dict[str, str]
    counterpart_log: dict[str, Any]
    conversation_id: str
    system: str
    concurrency: int
    identity: tuple[str, str, int]
    turns: tuple[Message, ...]
    failure_turn: int
    preserved: bool
    recovered: bool
    mode: str


def _continuity_conversations(path: Path, max_bytes: int) -> dict[str, dict[str, Any]]:
    conversations: dict[str, dict[str, Any]] = {}
    for value in require_array(
        load_json(path, max_bytes=max_bytes),
        "continuity-bench conversations",
    ):
        row = exact_object(
            value,
            required={
                "id",
                "turns",
                "fact_turn_index",
                "probe_turn_index",
                "expected_fact",
                "fact_type",
            },
            label="continuity-bench conversation",
        )
        conversation_id = string(row["id"], "conversation id")
        if conversation_id in conversations:
            raise NormalizationError("continuity-bench repeats conversation id")
        turns = messages(row["turns"], "continuity-bench turns")
        fact_index = integer(row["fact_turn_index"], "fact_turn_index")
        probe_index = integer(row["probe_turn_index"], "probe_turn_index")
        if fact_index >= len(turns) or probe_index >= len(turns):
            raise NormalizationError("continuity-bench turn index is out of range")
        conversations[conversation_id] = row
    return conversations


def _continuity_manifest(path: Path, max_bytes: int) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(
        require_array(load_json(path, max_bytes=max_bytes), "continuity manifest")
    ):
        row = exact_object(
            value,
            required={
                "conversation_id",
                "failure_turn",
                "mode",
                "fallback_provider",
            },
            label=f"continuity manifest[{index}]",
        )
        conversation_id = string(row["conversation_id"], "manifest conversation_id")
        if conversation_id in result:
            raise NormalizationError("continuity manifest repeats conversation_id")
        integer(row["failure_turn"], "manifest failure_turn")
        mode = string(row["mode"], "manifest failure mode")
        if mode not in CONTINUITY_FAULT_KINDS:
            raise NormalizationError(
                "continuity manifest has an unsupported fault mode"
            )
        if row["fallback_provider"] is not None:
            string(row["fallback_provider"], "manifest fallback_provider")
        result[conversation_id] = row
    if not result:
        raise NormalizationError("continuity manifest is empty")
    return result


def _continuity_logs(
    path: Path, *, expected_system: str
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    request_ids: set[str] = set()
    for index, value in enumerate(iter_jsonl(path)):
        row = exact_object(
            value,
            required={
                "timestamp",
                "request_id",
                "conversation_id",
                "turn_index",
                "provider",
                "model",
                "failed_over",
                "failover_from",
                "failure_mode",
                "latency_ms",
                "time_to_first_token_ms",
                "response_text",
                "error",
                "system",
            },
            label=f"continuity {expected_system} log[{index}]",
        )
        timestamp = string(row["timestamp"], "continuity log timestamp")
        try:
            observed_at = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise NormalizationError("continuity log timestamp is invalid") from exc
        if observed_at.tzinfo is None:
            raise NormalizationError("continuity log timestamp must be timezone-aware")
        request_id = string(row["request_id"], "continuity request_id")
        if request_id in request_ids:
            raise NormalizationError("continuity log repeats request_id")
        request_ids.add(request_id)
        conversation_id = string(row["conversation_id"], "continuity conversation_id")
        system = string(row["system"], "continuity log system")
        if system != expected_system:
            raise NormalizationError("continuity log is stored under the wrong system")
        _validate_log_fields(row)
        key = (conversation_id, system)
        if key in result:
            raise NormalizationError("continuity log repeats conversation/system")
        result[key] = row
    if not result:
        raise NormalizationError(f"continuity {expected_system} log is empty")
    return result


def _validate_log_fields(row: dict[str, Any]) -> None:
    integer(row["turn_index"], "continuity log turn_index")
    string(row["provider"], "continuity log provider")
    string(row["model"], "continuity log model")
    boolean(row["failed_over"], "continuity log failed_over")
    if row["failover_from"] is not None:
        string(row["failover_from"], "continuity log failover_from")
    if row["failure_mode"] is not None:
        string(row["failure_mode"], "continuity log failure_mode")
    number(row["latency_ms"], "continuity log latency_ms")
    if row["time_to_first_token_ms"] is not None:
        number(row["time_to_first_token_ms"], "continuity log time_to_first_token_ms")
    if row["response_text"] is not None:
        string(row["response_text"], "continuity response_text", allow_empty=True)
    if row["error"] is not None:
        string(row["error"], "continuity log error")


def continuity_source_bundle(root: Path, artifacts: dict[str, Any]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    requirement = artifacts["conversations"]
    conversations = _continuity_conversations(
        required_file(root, requirement), requirement.max_bytes
    )
    manifest_requirement = artifacts["experiment-manifest"]
    manifest = _continuity_manifest(
        required_file(root, manifest_requirement), manifest_requirement.max_bytes
    )
    if set(manifest) != set(conversations):
        raise NormalizationError(
            "continuity manifest and conversation identities do not align"
        )
    logs: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact_id, system in (
        ("baseline-log", "baseline"),
        ("treatment-log", "treatment"),
    ):
        rows = _continuity_logs(
            required_file(root, artifacts[artifact_id]), expected_system=system
        )
        if set(logs).intersection(rows):
            raise NormalizationError("continuity logs repeat conversation/system")
        logs.update(rows)
    return conversations, manifest, logs


def continuity_metric_matrix(root: Path, artifact: Any) -> tuple[
    tuple[dict[str, str], ...],
    dict[tuple[str, str, int], dict[str, str]],
    int,
]:
    rows = tuple(iter_csv(required_file(root, artifact), _CONTINUITY_HEADER))
    by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (
            string(row["conversation_id"], "conversation_id"),
            string(row["system"], "continuity system"),
            integer(row["concurrency"], "continuity concurrency", minimum=1),
        )
        if key in by_key:
            raise NormalizationError("continuity-bench metrics repeat a cohort row")
        by_key[key] = row
    native_pair_count = sum(key[1] == "treatment" for key in by_key)
    return rows, by_key, native_pair_count


def build_continuity_context(
    row: dict[str, str],
    conversations: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    logs: dict[tuple[str, str], dict[str, Any]],
    metric_by_key: dict[tuple[str, str, int], dict[str, str]],
    seen: set[tuple[str, str, int]],
) -> ContinuityContext:
    conversation_id = string(row["conversation_id"], "conversation_id")
    system = string(row["system"], "continuity system")
    if system not in {"baseline", "treatment"}:
        raise NormalizationError("continuity system must be baseline or treatment")
    concurrency = integer(row["concurrency"], "continuity concurrency", minimum=1)
    identity = (conversation_id, system, concurrency)
    if identity in seen or conversation_id not in conversations:
        raise NormalizationError(
            "continuity-bench metrics repeat or reference unknown case"
        )
    seen.add(identity)
    source, plan = conversations[conversation_id], manifest[conversation_id]
    if row["expected_fact"] != source["expected_fact"]:
        raise NormalizationError("continuity-bench expected fact drifted")
    preserved = boolean(row["preserved"], "continuity preserved")
    if not boolean(row["failed_over"], "continuity failed_over"):
        raise NormalizationError(
            "continuity diagnostic requires a source provider-failover label"
        )
    _validate_metric_fields(row)
    turns = messages(source["turns"], "continuity-bench turns")
    if integer(row["turn_count"], "continuity turn_count", minimum=1) != len(turns):
        raise NormalizationError("continuity-bench turn count does not match case")
    failure_turn = integer(plan["failure_turn"], "continuity failure turn")
    if failure_turn >= len(turns):
        raise NormalizationError("continuity failure turn is outside the conversation")
    log = logs.get((conversation_id, system))
    if log is None:
        raise NormalizationError(
            "continuity metrics have no matching source execution log"
        )
    mode = _validated_log_alignment(log, plan, failure_turn)
    counterpart_system = "treatment" if system == "baseline" else "baseline"
    counterpart_row = metric_by_key.get(
        (conversation_id, counterpart_system, concurrency)
    )
    counterpart_log = logs.get((conversation_id, counterpart_system))
    if counterpart_row is None or counterpart_log is None:
        raise NormalizationError(
            "continuity baseline/treatment cohort pairing is incomplete"
        )
    return ContinuityContext(
        row=row,
        source=source,
        plan=plan,
        log=log,
        counterpart_row=counterpart_row,
        counterpart_log=counterpart_log,
        conversation_id=conversation_id,
        system=system,
        concurrency=concurrency,
        identity=identity,
        turns=turns,
        failure_turn=failure_turn,
        preserved=preserved,
        recovered=log["error"] is None and bool(log["response_text"]),
        mode=mode,
    )


def _validate_metric_fields(row: dict[str, str]) -> None:
    number(row["latency_ms"], "continuity latency_ms")
    number(row["queue_wait_ms"], "continuity queue_wait_ms")
    string(row["reasoning"], "continuity reasoning", allow_empty=True)
    string(row["provider"], "continuity provider")


def _validated_log_alignment(
    log: dict[str, Any], plan: dict[str, Any], failure_turn: int
) -> str:
    mode = string(plan["mode"], "continuity failure mode")
    if (
        integer(log["turn_index"], "continuity log turn_index") != failure_turn
        or string(log["failure_mode"], "continuity log failure_mode") != mode
        or not boolean(log["failed_over"], "continuity log failed_over")
        or not log["failover_from"]
    ):
        raise NormalizationError(
            "continuity manifest, log, and failover label do not align"
        )
    fallback_provider = plan["fallback_provider"]
    if fallback_provider is not None and log["provider"] != fallback_provider:
        raise NormalizationError(
            "continuity observed provider differs from the frozen fallback"
        )
    return mode


def validate_continuity_log_coverage(
    logs: dict[tuple[str, str], dict[str, Any]],
    seen: set[tuple[str, str, int]],
) -> None:
    expected_log_keys = {
        (
            string(row["conversation_id"], "conversation_id"),
            string(row["system"], "system"),
        )
        for row in logs.values()
    }
    if {(case[0], case[1]) for case in seen} != expected_log_keys:
        raise NormalizationError(
            "continuity logs and raw-metric conversation/system rows do not align"
        )
