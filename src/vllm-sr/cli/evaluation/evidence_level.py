"""Derive evidence layers only from executor-registered source contracts."""

from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.executor_contracts import ExecutorContract
from cli.evaluation.reporting import EvidenceLevel

_LEVEL_ORDER: tuple[EvidenceLevel, ...] = ("E0", "E1", "E2", "E3", "E4", "E5")
_NORMALIZED_REPLAY_TRACK_LEVEL: dict[str, EvidenceLevel] = {
    "routing": "E3",
    "model_pool": "E4",
    "joint": "E5",
    "agentic": "E5",
    "multimodal": "E5",
    "preference": "E4",
    "safety": "E4",
    "capacity": "E5",
}


def _parse_evidence_level(value: str) -> EvidenceLevel | None:
    return next((level for level in _LEVEL_ORDER if value == level), None)


def _normalized_replay_level(
    executor_id: str,
    track_id: str,
    records: list[ExecutionRecord],
) -> EvidenceLevel:
    if not records or any(record.status == "unavailable" for record in records):
        return "E0"
    ceilings: list[EvidenceLevel] = []
    prefix = f"{executor_id};ceiling="
    for record in records:
        kind = record.evidence_kind or ""
        if not kind.startswith(prefix):
            return "E0"
        ceiling = kind.removeprefix(prefix)
        parsed_ceiling = _parse_evidence_level(ceiling)
        if parsed_ceiling is None:
            return "E0"
        ceilings.append(parsed_ceiling)
    if not ceilings:
        return "E0"
    ceiling = min(ceilings, key=_LEVEL_ORDER.index)
    produced = _NORMALIZED_REPLAY_TRACK_LEVEL[track_id]
    return min((ceiling, produced), key=_LEVEL_ORDER.index)


def track_evidence_level(
    mode: str,
    executor: ExecutorContract,
    track_id: str,
    records: list[ExecutionRecord],
) -> EvidenceLevel:
    """Qualify one track against the exact frozen executor contract."""

    if mode != executor.mode or track_id not in executor.track_ids:
        return "E0"
    if mode == "replay":
        return _normalized_replay_level(executor.id, track_id, records)
    if not records or any(record.status == "unavailable" for record in records):
        return "E0"
    return executor.evidence_qualifications.qualify_records(track_id, records) or "E0"


def run_evidence_level(
    mode: str,
    executor: ExecutorContract,
    track_ids: tuple[str, ...],
    records: list[ExecutionRecord],
) -> EvidenceLevel:
    levels = [
        track_evidence_level(
            mode,
            executor,
            track_id,
            [record for record in records if record.track_id == track_id],
        )
        for track_id in track_ids
    ]
    observed = min(levels, key=_LEVEL_ORDER.index, default="E0")
    if executor.evidence_level_ceiling is None:
        return observed
    return min(
        (observed, executor.evidence_level_ceiling),
        key=_LEVEL_ORDER.index,
    )
