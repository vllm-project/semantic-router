"""Derive evidence layers from produced records, never connectivity alone."""

from __future__ import annotations

from typing import cast

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.reporting import EvidenceLevel

_LIVE_TRACK_LEVEL: dict[str, EvidenceLevel] = {
    "routing": "E3",
    "model_pool": "E4",
    "joint": "E5",
    "agentic": "E5",
    "multimodal": "E5",
    "preference": "E1",
    "safety": "E2",
    "capacity": "E5",
}
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
_NORMALIZED_REPLAY_PREFIX = "normalized-suite-replay.v1;ceiling="
_QUALIFIED_LIVE_PREFIX = "qualified-live.v1;level="
_LEVEL_ORDER: tuple[EvidenceLevel, ...] = ("E0", "E1", "E2", "E3", "E4", "E5")


def _normalized_replay_level(
    track_id: str, records: list[ExecutionRecord]
) -> EvidenceLevel:
    qualified = [record for record in records if record.status != "unavailable"]
    ceilings: list[EvidenceLevel] = []
    for record in qualified:
        kind = record.evidence_kind or ""
        if not kind.startswith(_NORMALIZED_REPLAY_PREFIX):
            return "E0"
        ceiling = kind.removeprefix(_NORMALIZED_REPLAY_PREFIX)
        if ceiling not in _LEVEL_ORDER:
            return "E0"
        ceilings.append(cast(EvidenceLevel, ceiling))
    if not ceilings:
        return "E0"
    ceiling = min(ceilings, key=_LEVEL_ORDER.index)
    produced = _NORMALIZED_REPLAY_TRACK_LEVEL[track_id]
    return min((ceiling, produced), key=_LEVEL_ORDER.index)


def track_evidence_level(
    mode: str,
    track_id: str,
    records: list[ExecutionRecord],
) -> EvidenceLevel:
    if mode == "replay":
        return _normalized_replay_level(track_id, records)
    qualified = [record for record in records if record.status != "unavailable"]
    if not qualified:
        return "E0"
    levels: list[EvidenceLevel] = []
    for record in qualified:
        kind = record.evidence_kind or ""
        if not kind.startswith(_QUALIFIED_LIVE_PREFIX):
            return "E0"
        level = kind.removeprefix(_QUALIFIED_LIVE_PREFIX)
        if level not in _LEVEL_ORDER:
            return "E0"
        levels.append(cast(EvidenceLevel, level))
    observed = min(levels, key=_LEVEL_ORDER.index)
    return min((observed, _LIVE_TRACK_LEVEL[track_id]), key=_LEVEL_ORDER.index)


def run_evidence_level(
    mode: str,
    track_ids: tuple[str, ...],
    records: list[ExecutionRecord],
) -> EvidenceLevel:
    levels = [
        track_evidence_level(
            mode,
            track_id,
            [record for record in records if record.track_id == track_id],
        )
        for track_id in track_ids
    ]
    return max(levels, key=_LEVEL_ORDER.index, default="E0")
