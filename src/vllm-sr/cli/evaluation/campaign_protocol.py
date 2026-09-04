"""Versioned catalog descriptor for Campaign source cohorts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.execution_contract import (
    LIVE_RUNTIME_EXECUTOR_ID,
    MOM_REPLAY_EXECUTOR_ID,
)
from cli.evaluation.reporting import EvidenceLevel, TrackID

CAMPAIGN_COHORT_SCHEMA_VERSION = "evaluation-campaign-cohort.v1"


class CampaignProtocol(StrictModel):
    """Declares that a suite can supply one Campaign comparison cohort."""

    schema_version: Literal[CAMPAIGN_COHORT_SCHEMA_VERSION] = (
        CAMPAIGN_COHORT_SCHEMA_VERSION
    )
    minimum_cases: int = Field(gt=0, strict=True)


def validate_campaign_protocol(
    protocol: CampaignProtocol | None,
    *,
    modes: tuple[Literal["replay", "live"], ...],
    executors: Mapping[Literal["replay", "live"], str],
    track_ids: tuple[TrackID, ...],
    evidence_level: EvidenceLevel,
    case_count: int | None,
) -> None:
    """Validate a present descriptor against only its containing suite."""

    if protocol is None:
        return
    if (
        modes != ("replay", "live")
        or executors
        != {
            "replay": MOM_REPLAY_EXECUTOR_ID,
            "live": LIVE_RUNTIME_EXECUTOR_ID,
        }
        or track_ids != ("routing", "model_pool", "joint")
        or evidence_level != "E0"
        or case_count is None
        or case_count <= 0
        or protocol.minimum_cases > case_count
    ):
        raise ValueError(
            "campaign protocol requires a valid E0 replay/live MoM cohort contract"
        )
