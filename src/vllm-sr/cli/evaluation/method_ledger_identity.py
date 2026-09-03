"""Shared immutable subject and freshness contracts for live method ledgers."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import Field, field_validator

from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contract_validation import validate_portable_id as _validate_id
from cli.evaluation.manifest_identity import mixture_snapshot_digest
from cli.evaluation.target_contracts import ManifestMixture

MAXIMUM_METHOD_LEDGER_FRESHNESS = timedelta(hours=24)


class MethodMixtureBinding(StrictModel):
    """Auditable component identities plus the digest of the complete Mixture."""

    id: str
    snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    recipe_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    pool_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selector_policy_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selector_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    adaptation_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    binding_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    _id = field_validator("id")(_validate_id)


def method_mixture_binding(mixture: ManifestMixture) -> MethodMixtureBinding:
    return MethodMixtureBinding(
        id=mixture.id,
        snapshot_digest=mixture_snapshot_digest(mixture),
        recipe_digest=mixture.recipe_digest,
        pool_digest=mixture.pool_digest,
        selector_policy_digest=mixture.selector_policy_digest,
        selector_digest=mixture.selector_digest,
        adaptation_digest=mixture.adaptation_digest,
        binding_digest=mixture.binding_digest,
    )


def validate_method_ledger_freshness(
    sealed_at: datetime, fetched_at: datetime | None
) -> None:
    """Fail closed against a Dashboard-broker-observed fetch timestamp."""

    if (
        fetched_at is None
        or sealed_at.tzinfo is None
        or sealed_at.utcoffset() is None
        or fetched_at.tzinfo is None
        or fetched_at.utcoffset() is None
    ):
        raise ValueError("method ledger lacks a server-broker fetch timestamp")
    if sealed_at > fetched_at:
        raise ValueError("method ledger seal is in the future relative to broker fetch")
    if fetched_at - sealed_at > MAXIMUM_METHOD_LEDGER_FRESHNESS:
        raise ValueError("method ledger exceeds the maximum 24-hour freshness window")
