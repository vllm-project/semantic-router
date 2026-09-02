"""Fail-closed provenance for imported normalized benchmark suites.

A verified source checkout and a deterministic parser prove what bytes were
imported. They do not prove that the upstream benchmark or native dataset
generator produced those bytes. Until a server-owned native execution receipt
exists, every normalized import is exploratory E0 evidence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from cli.evaluation.canonical import digest_value
from cli.evaluation.contract_primitives import StrictModel

SUITE_QUALIFICATION_CONTRACT_VERSION = "evaluation-suite-qualification.v2"
NORMALIZED_REPLAY_IMPLEMENTATION_VERSION = "normalized-suite-replay.v1"
NORMALIZED_REPLAY_EXECUTOR_DIGEST = digest_value(
    {
        "contract": SUITE_QUALIFICATION_CONTRACT_VERSION,
        "implementation": NORMALIZED_REPLAY_IMPLEMENTATION_VERSION,
    }
)

NormalizationOrigin = Literal[
    "registered_parser_import",
    "user_provided_import",
]


class UnqualifiedBenchmarkEvidence(StrictModel):
    """System-issued statement of the exact limits of one normalized import."""

    schema_version: Literal[SUITE_QUALIFICATION_CONTRACT_VERSION] = (
        SUITE_QUALIFICATION_CONTRACT_VERSION
    )
    status: Literal["exploratory_import"] = "exploratory_import"
    origin: NormalizationOrigin
    parser_verified: bool
    native_execution_attested: Literal[False] = False
    promotion_eligible: Literal[False] = False

    @model_validator(mode="after")
    def parser_claim_matches_origin(self) -> UnqualifiedBenchmarkEvidence:
        expected = self.origin == "registered_parser_import"
        if self.parser_verified != expected:
            raise ValueError("parser verification must match the import origin")
        return self


class BenchmarkQualificationReceipt(StrictModel):
    """Immutable E0 receipt binding source metadata and normalized artifacts."""

    schema_version: Literal[SUITE_QUALIFICATION_CONTRACT_VERSION] = (
        SUITE_QUALIFICATION_CONTRACT_VERSION
    )
    evidence_level: Literal["E0"] = "E0"
    manifest_subject_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_set_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    executor_id: Literal["normalized-suite-replay.v1"] = "normalized-suite-replay.v1"
    executor_digest: Literal[NORMALIZED_REPLAY_EXECUTOR_DIGEST] = (
        NORMALIZED_REPLAY_EXECUTOR_DIGEST
    )
    qualification: UnqualifiedBenchmarkEvidence

    @property
    def qualified_gate_ids(self) -> tuple[str, ...]:
        return ()
