"""Canonical ordering, visibility, and receipts for report bundles."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any, Protocol

from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.reporting import EvaluationArtifact

REPORT_BUNDLE_REQUIRED_NAMES = (
    "run-manifest.json",
    "cases.jsonl",
    "records.jsonl",
    "grading-cases.jsonl",
    "metrics.json",
    "gates.json",
    "lineage.json",
    "provenance.json",
    "failure-summary.json",
    "checksums.sha256",
    "private-checksums.sha256",
    "report.json",
)
REPORT_BUNDLE_OPTIONAL_NAMES = (
    "routing-traces.jsonl",
    "capacity-profile.json",
)
PUBLIC_REPORT_OPTIONAL_NAMES = ("capacity-profile.json",)
RUN_CONTROL_ARTIFACT_NAMES = ("status.json", "events.jsonl")
RUN_ARTIFACT_NAMES = frozenset(
    REPORT_BUNDLE_REQUIRED_NAMES
    + REPORT_BUNDLE_OPTIONAL_NAMES
    + RUN_CONTROL_ARTIFACT_NAMES
)
REPORT_TRANSACTION_ARTIFACT_NAMES = frozenset(
    name
    for name in REPORT_BUNDLE_REQUIRED_NAMES + REPORT_BUNDLE_OPTIONAL_NAMES
    if name != "run-manifest.json"
)
REPORT_TRANSACTION_REQUIRED_NAMES = frozenset(
    name for name in REPORT_BUNDLE_REQUIRED_NAMES if name != "run-manifest.json"
)
PUBLIC_RECEIPT_ARTIFACT_NAMES = (
    "metrics.json",
    "gates.json",
    "provenance.json",
    "failure-summary.json",
)
PUBLIC_REPORT_ARTIFACT_NAMES = frozenset(
    PUBLIC_RECEIPT_ARTIFACT_NAMES + PUBLIC_REPORT_OPTIONAL_NAMES + ("checksums.sha256",)
)
PRIVATE_RECEIPT_PREFIX_NAMES = (
    "run-manifest.json",
    "cases.jsonl",
    "records.jsonl",
    "grading-cases.jsonl",
    "metrics.json",
    "gates.json",
    "lineage.json",
    "provenance.json",
    "failure-summary.json",
)


def public_receipt_names(present_names: Collection[str]) -> tuple[str, ...]:
    return PUBLIC_RECEIPT_ARTIFACT_NAMES + tuple(
        name for name in PUBLIC_REPORT_OPTIONAL_NAMES if name in present_names
    )


def private_receipt_names(present_names: Collection[str]) -> tuple[str, ...]:
    return (
        PRIVATE_RECEIPT_PREFIX_NAMES
        + tuple(name for name in REPORT_BUNDLE_OPTIONAL_NAMES if name in present_names)
        + ("checksums.sha256",)
    )


class ReportBundleWriter(Protocol):
    def write_bytes(self, name: str, data: bytes) -> ArtifactRef: ...

    def write_json(self, name: str, value: Any) -> ArtifactRef: ...

    def write_jsonl(self, name: str, values: Iterable[Any]) -> ArtifactRef: ...


def artifact_media_type(name: str) -> str:
    if name.endswith(".jsonl"):
        return "application/x-ndjson"
    if name.endswith(".sha256"):
        return "text/plain"
    return "application/json"


def failure_summary(records: list[ExecutionRecord]) -> dict[str, object]:
    """Aggregate failures without exposing case or grading identity."""

    tracks: dict[str, dict[str, int]] = {}
    for record in records:
        counts = tracks.setdefault(
            record.track_id,
            {"succeeded": 0, "failed": 0, "unavailable": 0},
        )
        counts[record.status] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "total_records": len(records),
        "failed": sum(record.status == "failed" for record in records),
        "unavailable": sum(record.status == "unavailable" for record in records),
        "by_track": [
            {"track_id": track_id, **tracks[track_id]} for track_id in sorted(tracks)
        ],
    }


def public_artifact(name: str, ref: ArtifactRef) -> EvaluationArtifact:
    return EvaluationArtifact(
        id=name.replace(".", "-"),
        name=name,
        kind=name.rsplit(".", 1)[-1],
        uri=name,
        digest=ref.digest,
        media_type=ref.media_type,
        size_bytes=ref.size_bytes,
    )


def checksum_bytes(artifacts: list[tuple[str, ArtifactRef]]) -> bytes:
    """Return a deterministic receipt for the supplied visibility domain."""

    rows = [f"{ref.digest.removeprefix('sha256:')}  {name}" for name, ref in artifacts]
    return ("\n".join(rows) + "\n").encode("utf-8")


def public_artifacts(
    artifacts: list[tuple[str, ArtifactRef]],
) -> tuple[EvaluationArtifact, ...]:
    """Expose only prompt-free, connectivity-free report artifacts."""

    return tuple(
        public_artifact(name, ref)
        for name, ref in artifacts
        if name in PUBLIC_REPORT_ARTIFACT_NAMES
    )
