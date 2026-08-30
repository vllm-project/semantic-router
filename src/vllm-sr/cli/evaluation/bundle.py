"""Helpers for non-self-referential artifact bundles."""

from __future__ import annotations

from cli.evaluation.contracts import ArtifactRef
from cli.evaluation.reporting import EvaluationArtifact

_PUBLIC_ARTIFACT_NAMES = frozenset(
    {
        "metrics.json",
        "gates.json",
        "provenance.json",
        "failure-summary.json",
        "routing-traces.jsonl",
        "capacity-profile.json",
        "checksums.sha256",
    }
)


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
        if name in _PUBLIC_ARTIFACT_NAMES
    )
