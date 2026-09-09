"""The artifact a quality baseline run measures, and the proof that it is that one.

Resolution is kept apart from the harness so that the binding between a
referenced manifest and the bytes on disk can be tested without the training
extras the harness itself needs.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifact_inventory import ServedArtifact
from provenance.crossref import verify_artifact_bytes
from provenance.emit import resolve_hf_revision
from provenance.manifest import load_manifest

ARTIFACT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]

logger = logging.getLogger("QualityBaseline")


def download_artifact(repo: str, revision: str, patterns: list[str]) -> Path:
    """Fetch the artifact files that the run will score.

    The Hub client is imported here rather than at module scope so that
    resolving and checking an artifact stays testable wherever only the schema
    dependencies are installed.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - lazy: download only

    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


class BaselineError(RuntimeError):
    """Raised when the run cannot produce a trustworthy baseline."""


@dataclass(frozen=True)
class MeasuredArtifact:
    """The artifact this run will score, and how its identity was established."""

    model_dir: Path
    repo: str
    revision: str
    referenced: dict[str, Any] | None

    def is_served_by(self, served: ServedArtifact) -> bool:
        return self.repo == served.hf_repo


def referenced_artifact(path: Path | None) -> dict[str, Any] | None:
    """Load an artifact manifest a training run already published."""
    if path is None:
        return None
    return load_manifest(path, expected_kind="artifact")


def resolve_measured_artifact(
    args: argparse.Namespace, served: ServedArtifact
) -> MeasuredArtifact:
    """Decide which bytes this run scores, and warn when they are not the served ones.

    A referenced manifest lends its identity to every number this run reports,
    so it also decides which bytes are fetched, and the bytes are re-hashed
    against it before anything is scored. Otherwise a run could measure one
    artifact and publish another's digest beside the result.
    """
    referenced = referenced_artifact(args.artifact_manifest)

    if args.artifact_dir is not None:
        if referenced is None:
            raise BaselineError(
                "--artifact-dir requires --artifact-manifest so the evaluation can "
                "reference the identity the run already published"
            )
        measured = MeasuredArtifact(
            model_dir=args.artifact_dir,
            repo=referenced["identity"]["repo"],
            revision=referenced["identity"]["revision"],
            referenced=referenced,
        )
        logger.warning(
            "measuring local artifact %s, which is NOT the artifact %s serves (%s)",
            measured.model_dir,
            args.config,
            served.hf_repo,
        )
        return _verified(measured)

    repo = args.artifact_repo or served.hf_repo
    patterns = ARTIFACT_PATTERNS
    if referenced is not None:
        identity = referenced["identity"]
        if args.artifact_repo is not None and args.artifact_repo != identity["repo"]:
            raise BaselineError(
                f"--artifact-manifest describes {identity['repo']}, but "
                f"--artifact-repo asks for {args.artifact_repo}; pass the manifest "
                "that was emitted for the artifact being measured"
            )
        repo = identity["repo"]
        revision = identity["revision"]
        patterns = [entry["path"] for entry in referenced["files"]]
    else:
        revision = resolve_hf_revision(repo)
    if repo != served.hf_repo:
        logger.warning(
            "measuring %s, which is NOT the artifact %s serves (%s)",
            repo,
            args.config,
            served.hf_repo,
        )
    return _verified(
        MeasuredArtifact(
            model_dir=download_artifact(repo, revision, patterns),
            repo=repo,
            revision=revision,
            referenced=referenced,
        )
    )


def _verified(measured: MeasuredArtifact) -> MeasuredArtifact:
    """Refuse to score bytes that are not the ones the referenced manifest names."""
    if measured.referenced is None:
        return measured
    problems = verify_artifact_bytes(measured.referenced, measured.model_dir)
    if problems:
        raise BaselineError(
            f"{measured.model_dir} is not the artifact {measured.referenced['id']} "
            f"describes, so it cannot be measured under that identity: "
            + "; ".join(problems)
        )
    return measured
