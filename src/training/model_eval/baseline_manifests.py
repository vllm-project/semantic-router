"""Provenance manifest emission for the Router Model quality baseline.

An evaluation manifest references the artifact identity a training run already
published rather than minting a second identity for the same bytes, so a
candidate measurement can never be mistaken for the served baseline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from artifact_inventory import ServedArtifact
from baseline_tasks import TaskSpec, tokenizer_class
from provenance.emit import (
    build_artifact_manifest,
    build_dataset_manifest,
    build_evaluation_manifest,
    split_digest,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = "src/training/model_eval/quality_baseline.py"


def emit_baseline_manifests(
    *,
    args: argparse.Namespace,
    artifact: ServedArtifact,
    revision: str,
    model_dir: Path,
    measured_repo: str,
    referenced: dict[str, Any] | None,
    model_config: dict[str, Any],
    spec: TaskSpec,
    dataset_revision: str,
    texts: list[str],
    labels: np.ndarray,
    split_rows: int,
    mapping: dict[str, int],
    summary: dict[str, Any],
) -> dict[str, str]:
    manifest_dir = args.manifest_dir or (args.output_dir / "manifests")
    rows = list(zip(texts, labels.tolist(), strict=True))

    dataset_id = _slug(spec.dataset_repo.rsplit("/", maxsplit=1)[-1])
    artifact_id = _slug(measured_repo.rsplit("/", maxsplit=1)[-1])
    evaluation_id = f"{artifact_id}-{spec.split}-{revision[:12]}"

    dataset = _build_dataset(
        dataset_id=dataset_id,
        task=args.task,
        spec=spec,
        dataset_revision=dataset_revision,
        rows=rows,
        split_rows=split_rows,
        scored_rows=summary["metrics"]["rows"],
        declared_limit=args.limit,
        labels=labels,
        mapping=mapping,
    )

    if referenced is not None:
        artifact_manifest = referenced
        artifact_id = referenced["id"]
        evaluation_id = f"{artifact_id}-{spec.split}-{revision[:12]}"
        written_artifact = None
    else:
        artifact_manifest = _build_measured_artifact(
            artifact_id=artifact_id,
            task=args.task,
            measured_repo=measured_repo,
            served_repo=artifact.hf_repo,
            served_path=artifact.model_path,
            revision=revision,
            model_dir=model_dir,
            mapping=mapping,
            model_config=model_config,
        )
        written_artifact = manifest_dir / f"{artifact_id}.manifest.yaml"

    evaluation = _build_evaluation(
        evaluation_id=evaluation_id,
        args=args,
        artifact_ref={
            "id": artifact_id,
            "revision": revision,
            "digest": artifact_manifest["identity"]["digest"],
        },
        dataset_ref={
            "id": dataset_id,
            "revision": dataset_revision,
            "splits": [spec.split],
        },
        mapping=mapping,
        summary=summary,
    )

    written = {
        "dataset": write_manifest(
            dataset, manifest_dir / f"{dataset_id}.manifest.yaml"
        ),
        "evaluation": write_manifest(
            evaluation, manifest_dir / f"{_slug(evaluation_id)}.manifest.yaml"
        ),
    }
    if written_artifact is not None:
        written["artifact"] = write_manifest(artifact_manifest, written_artifact)
    return {kind: str(path) for kind, path in written.items()}


def _build_measured_artifact(
    *,
    artifact_id: str,
    task: str,
    measured_repo: str,
    served_repo: str,
    served_path: str,
    revision: str,
    model_dir: Path,
    mapping: dict[str, int],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    is_served = measured_repo == served_repo
    return build_artifact_manifest(
        manifest_id=artifact_id,
        task=task,
        repo=measured_repo,
        revision=revision,
        artifact_dir=model_dir,
        label_mapping=mapping,
        architecture=(model_config.get("architectures") or ["unknown"])[0],
        max_position_embeddings=int(model_config["max_position_embeddings"]),
        torch_dtype=model_config.get("torch_dtype") or model_config.get("dtype"),
        tokenizer_class=tokenizer_class(model_dir),
        served_paths=[served_path] if is_served else [],
        description=(
            "Artifact resolved from the maintained router configuration."
            if is_served
            else "Candidate artifact measured for comparison; not served."
        ),
    )


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value.lower()
    ).strip("-")


def _build_dataset(
    *,
    dataset_id: str,
    task: str,
    spec: TaskSpec,
    dataset_revision: str,
    rows: list[tuple[str, int]],
    split_rows: int,
    scored_rows: int,
    declared_limit: int | None,
    labels: np.ndarray,
    mapping: dict[str, int],
) -> dict[str, Any]:
    """Describe the split as it was consumed, not as it is published.

    When rows are dropped because the artifact does not define their label, the
    recorded count is the count actually scored. A manifest that claimed the
    published size would not reproduce.
    """
    index_to_label = {index: name for name, index in mapping.items()}
    recorded_rows = split_rows
    if declared_limit is None and scored_rows != split_rows:
        recorded_rows = scored_rows
    return build_dataset_manifest(
        manifest_id=dataset_id,
        task=task,
        source_type="huggingface",
        locator=spec.dataset_repo,
        revision=dataset_revision,
        license_id="unknown-upstream",
        splits=[
            {
                "name": spec.split,
                "rows": recorded_rows,
                "digest": split_digest(rows),
                "label_counts": {
                    index_to_label[index]: int((labels == index).sum())
                    for index in sorted(index_to_label)
                },
            }
        ],
        text_field=spec.text_field,
        label_field=spec.label_field,
        preprocessing_steps=[
            "load the published split without shuffling",
            "map label values onto the artifact's own class order",
            "drop rows whose label the artifact does not define",
        ],
        label_mapping=mapping,
    )


def _build_evaluation(
    *,
    evaluation_id: str,
    args: argparse.Namespace,
    artifact_ref: dict[str, Any],
    dataset_ref: dict[str, Any],
    mapping: dict[str, int],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return build_evaluation_manifest(
        manifest_id=_slug(evaluation_id),
        task=args.task,
        artifact_ref=artifact_ref,
        dataset_ref=dataset_ref,
        entrypoint=ENTRYPOINT,
        repo_root=REPO_ROOT,
        device=args.device,
        device_name=(
            torch.cuda.get_device_name(0)
            if args.device == "cuda"
            else os.uname().machine
        ),
        batch_size=args.batch_size,
        max_length=args.max_length,
        sample_limit=args.limit,
        seed=args.seed,
        label_mapping=mapping,
        metrics=summary["metrics"],
        calibration=summary["calibration"],
        abstention=summary["abstention"],
        performance=summary["performance"],
        slices=summary["slices"],
    )
