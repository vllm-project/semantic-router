"""Strict multimodal normalization for MMR-Bench merged artifacts."""

from __future__ import annotations

import base64
from pathlib import Path

from cli.evaluation.benchmark_normalization_io import (
    NormalizationError,
    boolean,
    checked_media_file,
    iter_csv,
    number,
    required_file,
    string,
)
from cli.evaluation.benchmark_normalization_types import (
    BenchmarkNormalizerDescriptor,
    NormalizedAdapterPayload,
)
from cli.evaluation.benchmark_normalizers_common import (
    applicable_track_ids,
    arm_map,
    load_model_manifest,
    native_digest,
    opaque_id,
)
from cli.evaluation.canonical import sha256_digest
from cli.evaluation.contract_primitives import (
    ImagePart,
    ImageURL,
    Message,
    TextPart,
)
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.suite_contract import (
    NormalizedMultimodalObservation,
    NormalizedOutcome,
)
from cli.evaluation.suite_install_contract import NormalizedMediaEntry

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _media_entry(
    root: Path, relative_media: str
) -> tuple[bytes, str, NormalizedMediaEntry]:
    media_path, metadata = checked_media_file(root, relative_media)
    media_type = _MEDIA_TYPES.get(media_path.suffix.lower())
    if media_type is None:
        raise NormalizationError("MMR-Bench media extension is not supported")
    media_bytes = media_path.read_bytes()
    return (
        media_bytes,
        media_type,
        NormalizedMediaEntry(
            id=opaque_id("media", relative_media),
            digest=sha256_digest(media_bytes),
            media_type=media_type,
            size_bytes=metadata.st_size,
            modality="image",
            license_id="benchmark-source",
        ),
    )


def _mmr_case(
    descriptor: BenchmarkNormalizerDescriptor,
    dataset_idx: str,
    question: str,
    answer: str,
    media_bytes: bytes,
    media_type: str,
) -> tuple[CaseVisible, CaseGrading]:
    case_id = opaque_id("case", "mmr-bench", dataset_idx, question)
    visible = CaseVisible(
        id=case_id,
        track_ids=applicable_track_ids(descriptor.track_ids, modality="image"),
        messages=(
            Message(
                role="user",
                content=(
                    TextPart(text=question),
                    ImagePart(
                        image_url=ImageURL(
                            url=(
                                f"data:{media_type};base64,"
                                + base64.b64encode(media_bytes).decode("ascii")
                            )
                        )
                    ),
                ),
            ),
        ),
        modality="image",
        tags=("mmr-bench",),
    )
    return visible, CaseGrading(case_id=case_id, expected_answer=answer or None)


def _mmr_outcomes(
    row: dict[str, str],
    models: tuple[str, ...],
    arms: dict[str, str],
    descriptor: BenchmarkNormalizerDescriptor,
    case_id: str,
    dataset_idx: str,
) -> tuple[list[NormalizedOutcome], list[float]]:
    outcomes: list[NormalizedOutcome] = []
    qualities: list[float] = []
    for model in models:
        correct = boolean(row[f"{model}_correct"], f"MMR-Bench {model} correct")
        quality = 1.0 if correct else 0.0
        qualities.append(quality)
        number(row[f"{model}_cost"], f"MMR-Bench {model} normalized cost")
        outcomes.append(
            NormalizedOutcome(
                case_id=case_id,
                arm_id=arms[model],
                success=correct,
                quality=quality,
                grader_id="mmr-bench.correct",
                grader_revision=descriptor.export_schema_id,
                split="merged",
                source_record_digest=native_digest(
                    {
                        "dataset_idx": dataset_idx,
                        "model": model,
                        "correct": row[f"{model}_correct"],
                        "cost": row[f"{model}_cost"],
                    }
                ),
            )
        )
    return outcomes, qualities


def normalize_mmr_bench(
    root: Path, descriptor: BenchmarkNormalizerDescriptor
) -> NormalizedAdapterPayload:
    artifacts = {item.id: item for item in descriptor.required_artifacts}
    model_requirement = artifacts["models"]
    models = load_model_manifest(
        required_file(root, model_requirement), max_bytes=model_requirement.max_bytes
    )
    arms = arm_map(models)
    header = ["question", "answer", "dataset_idx", "img_path"]
    for model in models:
        header.extend((f"{model}_correct", f"{model}_cost"))
    visible = []
    grading = []
    outcomes = []
    observations = []
    media_by_path: dict[str, NormalizedMediaEntry] = {}
    seen: set[str] = set()
    for row in iter_csv(required_file(root, artifacts["merged-data"]), header):
        dataset_idx = string(row["dataset_idx"], "MMR-Bench dataset_idx")
        question = string(row["question"], "MMR-Bench question")
        source_key = f"{dataset_idx}\x00{question}"
        if source_key in seen:
            raise NormalizationError("MMR-Bench repeats dataset_idx and question")
        seen.add(source_key)
        relative_media = string(row["img_path"], "MMR-Bench img_path")
        media_bytes, media_type, entry = _media_entry(root, relative_media)
        prior = media_by_path.get(relative_media)
        if prior is not None and prior != entry:
            raise NormalizationError("MMR-Bench media changed while it was normalized")
        media_by_path[relative_media] = entry
        answer = string(row["answer"], "MMR-Bench answer", allow_empty=True)
        case_visible, case_grading = _mmr_case(
            descriptor,
            dataset_idx,
            question,
            answer,
            media_bytes,
            media_type,
        )
        visible.append(case_visible)
        grading.append(case_grading)
        case_outcomes, qualities = _mmr_outcomes(
            row, models, arms, descriptor, case_visible.id, dataset_idx
        )
        outcomes.extend(case_outcomes)
        observations.append(
            NormalizedMultimodalObservation(
                case_id=case_visible.id,
                modality="image",
                supported=True,
                quality=max(qualities),
                privacy_violations=0,
                source_record_digest=native_digest(
                    {"dataset_idx": dataset_idx, "img_path": relative_media}
                ),
            )
        )
    if not visible:
        raise NormalizationError("MMR-Bench merged CSV is empty")
    return NormalizedAdapterPayload(
        visible_cases=tuple(visible),
        grading_cases=tuple(grading),
        outcomes=tuple(outcomes),
        multimodal_observations=tuple(observations),
        media_manifest=tuple(media_by_path[path] for path in sorted(media_by_path)),
        arm_ids=tuple(arms.values()),
        split_protocol="Merged native image matrix with complete declared-model outcomes.",
    )
