"""Exact first-party admissions for installed normalized-suite live execution."""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass
from typing import Literal, cast

from cli.evaluation.catalog_tracks import CatalogMethodEvidenceSource
from cli.evaluation.contract_primitives import ImagePart
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.normalized_suite_live_robustness import (
    DECLARED_SHIFT_LIVE_METHOD_ID,
    declared_shift_source_is_eligible,
)
from cli.evaluation.suite_contract import (
    BenchmarkSuiteManifest,
    NormalizedMultimodalObservation,
)
from cli.evaluation.suite_install_contract import NormalizedMediaEntry
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.suite_store_error import SuiteStoreError

NORMALIZED_MULTIMODAL_LIVE_METHOD_ID = "multimodal.hidden-answer.server-live.v1"


@dataclass(frozen=True)
class NormalizedSuiteLiveAdmission:
    """One exact server-owned method that an immutable source may drive."""

    method_id: str
    track_id: Literal["routing", "multimodal"]
    qualified_gate_ids: tuple[str, ...]
    evidence_source: CatalogMethodEvidenceSource


@dataclass(frozen=True)
class _ImageSubject:
    digest: str
    media_type: str
    size_bytes: int


def _invalid_multimodal(reason: str) -> SuiteStoreError:
    return SuiteStoreError(
        f"installed multimodal live qualification is invalid: {reason}"
    )


def _inline_image_subject(value: str) -> _ImageSubject:
    metadata, separator, encoded = value.partition(",")
    if (
        not separator
        or not metadata.startswith("data:image/")
        or not metadata.endswith(";base64")
        or not encoded
    ):
        raise _invalid_multimodal("visible image must be an inline image data URL")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _invalid_multimodal("visible image data is invalid") from exc
    if not data:
        raise _invalid_multimodal("visible image data is empty")
    return _ImageSubject(
        digest="sha256:" + hashlib.sha256(data).hexdigest(),
        media_type=metadata.removeprefix("data:").removesuffix(";base64"),
        size_bytes=len(data),
    )


def _visible_image_cohort(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> dict[str, _ImageSubject]:
    images: dict[str, _ImageSubject] = {}
    for row in store.load_jsonl(manifest.id, "visible_cases"):
        case = cast(CaseVisible, row)
        if (
            case.id in images
            or case.modality != "image"
            or "multimodal" not in case.track_ids
        ):
            raise _invalid_multimodal(
                "visible cohort contains a duplicate or non-image case"
            )
        image_parts = [
            part
            for message in case.messages
            if not isinstance(message.content, str)
            for part in message.content
            if isinstance(part, ImagePart)
        ]
        if len(image_parts) != 1:
            raise _invalid_multimodal("each visible case must bind exactly one image")
        images[case.id] = _inline_image_subject(image_parts[0].image_url.url)
    if len(images) != manifest.case_count:
        raise _invalid_multimodal("visible image cohort is incomplete")
    return images


def _media_inventory(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> dict[str, _ImageSubject]:
    media: dict[str, _ImageSubject] = {}
    seen_ids: set[str] = set()
    for row in store.load_jsonl(manifest.id, "media_manifest"):
        entry = cast(NormalizedMediaEntry, row)
        if (
            entry.id in seen_ids
            or entry.size_bytes <= 0
            or entry.modality != "image"
            or not entry.media_type.startswith("image/")
        ):
            raise _invalid_multimodal("media manifest contains an invalid image")
        seen_ids.add(entry.id)
        subject = _ImageSubject(
            digest=entry.digest,
            media_type=entry.media_type,
            size_bytes=entry.size_bytes,
        )
        prior = media.get(entry.digest)
        if prior is not None and prior != subject:
            raise _invalid_multimodal("media digest metadata conflicts")
        media[entry.digest] = subject
    if not media:
        raise _invalid_multimodal("media manifest is empty")
    return media


def _validate_hidden_labels(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
    case_ids: frozenset[str],
) -> None:
    grading: set[str] = set()
    for row in store.load_jsonl(manifest.id, "grading_cases"):
        label = cast(CaseGrading, row)
        if (
            label.case_id not in case_ids
            or label.case_id in grading
            or label.expected_answer is None
            or not label.expected_answer.strip()
        ):
            raise _invalid_multimodal(
                "grading cohort lacks an exact unique hidden answer"
            )
        grading.add(label.case_id)

    observations: set[str] = set()
    for row in store.load_jsonl(manifest.id, "multimodal_observations"):
        observation = cast(NormalizedMultimodalObservation, row)
        if (
            observation.case_id not in case_ids
            or observation.case_id in observations
            or observation.modality != "image"
        ):
            raise _invalid_multimodal(
                "multimodal observation cohort is not the exact image cohort"
            )
        observations.add(observation.case_id)
    if grading != set(case_ids) or observations != set(case_ids):
        raise _invalid_multimodal(
            "hidden labels and observations do not cover the exact case cohort"
        )


def multimodal_hidden_answer_source_is_eligible(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> bool:
    """Admit only the registered MMR parser's exact hidden-answer image cohort."""

    if manifest.adapter_id != "mmr-bench" or "multimodal" not in manifest.track_ids:
        return False
    if not manifest.qualification_receipt.qualification.parser_verified:
        return False
    if (
        manifest.artifacts.media_manifest is None
        or manifest.artifacts.multimodal_observations is None
    ):
        return False

    images = _visible_image_cohort(store, manifest)
    media = _media_inventory(store, manifest)
    used_media: set[str] = set()
    for image in images.values():
        if media.get(image.digest) != image:
            raise _invalid_multimodal(
                "visible image is not bound to its media manifest"
            )
        used_media.add(image.digest)
    if used_media != set(media):
        raise _invalid_multimodal("media manifest contains an unused object")
    _validate_hidden_labels(store, manifest, frozenset(images))
    return True


def normalized_suite_live_admissions(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> tuple[NormalizedSuiteLiveAdmission, ...]:
    """Return only exact first-party live methods for one installed source."""

    admissions: list[NormalizedSuiteLiveAdmission] = []
    if declared_shift_source_is_eligible(store, manifest):
        admissions.append(
            NormalizedSuiteLiveAdmission(
                method_id=DECLARED_SHIFT_LIVE_METHOD_ID,
                track_id="routing",
                qualified_gate_ids=("G4",),
                evidence_source=CatalogMethodEvidenceSource.SERVER_BROKERED_LIVE,
            )
        )
    if multimodal_hidden_answer_source_is_eligible(store, manifest):
        admissions.append(
            NormalizedSuiteLiveAdmission(
                method_id=NORMALIZED_MULTIMODAL_LIVE_METHOD_ID,
                track_id="multimodal",
                qualified_gate_ids=(),
                evidence_source=CatalogMethodEvidenceSource.LIVE_RUNTIME,
            )
        )
    return tuple(admissions)


def normalized_suite_live_tracks(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> frozenset[str]:
    return frozenset(
        admission.track_id
        for admission in normalized_suite_live_admissions(store, manifest)
    )
