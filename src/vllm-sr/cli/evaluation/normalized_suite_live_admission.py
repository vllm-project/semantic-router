"""Platform-owned live admissions for installed normalized-suite data."""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass
from typing import Literal, cast

from cli.evaluation.catalog_tracks import CatalogMethodEvidenceSource
from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contract_primitives import ImagePart
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.errors import SuiteStoreError
from cli.evaluation.normalized_suite_live_robustness import (
    DECLARED_SHIFT_LIVE_METHOD_ID,
    declared_shift_source_is_eligible,
)
from cli.evaluation.reporting import TrackID
from cli.evaluation.suite_contract import (
    BenchmarkSuiteManifest,
    NormalizedMultimodalObservation,
)
from cli.evaluation.suite_install_contract import NormalizedMediaEntry
from cli.evaluation.suite_store import NormalizedSuiteStore

NORMALIZED_MULTIMODAL_LIVE_METHOD_ID = "multimodal.hidden-answer.server-live.v1"
BENCHMARK_PACK_LIVE_METHOD_PREFIX = "benchmark-pack.server-live"
_BENCHMARK_PACK_LIVE_TRACKS = frozenset(
    {"routing", "model_pool", "joint", "multimodal", "capacity"}
)


@dataclass(frozen=True)
class NormalizedSuiteLiveAdmission:
    """One exact server-owned method that an immutable source may drive."""

    method_id: str
    track_id: TrackID
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
    seen_case_ids: set[str] = set()
    for row in store.load_jsonl(manifest.id, "visible_cases"):
        case = cast(CaseVisible, row)
        if case.id in seen_case_ids:
            raise _invalid_multimodal("visible cohort contains a duplicate case")
        seen_case_ids.add(case.id)
        if "multimodal" not in case.track_ids:
            continue
        if case.modality != "image":
            raise _invalid_multimodal("a planned multimodal case is not an image")
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
    if not images:
        raise _invalid_multimodal("visible image cohort is empty")
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
    *,
    require_observations: bool,
) -> None:
    grading: set[str] = set()
    for row in store.load_jsonl(manifest.id, "grading_cases"):
        label = cast(CaseGrading, row)
        if label.case_id not in case_ids and not require_observations:
            continue
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

    if grading != set(case_ids):
        raise _invalid_multimodal("hidden labels do not cover the exact case cohort")
    if require_observations:
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
        if observations != set(case_ids):
            raise _invalid_multimodal(
                "multimodal observations do not cover the exact case cohort"
            )


def multimodal_hidden_answer_source_is_eligible(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> bool:
    """Admit an exact hidden-answer image cohort from MMR or a data-only pack."""

    if "multimodal" not in manifest.track_ids:
        return False
    parser_verified = manifest.qualification_receipt.qualification.parser_verified
    registered_mmr = manifest.adapter_id == "mmr-bench" and parser_verified
    benchmark_pack = manifest.source_receipt.source_kind == "benchmark_pack"
    if not registered_mmr and not benchmark_pack:
        return False
    if manifest.artifacts.media_manifest is None:
        return False
    if registered_mmr and manifest.artifacts.multimodal_observations is None:
        return False

    images = _visible_image_cohort(store, manifest)
    if registered_mmr and len(images) != manifest.case_count:
        raise _invalid_multimodal("registered MMR image cohort is incomplete")
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
    _validate_hidden_labels(
        store,
        manifest,
        frozenset(images),
        require_observations=registered_mmr,
    )
    return True


def _benchmark_pack_live_tracks(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> frozenset[TrackID]:
    if manifest.source_receipt.source_kind != "benchmark_pack":
        return frozenset()
    case_ids_by_track: dict[str, set[str]] = {
        track_id: set() for track_id in manifest.track_ids
    }
    visible_case_ids: set[str] = set()
    for row in store.load_jsonl(manifest.id, "visible_cases"):
        case = cast(CaseVisible, row)
        if case.id in visible_case_ids:
            raise SuiteStoreError(
                "installed benchmark pack contains duplicate visible case ids"
            )
        visible_case_ids.add(case.id)
        for track_id in case.track_ids:
            case_ids_by_track[track_id].add(case.id)
    labels: dict[str, CaseGrading] = {}
    for row in store.load_jsonl(manifest.id, "grading_cases"):
        label = cast(CaseGrading, row)
        if label.case_id not in visible_case_ids or label.case_id in labels:
            raise SuiteStoreError(
                "installed benchmark pack grading cases do not match its visible cases"
            )
        labels[label.case_id] = label
    if set(labels) != visible_case_ids:
        raise SuiteStoreError(
            "installed benchmark pack grading cases do not cover its visible cases"
        )
    admitted: set[TrackID] = set()
    for track_id in manifest.track_ids:
        case_ids = case_ids_by_track[track_id]
        if _benchmark_pack_track_is_live_eligible(
            track_id,
            case_ids,
            labels,
            store,
            manifest,
        ):
            admitted.add(track_id)
    return frozenset(admitted)


def _complete_hidden_labels(
    case_ids: set[str],
    labels: dict[str, CaseGrading],
    field: Literal["expected_route", "expected_answer"],
) -> bool:
    return bool(case_ids) and all(
        (value := getattr(labels[case_id], field)) is not None and bool(value.strip())
        for case_id in case_ids
    )


def _benchmark_pack_track_is_live_eligible(
    track_id: TrackID,
    case_ids: set[str],
    labels: dict[str, CaseGrading],
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> bool:
    if track_id not in _BENCHMARK_PACK_LIVE_TRACKS or not case_ids:
        return False
    if track_id == "capacity":
        return True
    if track_id == "routing":
        return _complete_hidden_labels(case_ids, labels, "expected_route")
    if track_id in {"model_pool", "joint"}:
        return _complete_hidden_labels(case_ids, labels, "expected_answer")
    return (
        track_id == "multimodal"
        and _complete_hidden_labels(case_ids, labels, "expected_answer")
        and multimodal_hidden_answer_source_is_eligible(store, manifest)
    )


def benchmark_pack_live_method_id(track_id: str) -> str:
    return f"{BENCHMARK_PACK_LIVE_METHOD_PREFIX}.{track_id}.v1"


def normalized_suite_live_admissions(
    store: NormalizedSuiteStore,
    manifest: BenchmarkSuiteManifest,
) -> tuple[NormalizedSuiteLiveAdmission, ...]:
    """Return the platform live methods supported by one installed source."""

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
    if (
        manifest.source_receipt.source_kind == "registered_adapter"
        and multimodal_hidden_answer_source_is_eligible(store, manifest)
    ):
        admissions.append(
            NormalizedSuiteLiveAdmission(
                method_id=NORMALIZED_MULTIMODAL_LIVE_METHOD_ID,
                track_id="multimodal",
                qualified_gate_ids=(),
                evidence_source=CatalogMethodEvidenceSource.LIVE_RUNTIME,
            )
        )
    pack_tracks = _benchmark_pack_live_tracks(store, manifest)
    for track_id in TRACK_IDS:
        if track_id not in pack_tracks:
            continue
        admissions.append(
            NormalizedSuiteLiveAdmission(
                method_id=benchmark_pack_live_method_id(track_id),
                track_id=track_id,
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
