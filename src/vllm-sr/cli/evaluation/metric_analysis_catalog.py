"""Versioned, fail-closed analysis contracts for publishable metrics.

The packaged JSON resource is canonical. Go and TypeScript ship generated,
byte-identical mirrors so every runtime can validate reports without a Python
dependency. ``tools/ci/sync_evaluation_catalogs.py`` owns those mirrors.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from cli.evaluation.canonical import strict_json_loads
from cli.evaluation.metric_analysis_catalog_validation import (
    CATALOG_SCHEMA_VERSION as _CATALOG_SCHEMA_VERSION,
)
from cli.evaluation.metric_analysis_catalog_validation import (
    PROVENANCE_CONTRACT_VERSION as _PROVENANCE_CONTRACT_VERSION,
)
from cli.evaluation.metric_analysis_catalog_validation import (
    TRACK_IDS as _TRACK_IDS,
)
from cli.evaluation.metric_analysis_catalog_validation import (
    validate_document as _validate_document,
)

CATALOG_RESOURCE = "golden/metric_analysis_catalog.v1.json"

CATALOG_SCHEMA_VERSION = _CATALOG_SCHEMA_VERSION
PROVENANCE_CONTRACT_VERSION = _PROVENANCE_CONTRACT_VERSION


@dataclass(frozen=True)
class CatalogMetricAnalysisSpecification:
    analysis_ref: str
    track_id: str
    estimator_id: str
    estimator_version: str
    analysis_unit: str
    cluster_unit: str
    weighting: str
    missingness: str
    exclusion_policy: str
    planned_unit_projection: Mapping[str, Any]


@dataclass(frozen=True)
class CatalogMetricAnalysisMatch:
    metric_id: str
    family_id: str | None
    captures: Mapping[str, str]
    specification: CatalogMetricAnalysisSpecification


def metric_analysis_catalog_bytes() -> bytes:
    """Read the package resource used by source installs and built wheels."""

    try:
        return files("cli.evaluation").joinpath(CATALOG_RESOURCE).read_bytes()
    except OSError as exc:  # pragma: no cover - package smoke gate
        raise RuntimeError(f"read metric analysis catalog: {exc}") from exc


def _load_document() -> dict[str, Any]:
    try:
        value = strict_json_loads(metric_analysis_catalog_bytes())
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"decode metric analysis catalog: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("metric analysis catalog root must be an object")
    _validate_document(value)
    return value


def encode_metric_subject_id(raw_id: str) -> str:
    encoding = _ENCODING
    if re.fullmatch(encoding["raw_pattern"], raw_id) is None:
        raise ValueError("metric subject id is not a portable raw identifier")
    if (
        not raw_id.startswith(encoding["reserved_prefix"])
        and re.fullmatch(encoding["direct_pattern"], raw_id) is not None
    ):
        return raw_id
    encoded = (
        base64.urlsafe_b64encode(raw_id.encode("ascii")).decode("ascii").rstrip("=")
    )
    result = f"{encoding['reserved_prefix']}{encoded}"
    if re.fullmatch(encoding["encoded_pattern"], result) is None:
        raise ValueError("metric subject id exceeds the encoded segment contract")
    return result


def decode_metric_subject_id(encoded_id: str) -> str:
    encoding = _ENCODING
    if not encoded_id.startswith(encoding["reserved_prefix"]):
        if re.fullmatch(encoding["direct_pattern"], encoded_id) is None:
            raise ValueError("metric subject segment is not canonical")
        return encoded_id
    if re.fullmatch(encoding["encoded_pattern"], encoded_id) is None:
        raise ValueError("metric subject segment is not canonical base64url")
    payload = encoded_id[len(encoding["reserved_prefix"]) :]
    try:
        padding = "=" * (-len(payload) % 4)
        raw = base64.b64decode(payload + padding, altchars=b"-_", validate=True).decode(
            "ascii"
        )
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("metric subject segment is not canonical base64url") from exc
    if encode_metric_subject_id(raw) != encoded_id:
        raise ValueError("metric subject segment has a non-canonical encoding")
    return raw


def _capture_values(family: Mapping[str, Any], match: re.Match[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for capture in family["captures"]:
        raw = match.group(capture["group"])
        capture_type = capture["type"]
        if capture_type == "encoded_portable_id":
            decode_metric_subject_id(raw)
        elif capture_type == "positive_int":
            number = int(raw)
            if (
                raw != str(number)
                or not capture["minimum"] <= number <= capture["maximum"]
            ):
                raise ValueError("metric identifier integer capture is out of range")
        elif capture_type == "enum":
            if raw not in capture["values"]:
                raise ValueError("metric identifier enum capture is invalid")
        else:  # pragma: no cover - rejected during catalog load
            raise ValueError("metric identifier capture type is invalid")
        result[capture["name"]] = raw
    return result


def _resolve_from_parts(metric_id: str) -> CatalogMetricAnalysisMatch:
    static = _STATIC_BY_ID.get(metric_id)
    if static is not None:
        specification = _SPECIFICATIONS[static["analysis_ref"]]
        return CatalogMetricAnalysisMatch(
            metric_id, None, MappingProxyType({}), specification
        )
    matches: list[tuple[Mapping[str, Any], dict[str, str]]] = []
    for family, pattern in _COMPILED_FAMILIES:
        match = pattern.fullmatch(metric_id)
        if match is not None:
            matches.append((family, _capture_values(family, match)))
    if len(matches) != 1:
        kind = "unknown" if not matches else "ambiguous"
        raise ValueError(f"{kind} evaluation metric id: {metric_id}")
    family, captures = matches[0]
    selector = captures[family["selector_capture"]]
    variants = {item["value"]: item["analysis_ref"] for item in family["variants"]}
    analysis_ref = variants.get(selector, variants.get("*"))
    if analysis_ref is None:  # pragma: no cover - rejected during catalog load
        raise ValueError(f"evaluation metric id has no analysis variant: {metric_id}")
    return CatalogMetricAnalysisMatch(
        metric_id,
        family["id"],
        MappingProxyType(captures),
        _SPECIFICATIONS[analysis_ref],
    )


def resolve_metric_analysis(metric_id: str) -> CatalogMetricAnalysisMatch:
    """Resolve exactly one registered metric, rejecting unknown or ambiguity."""

    if (
        not isinstance(metric_id, str)
        or not metric_id
        or metric_id.strip() != metric_id
    ):
        raise ValueError("evaluation metric id must be a trimmed non-empty string")
    return _resolve_from_parts(metric_id)


def static_metric_ids_for_track(track_id: str) -> tuple[str, ...]:
    if track_id not in _TRACK_IDS:
        raise ValueError(f"unknown evaluation track: {track_id}")
    return tuple(
        item["id"]
        for item in _DOCUMENT["static_metrics"]
        if _SPECIFICATIONS[item["analysis_ref"]].track_id == track_id
    )


_DOCUMENT = _load_document()
_ENCODING = MappingProxyType(_DOCUMENT["identifier_encoding"])
_SPECIFICATIONS = MappingProxyType(
    {
        item["id"]: CatalogMetricAnalysisSpecification(
            analysis_ref=item["id"],
            track_id=item["track_id"],
            estimator_id=item["estimator_id"],
            estimator_version=item["estimator_version"],
            analysis_unit=item["analysis_unit"],
            cluster_unit=item["cluster_unit"],
            weighting=item["weighting"],
            missingness=item["missingness"],
            exclusion_policy=item["exclusion_policy"],
            planned_unit_projection=MappingProxyType(item["planned_unit_projection"]),
        )
        for item in _DOCUMENT["analysis_templates"]
    }
)
_STATIC_BY_ID = MappingProxyType(
    {item["id"]: item for item in _DOCUMENT["static_metrics"]}
)
_COMPILED_FAMILIES = tuple(
    (item, re.compile(item["pattern"])) for item in _DOCUMENT["dynamic_families"]
)
STATIC_METRIC_IDS = tuple(_STATIC_BY_ID)
DYNAMIC_FAMILY_IDS = tuple(item["id"] for item, _ in _COMPILED_FAMILIES)
