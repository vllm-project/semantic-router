"""Fail-closed validation for the packaged metric analysis catalog."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from typing import Any

from cli.evaluation.constants import TRACK_IDS as CANONICAL_TRACK_IDS

CATALOG_SCHEMA_VERSION = "metric-analysis-catalog.v1"
PROVENANCE_CONTRACT_VERSION = "metric-analysis.v1"

TRACK_IDS = frozenset(CANONICAL_TRACK_IDS)
_PROJECTION_SOURCES = frozenset(
    {
        "evaluation_case_plan",
        "frozen_model_pool_matrix",
        "method_ledger",
        "capacity_load_plan",
        "compound_budget_plan",
        "routing_recipe_plan",
    }
)
_CAPTURE_TYPES = frozenset({"encoded_portable_id", "positive_int", "enum"})
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9.-]{0,159}$")


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise RuntimeError(f"{context} fields are invalid")


def _sorted_unique(values: list[str], context: str) -> None:
    if values != sorted(values) or len(values) != len(set(values)):
        raise RuntimeError(f"{context} must be sorted and unique")


def _validate_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise RuntimeError(f"{context} identifier is invalid")
    return value


def _validate_projection_rows(value: Mapping[str, Any], field: str) -> None:
    rows = value.get(field, [])
    if not isinstance(rows, list) or any(
        not isinstance(item, str) or not item or item.strip() != item for item in rows
    ):
        raise RuntimeError("planned-unit projection coordinates are invalid")
    if len(rows) != len(set(rows)):
        raise RuntimeError(f"planned-unit projection {field} must be unique")


def _validate_projection_filters(
    value: Mapping[str, Any], captures: set[str] | None
) -> None:
    filters = value.get("filters", [])
    if not isinstance(filters, list):
        raise RuntimeError("planned-unit projection filters are invalid")
    filter_fields: list[str] = []
    for item in filters:
        if not isinstance(item, dict):
            raise RuntimeError("planned-unit projection filter must be an object")
        _exact_keys(item, {"field", "capture"}, "planned-unit projection filter")
        if (
            not isinstance(item["field"], str)
            or not item["field"]
            or not isinstance(item["capture"], str)
            or (captures is not None and item["capture"] not in captures)
        ):
            raise RuntimeError("planned-unit projection filter is invalid")
        filter_fields.append(item["field"])
    if len(filter_fields) != len(set(filter_fields)):
        raise RuntimeError("planned-unit projection filter fields must be unique")


def validate_projection(value: Any, captures: set[str] | None = None) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("planned-unit projection must be an object")
    allowed = {"source", "track_id", "coordinates", "required_dimensions", "filters"}
    required = {"source", "track_id", "coordinates"}
    if set(value) - allowed or not required <= set(value):
        raise RuntimeError("planned-unit projection fields are invalid")
    if value["source"] not in _PROJECTION_SOURCES or value["track_id"] not in TRACK_IDS:
        raise RuntimeError("planned-unit projection source or track is invalid")
    _validate_projection_rows(value, "coordinates")
    _validate_projection_rows(value, "required_dimensions")
    if not value["coordinates"]:
        raise RuntimeError("planned-unit projection requires a coordinate")
    _validate_projection_filters(value, captures)


def _validate_template(value: Any) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("analysis template must be an object")
    _exact_keys(
        value,
        {
            "id",
            "track_id",
            "estimator_id",
            "estimator_version",
            "analysis_unit",
            "cluster_unit",
            "weighting",
            "missingness",
            "exclusion_policy",
            "planned_unit_projection",
        },
        "analysis template",
    )
    _validate_identifier(value["id"], "analysis template")
    for field in (
        "estimator_id",
        "estimator_version",
        "analysis_unit",
        "cluster_unit",
        "weighting",
    ):
        if (
            not isinstance(value[field], str)
            or not value[field]
            or value[field].strip() != value[field]
        ):
            raise RuntimeError(f"analysis template {field} is invalid")
    if value["track_id"] not in TRACK_IDS:
        raise RuntimeError("analysis template track is invalid")
    if (
        value["missingness"] != "fail_closed"
        or value["exclusion_policy"] != "exclude_unavailable_evidence"
    ):
        raise RuntimeError("analysis template missingness contract is invalid")
    validate_projection(value["planned_unit_projection"])


def _encoding(document: Mapping[str, Any]) -> Mapping[str, Any]:
    value = document.get("identifier_encoding")
    if not isinstance(value, dict):
        raise RuntimeError("metric identifier encoding is missing")
    _exact_keys(
        value,
        {
            "scheme",
            "raw_pattern",
            "direct_pattern",
            "reserved_prefix",
            "encoded_pattern",
            "vectors",
        },
        "metric identifier encoding",
    )
    if (
        value["scheme"] != "portable-segment-base64url.v1"
        or value["reserved_prefix"] != "u-"
    ):
        raise RuntimeError("metric identifier encoding version is invalid")
    for field in ("raw_pattern", "direct_pattern", "encoded_pattern"):
        if not isinstance(value[field], str):
            raise RuntimeError("metric identifier encoding pattern is invalid")
        re.compile(value[field])
    if not isinstance(value["vectors"], list) or not value["vectors"]:
        raise RuntimeError("metric identifier encoding vectors are missing")
    return value


def _validate_templates(document: Mapping[str, Any]) -> tuple[list[Any], set[str]]:
    templates = document["analysis_templates"]
    if not isinstance(templates, list) or not templates:
        raise RuntimeError("metric analysis catalog templates are missing")
    for item in templates:
        _validate_template(item)
    template_ids = [item["id"] for item in templates]
    _sorted_unique(template_ids, "analysis template ids")
    return templates, set(template_ids)


def _validate_capture(capture: Any, index: int) -> str:
    if (
        not isinstance(capture, dict)
        or capture.get("group") != index
        or capture.get("type") not in _CAPTURE_TYPES
    ):
        raise RuntimeError("dynamic family capture is invalid")
    expected = {"name", "group", "type"}
    if capture["type"] == "enum":
        expected.add("values")
        values = capture.get("values")
        if not isinstance(values, list) or not values:
            raise RuntimeError("dynamic family enum capture is empty")
        _sorted_unique(values, "dynamic family enum values")
    elif capture["type"] == "positive_int":
        expected |= {"minimum", "maximum"}
        minimum = capture.get("minimum")
        maximum = capture.get("maximum")
        if (
            not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum < 1
            or maximum < minimum
        ):
            raise RuntimeError("dynamic family integer bounds are invalid")
    _exact_keys(capture, expected, "dynamic family capture")
    return capture["name"]


def _validate_variants(
    family: Mapping[str, Any],
    captures: list[Mapping[str, Any]],
    names: list[str],
    templates_by_id: Mapping[str, Any],
) -> None:
    selector_name = family["selector_capture"]
    if selector_name not in names:
        raise RuntimeError("dynamic family selector capture is invalid")
    selector = next(item for item in captures if item["name"] == selector_name)
    variants = family["variants"]
    if not isinstance(variants, list) or not variants:
        raise RuntimeError("dynamic family variants are missing")
    variant_values = [item.get("value") for item in variants if isinstance(item, dict)]
    if len(variant_values) != len(variants):
        raise RuntimeError("dynamic family variant is invalid")
    _sorted_unique(variant_values, "dynamic family variant values")
    for item in variants:
        _exact_keys(item, {"value", "analysis_ref"}, "dynamic family variant")
        if item["analysis_ref"] not in templates_by_id:
            raise RuntimeError("dynamic family variant references an unknown analysis")
    expected = selector.get("values") if selector["type"] == "enum" else ["*"]
    if variant_values != expected:
        raise RuntimeError("dynamic family variants do not cover their selector")
    for item in variants:
        validate_projection(
            templates_by_id[item["analysis_ref"]]["planned_unit_projection"],
            set(names),
        )


def _validate_family(
    family: dict[str, Any], templates_by_id: Mapping[str, Any]
) -> tuple[str, re.Pattern[str]]:
    _exact_keys(
        family,
        {
            "id",
            "literal_prefix",
            "pattern",
            "captures",
            "selector_capture",
            "variants",
            "examples",
        },
        "dynamic family",
    )
    _validate_identifier(family["id"], "dynamic family")
    prefix = family["literal_prefix"]
    if not isinstance(prefix, str) or not prefix:
        raise RuntimeError("dynamic family literal prefix is invalid")
    grammar = family["pattern"]
    if (
        not isinstance(grammar, str)
        or not grammar.startswith("^")
        or not grammar.endswith("$")
    ):
        raise RuntimeError("dynamic family grammar must be anchored")
    pattern = re.compile(grammar)
    captures = family["captures"]
    if not isinstance(captures, list) or len(captures) != pattern.groups:
        raise RuntimeError("dynamic family captures do not match its grammar")
    names = [
        _validate_capture(capture, index)
        for index, capture in enumerate(captures, start=1)
    ]
    if len(names) != len(set(names)):
        raise RuntimeError("dynamic family capture names must be sorted and unique")
    _validate_variants(family, captures, names, templates_by_id)
    return prefix, pattern


def _validate_families(
    document: Mapping[str, Any], templates: list[Any], refs: set[str]
) -> list[tuple[dict[str, Any], re.Pattern[str]]]:
    families = document["dynamic_families"]
    if not isinstance(families, list) or not families:
        raise RuntimeError("metric analysis catalog dynamic families are missing")
    family_ids = [item.get("id") for item in families if isinstance(item, dict)]
    if len(family_ids) != len(families) or any(
        not isinstance(item, str) for item in family_ids
    ):
        raise RuntimeError("dynamic family identity is invalid")
    _sorted_unique(family_ids, "dynamic family ids")
    templates_by_id = {item["id"]: item for item in templates if item["id"] in refs}
    validated = [_validate_family(family, templates_by_id) for family in families]
    prefixes = [prefix for prefix, _ in validated]
    for index, left in enumerate(prefixes):
        for right in prefixes[index + 1 :]:
            if left.startswith(right) or right.startswith(left):
                raise RuntimeError("dynamic family literal prefixes may overlap")
    return [(family, validated[index][1]) for index, family in enumerate(families)]


def _validate_static_metrics(
    document: Mapping[str, Any],
    refs: set[str],
    compiled: list[tuple[dict[str, Any], re.Pattern[str]]],
) -> None:
    static = document["static_metrics"]
    if not isinstance(static, list) or not static:
        raise RuntimeError("metric analysis catalog static metrics are missing")
    static_ids: list[str] = []
    for item in static:
        if not isinstance(item, dict):
            raise RuntimeError("static metric entry must be an object")
        _exact_keys(item, {"id", "analysis_ref"}, "static metric")
        if (
            not isinstance(item["id"], str)
            or not item["id"]
            or item["analysis_ref"] not in refs
        ):
            raise RuntimeError("static metric entry is invalid")
        if any(pattern.fullmatch(item["id"]) for _, pattern in compiled):
            raise RuntimeError("static metric overlaps a dynamic family")
        static_ids.append(item["id"])
    _sorted_unique(static_ids, "static metric ids")


def _validate_exhaustive_template_references(
    document: Mapping[str, Any], template_ids: set[str]
) -> None:
    referenced = {item["analysis_ref"] for item in document["static_metrics"]} | {
        variant["analysis_ref"]
        for family in document["dynamic_families"]
        for variant in family["variants"]
    }
    if referenced != template_ids:
        raise RuntimeError("metric analysis templates must be referenced exhaustively")


def _canonical_encoded_value(raw: str, encoding: Mapping[str, Any]) -> str:
    direct = (
        not raw.startswith(encoding["reserved_prefix"])
        and re.fullmatch(encoding["direct_pattern"], raw) is not None
    )
    if direct:
        return raw
    payload = base64.urlsafe_b64encode(raw.encode("ascii")).decode("ascii").rstrip("=")
    return encoding["reserved_prefix"] + payload


def _validate_encoding_vectors(encoding: Mapping[str, Any]) -> None:
    raw_pattern = re.compile(encoding["raw_pattern"])
    encoded_pattern = re.compile(encoding["encoded_pattern"])
    for vector in encoding["vectors"]:
        if not isinstance(vector, dict):
            raise RuntimeError("identifier encoding vector is invalid")
        _exact_keys(vector, {"raw", "encoded"}, "identifier encoding vector")
        if raw_pattern.fullmatch(vector["raw"]) is None:
            raise RuntimeError("identifier encoding raw vector is invalid")
        expected = _canonical_encoded_value(vector["raw"], encoding)
        is_direct = expected == vector["raw"]
        if expected != vector["encoded"] or (
            not is_direct and encoded_pattern.fullmatch(expected) is None
        ):
            raise RuntimeError("identifier encoding vector is not canonical")


def _decode_capture(raw: str, encoding: Mapping[str, Any]) -> str:
    payload = raw[len(encoding["reserved_prefix"]) :]
    try:
        return base64.b64decode(
            payload + "=" * (-len(payload) % 4), altchars=b"-_", validate=True
        ).decode("ascii")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise RuntimeError("dynamic family encoded capture is invalid") from exc


def _capture_values_with_encoding(
    family: Mapping[str, Any], match: re.Match[str], encoding: Mapping[str, Any]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for capture in family["captures"]:
        raw = match.group(capture["group"])
        if capture["type"] == "encoded_portable_id":
            if raw.startswith(encoding["reserved_prefix"]):
                decoded = _decode_capture(raw, encoding)
                if (
                    re.fullmatch(encoding["raw_pattern"], decoded) is None
                    or _canonical_encoded_value(decoded, encoding) != raw
                ):
                    raise RuntimeError(
                        "dynamic family encoded capture is non-canonical"
                    )
            elif re.fullmatch(encoding["direct_pattern"], raw) is None:
                raise RuntimeError("dynamic family direct capture is invalid")
        elif capture["type"] == "positive_int":
            number = int(raw)
            if (
                raw != str(number)
                or not capture["minimum"] <= number <= capture["maximum"]
            ):
                raise RuntimeError("dynamic family integer capture is invalid")
        elif capture["type"] == "enum" and raw not in capture["values"]:
            raise RuntimeError("dynamic family enum capture is invalid")
        result[capture["name"]] = raw
    return result


def _validate_example(
    family: Mapping[str, Any],
    example: Any,
    compiled: list[tuple[dict[str, Any], re.Pattern[str]]],
    encoding: Mapping[str, Any],
) -> None:
    if not isinstance(example, dict):
        raise RuntimeError("dynamic family example is invalid")
    _exact_keys(
        example,
        {"metric_id", "captures", "analysis_ref"},
        "dynamic family example",
    )
    matches = [
        (candidate, match)
        for candidate, pattern in compiled
        if (match := pattern.fullmatch(example["metric_id"])) is not None
    ]
    if len(matches) != 1 or matches[0][0]["id"] != family["id"]:
        raise RuntimeError("dynamic family example is unknown or ambiguous")
    actual = _capture_values_with_encoding(family, matches[0][1], encoding)
    if actual != example["captures"]:
        raise RuntimeError("dynamic family example captures drifted")
    selector_value = actual[family["selector_capture"]]
    variants = {item["value"]: item["analysis_ref"] for item in family["variants"]}
    if variants.get(selector_value, variants.get("*")) != example["analysis_ref"]:
        raise RuntimeError("dynamic family example analysis drifted")


def _validate_examples(
    compiled: list[tuple[dict[str, Any], re.Pattern[str]]],
    encoding: Mapping[str, Any],
) -> None:
    for family, _pattern in compiled:
        examples = family["examples"]
        if not isinstance(examples, list) or not examples:
            raise RuntimeError("dynamic family requires a golden example")
        for example in examples:
            _validate_example(family, example, compiled, encoding)


def validate_document(document: dict[str, Any]) -> None:
    """Validate the complete catalog through narrow, independently testable stages."""

    _exact_keys(
        document,
        {
            "schema_version",
            "provenance_contract_version",
            "identifier_encoding",
            "analysis_templates",
            "static_metrics",
            "dynamic_families",
        },
        "metric analysis catalog",
    )
    if (
        document["schema_version"] != CATALOG_SCHEMA_VERSION
        or document["provenance_contract_version"] != PROVENANCE_CONTRACT_VERSION
    ):
        raise RuntimeError("metric analysis catalog version is invalid")
    encoding = _encoding(document)
    templates, refs = _validate_templates(document)
    compiled = _validate_families(document, templates, refs)
    _validate_static_metrics(document, refs, compiled)
    _validate_exhaustive_template_references(document, refs)
    _validate_encoding_vectors(encoding)
    _validate_examples(compiled, encoding)
