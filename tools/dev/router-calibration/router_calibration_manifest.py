"""Manifest parsing and decision-level robustness helpers for router calibration."""

from __future__ import annotations

import base64
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from router_calibration_probe import (
    ACCEPTANCE_FIELDS,
    DECISION_FIELDS,
    EVALUATION_FIELDS,
    PROBE_SCHEMA_VERSION,
    ROUTING_ASSET_FIELDS,
    TOP_LEVEL_FIELDS,
    VARIANT_FIELDS,
    Probe,
    ProbeGeneratedText,
    ProbeImageFixture,
    ProbePadding,
    ProbePlaygroundPolicy,
    load_grouped_probes,
    normalize_image_fixtures,
    reject_unknown_fields,
)
from router_calibration_schema import validate_probe_manifest_schema
from yaml.nodes import MappingNode, Node, SequenceNode
from yaml.tokens import AliasToken, AnchorToken, TagToken

__all__ = [
    "DECISION_FIELDS",
    "TOP_LEVEL_FIELDS",
    "VARIANT_FIELDS",
    "Probe",
    "ProbeGeneratedText",
    "ProbeImageFixture",
    "ProbePadding",
    "ProbePlaygroundPolicy",
    "load_probe_manifest",
    "report_safe_probe_manifest",
]

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_probe_manifest(path: Path) -> tuple[dict[str, Any], list[Probe]]:
    document = path.read_text(encoding="utf-8")
    reject_yaml_indirection(document, path)
    manifest = yaml.safe_load(document) or {}
    if not isinstance(manifest, dict):
        raise TypeError(f"{path} must contain a mapping")
    validate_probe_manifest_schema(manifest, path)
    reject_unknown_fields(manifest, TOP_LEVEL_FIELDS, str(path))
    version = str(manifest.get("schema_version") or "").strip()
    if version != PROBE_SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version must be {PROBE_SCHEMA_VERSION!r}, got {version!r}"
        )
    if not str(manifest.get("name") or "").strip():
        raise ValueError(f"{path} must declare a non-empty name")
    _validate_manifest_settings(path, manifest)
    image_fixtures = normalize_image_fixtures(manifest.get("fixtures"), path)
    raw_decisions = manifest.get("decisions")
    if isinstance(raw_decisions, list) and raw_decisions:
        return manifest, load_grouped_probes(raw_decisions, path, image_fixtures)

    raise ValueError(f"{path} must contain a non-empty 'decisions' list")


def reject_yaml_indirection(document: str, path: Path) -> None:
    """Reject YAML features that can hide or multiply untrusted probe data."""
    for token in yaml.scan(document):
        if isinstance(token, AnchorToken):
            raise ValueError(f"{path} must not contain YAML anchors")
        if isinstance(token, AliasToken):
            raise ValueError(f"{path} must not contain YAML aliases")
        if isinstance(token, TagToken):
            raise ValueError(f"{path} must not contain explicit YAML tags")
    root = yaml.compose(document)
    if root is not None and _contains_yaml_merge(root):
        raise ValueError(f"{path} must not contain YAML merge keys")


def _contains_yaml_merge(node: Node) -> bool:
    if isinstance(node, MappingNode):
        for key, value in node.value:
            if key.tag == "tag:yaml.org,2002:merge":
                return True
            if _contains_yaml_merge(key) or _contains_yaml_merge(value):
                return True
    elif isinstance(node, SequenceNode):
        return any(_contains_yaml_merge(item) for item in node.value)
    return False


def report_safe_probe_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Keep fixture identity and integrity metadata while omitting binary payloads."""
    safe_manifest = copy.deepcopy(manifest)
    fixtures = safe_manifest.get("fixtures")
    images = fixtures.get("images") if isinstance(fixtures, dict) else None
    if not isinstance(images, dict):
        return safe_manifest
    for fixture in images.values():
        if not isinstance(fixture, dict):
            continue
        data_base64 = fixture.pop("data_base64", "")
        compact = "".join(str(data_base64).split())
        fixture["bytes"] = len(base64.b64decode(compact, validate=True))
    return safe_manifest


def _validate_manifest_settings(path: Path, manifest: dict[str, Any]) -> None:
    assets = manifest.get("routing_assets")
    if not isinstance(assets, dict):
        raise TypeError(f"{path} routing_assets must be a mapping")
    reject_unknown_fields(assets, ROUTING_ASSET_FIELDS, f"{path} routing_assets")
    for field in sorted(ROUTING_ASSET_FIELDS):
        if not str(assets.get(field) or "").strip():
            raise ValueError(f"{path} routing_assets.{field} is required")

    evaluation = manifest.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise TypeError(f"{path} evaluation must be a mapping")
    reject_unknown_fields(evaluation, EVALUATION_FIELDS, f"{path} evaluation")

    acceptance = manifest.get("acceptance", {})
    if not isinstance(acceptance, dict):
        raise TypeError(f"{path} acceptance must be a mapping")
    reject_unknown_fields(acceptance, ACCEPTANCE_FIELDS, f"{path} acceptance")


def summarize_decision_results(
    results: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        key = str(result.get("decision_id") or result.get("expected_decision") or "")
        grouped[key].append(result)

    decision_specs = _decision_spec_lookup(manifest)
    acceptance = resolve_acceptance(manifest)
    summaries: list[dict[str, Any]] = []
    for decision_id in sorted(grouped):
        variants = grouped[decision_id]
        matched = sum(1 for variant in variants if variant["matched"])
        total = len(variants)
        pass_rate = round((matched / total) * 100, 1) if total else 0.0
        spec = decision_specs.get(decision_id, {})
        min_pass_rate = _coerce_percent(
            spec.get("robustness", {}).get("min_pass_rate"),
            acceptance["min_decision_pass_rate"],
        )
        summaries.append(
            {
                "decision_id": decision_id,
                "expected_decision": variants[0]["expected_decision"],
                "model": variants[0].get("model"),
                "expected_recipe": variants[0].get("expected_recipe"),
                "expected_algorithm": variants[0].get("expected_algorithm"),
                "expected_alias": variants[0].get("expected_alias"),
                "matched": matched,
                "total": total,
                "pass_rate": pass_rate,
                "required_pass_rate": min_pass_rate,
                "passed": pass_rate >= min_pass_rate,
                "failing_variants": [
                    variant["id"] for variant in variants if not variant["matched"]
                ],
                "variants": [
                    {
                        "id": variant["id"],
                        "variant_id": variant["variant_id"],
                        "matched": variant["matched"],
                        "actual_decision": variant["actual_decision"],
                        "actual_recipe": variant.get("actual_recipe"),
                        "actual_algorithm": variant.get("actual_algorithm"),
                        "error": variant.get("error"),
                        "tags": variant.get("tags") or [],
                    }
                    for variant in variants
                ],
            }
        )
    return summaries


def summarize_tag_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize robustness dimensions encoded as stable manifest tags."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for tag in result.get("tags") or []:
            grouped[str(tag)].append(result)

    summaries: list[dict[str, Any]] = []
    for tag in sorted(grouped):
        variants = grouped[tag]
        matched = sum(1 for variant in variants if variant["matched"])
        total = len(variants)
        summaries.append(
            {
                "tag": tag,
                "matched": matched,
                "total": total,
                "pass_rate": round((matched / total) * 100, 1) if total else 0.0,
                "passed": matched == total,
                "failing_variants": [
                    variant["id"] for variant in variants if not variant["matched"]
                ],
            }
        )
    return summaries


def resolve_acceptance(manifest: dict[str, Any]) -> dict[str, float]:
    acceptance = manifest.get("acceptance")
    if not isinstance(acceptance, dict):
        acceptance = {}
    return {
        "min_probe_pass_rate": _coerce_percent(
            acceptance.get("min_probe_pass_rate"), 100.0
        ),
        "min_decision_pass_rate": _coerce_percent(
            acceptance.get("min_decision_pass_rate"), 100.0
        ),
    }


def resolve_manifest_assets(
    manifest: dict[str, Any], yaml_override: Path | None, dsl_override: Path | None
) -> tuple[Path | None, Path | None]:
    routing_assets = manifest.get("routing_assets")
    yaml_path = _require_existing_path(
        yaml_override
        or _resolve_manifest_path(
            routing_assets.get("yaml") if isinstance(routing_assets, dict) else None
        ),
        "yaml",
    )
    dsl_path = _require_existing_path(
        dsl_override
        or _resolve_manifest_path(
            routing_assets.get("dsl") if isinstance(routing_assets, dict) else None
        ),
        "dsl",
    )
    return yaml_path, dsl_path


def _resolve_manifest_path(path_value: Any) -> Path | None:
    raw_path = str(path_value or "").strip()
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / raw_path
    if repo_candidate.exists():
        return repo_candidate
    return Path.cwd() / raw_path


def _require_existing_path(path: Path | None, label: str) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    raise FileNotFoundError(f"{label} asset does not exist: {path}")


def _decision_spec_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_decisions = manifest.get("decisions")
    if not isinstance(raw_decisions, list):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for item in raw_decisions:
        if not isinstance(item, dict):
            continue
        decision_id = str(item.get("id") or item.get("expected_decision") or "").strip()
        if decision_id:
            lookup[decision_id] = item
    return lookup


def _coerce_percent(raw_value: Any, default: float) -> float:
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(100.0, round(value, 1)))
