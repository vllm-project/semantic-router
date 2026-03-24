"""Manifest parsing and decision-level robustness helpers for router calibration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Probe:
    decision_id: str
    variant_id: str
    probe_id: str
    expected_decision: str
    query: str
    expected_alias: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()


def load_probe_manifest(path: Path) -> tuple[dict[str, Any], list[Probe]]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_decisions = manifest.get("decisions")
    if isinstance(raw_decisions, list) and raw_decisions:
        return manifest, _load_grouped_probes(raw_decisions)

    raw_probes = manifest.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError(
            f"{path} must contain a non-empty 'decisions' or legacy 'probes' list"
        )
    return manifest, _load_legacy_probes(raw_probes)


def _load_grouped_probes(raw_decisions: list[Any]) -> list[Probe]:
    probes: list[Probe] = []
    for index, item in enumerate(raw_decisions):
        if not isinstance(item, dict):
            raise ValueError(f"decision[{index}] must be a mapping")
        decision_id = str(item.get("id") or item.get("expected_decision") or "").strip()
        expected = str(item.get("expected_decision") or decision_id).strip()
        expected_alias = str(item.get("expected_alias") or "").strip() or None
        decision_notes = str(item.get("notes") or item.get("objective") or "").strip()
        raw_variants = item.get("variants")
        if not decision_id or not expected:
            raise ValueError(
                f"decision[{index}] must include non-empty id or expected_decision"
            )
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ValueError(
                f"decision[{index}] must include a non-empty 'variants' list"
            )
        for variant_index, variant in enumerate(raw_variants):
            if not isinstance(variant, dict):
                raise ValueError(
                    f"decision[{index}].variants[{variant_index}] must be a mapping"
                )
            variant_id = str(variant.get("id") or f"v{variant_index + 1}").strip()
            query = str(variant.get("query") or "").strip()
            if not variant_id or not query:
                raise ValueError(
                    f"decision[{index}].variants[{variant_index}] must include non-empty id and query"
                )
            probes.append(
                Probe(
                    decision_id=decision_id,
                    variant_id=variant_id,
                    probe_id=f"{decision_id}:{variant_id}",
                    expected_decision=expected,
                    query=query,
                    expected_alias=expected_alias,
                    notes=(
                        str(variant.get("notes") or "").strip()
                        or decision_notes
                        or None
                    ),
                    tags=_normalize_tags(variant.get("tags")),
                )
            )
    return probes


def _load_legacy_probes(raw_probes: list[Any]) -> list[Probe]:
    probes: list[Probe] = []
    for index, item in enumerate(raw_probes):
        if not isinstance(item, dict):
            raise ValueError(f"probe[{index}] must be a mapping")
        probe_id = str(item.get("id") or item.get("expected_decision") or "").strip()
        expected = str(item.get("expected_decision") or "").strip()
        query = str(item.get("query") or "").strip()
        if not probe_id or not expected or not query:
            raise ValueError(
                f"probe[{index}] must include non-empty id, expected_decision, and query"
            )
        probes.append(
            Probe(
                decision_id=expected,
                variant_id=probe_id,
                probe_id=probe_id,
                expected_decision=expected,
                query=query,
                expected_alias=str(item.get("expected_alias") or "").strip() or None,
                notes=str(item.get("notes") or "").strip() or None,
                tags=_normalize_tags(item.get("tags")),
            )
        )
    return probes


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
                        "tags": variant.get("tags") or [],
                    }
                    for variant in variants
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


def _normalize_tags(raw_tags: Any) -> tuple[str, ...]:
    if not isinstance(raw_tags, list):
        return ()
    normalized = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    return tuple(normalized)


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
