"""Manifest parsing and decision-level robustness helpers for router calibration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PADDING_PLACEMENTS = frozenset({"before", "after", "around"})
MAX_PROBE_REPEAT = 10_000


@dataclass(frozen=True)
class ProbePadding:
    text: str
    repeat: int
    placement: str


@dataclass
class Probe:
    decision_id: str
    variant_id: str
    probe_id: str
    expected_decision: str
    model: str | None = None
    expected_recipe: str | None = None
    expected_algorithm: str | None = None
    expected_plugins: tuple[str, ...] = ()
    expected_signals: tuple[tuple[str, str], ...] = ()
    query: str | None = None
    repeat: int = 1
    padding: ProbePadding | None = None
    messages: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    expected_alias: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()


def load_probe_manifest(path: Path) -> tuple[dict[str, Any], list[Probe]]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_decisions = manifest.get("decisions")
    if isinstance(raw_decisions, list) and raw_decisions:
        return manifest, _load_grouped_probes(raw_decisions)

    raise ValueError(f"{path} must contain a non-empty 'decisions' list")


def _load_grouped_probes(raw_decisions: list[Any]) -> list[Probe]:
    probes: list[Probe] = []
    for index, item in enumerate(raw_decisions):
        if not isinstance(item, dict):
            raise TypeError(f"decision[{index}] must be a mapping")
        decision_id = str(item.get("id") or item.get("expected_decision") or "").strip()
        expected = str(item.get("expected_decision") or decision_id).strip()
        model = str(item.get("model") or "").strip() or None
        expected_recipe = str(item.get("expected_recipe") or "").strip() or None
        expected_algorithm = str(item.get("expected_algorithm") or "").strip() or None
        expected_plugins = _normalize_string_list(
            item.get("expected_plugins"), "expected_plugins"
        )
        expected_alias = str(item.get("expected_alias") or "").strip() or None
        decision_expected_signals = _normalize_expected_signals(
            item.get("expected_signals"), decision_id
        )
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
                raise TypeError(
                    f"decision[{index}].variants[{variant_index}] must be a mapping"
                )
            variant_id = str(variant.get("id") or f"v{variant_index + 1}").strip()
            query = str(variant.get("query") or "").strip()
            repeat = _normalize_repeat(variant.get("repeat"), decision_id, variant_id)
            padding = _normalize_padding(
                variant.get("padding"), decision_id, variant_id
            )
            messages = _normalize_messages(variant.get("messages"))
            tools = _normalize_objects(variant.get("tools"), "tools")
            if not variant_id or (not query and not messages):
                raise ValueError(
                    f"decision[{index}].variants[{variant_index}] must include non-empty id and either query or messages"
                )
            probes.append(
                Probe(
                    decision_id=decision_id,
                    variant_id=variant_id,
                    probe_id=f"{decision_id}:{variant_id}",
                    expected_decision=expected,
                    model=model,
                    expected_recipe=expected_recipe,
                    expected_algorithm=expected_algorithm,
                    expected_plugins=expected_plugins,
                    expected_signals=_normalize_expected_signals(
                        variant.get("expected_signals"),
                        f"{decision_id}:{variant_id}",
                        default=decision_expected_signals,
                    ),
                    query=query or None,
                    repeat=repeat,
                    padding=padding,
                    messages=messages,
                    tools=tools,
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


def _normalize_tags(raw_tags: Any) -> tuple[str, ...]:
    if not isinstance(raw_tags, list):
        return ()
    normalized = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    return tuple(normalized)


def _normalize_string_list(raw_items: Any, label: str) -> tuple[str, ...]:
    if raw_items is None:
        return ()
    if not isinstance(raw_items, list):
        raise TypeError(f"{label} must be a list")
    normalized = tuple(str(item).strip() for item in raw_items if str(item).strip())
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must not contain duplicates")
    return normalized


def _normalize_expected_signals(
    raw_signals: Any,
    label: str,
    *,
    default: tuple[tuple[str, str], ...] = (),
) -> tuple[tuple[str, str], ...]:
    if raw_signals is None:
        return default
    if not isinstance(raw_signals, dict):
        raise TypeError(f"{label} expected_signals must be a mapping")

    normalized: list[tuple[str, str]] = []
    for signal_type, raw_names in raw_signals.items():
        normalized_type = str(signal_type).strip()
        if not normalized_type:
            raise ValueError(f"{label} expected_signals contains an empty type")
        names = _normalize_string_list(
            raw_names, f"{label} expected_signals.{normalized_type}"
        )
        if not names:
            raise ValueError(
                f"{label} expected_signals.{normalized_type} must not be empty"
            )
        normalized.extend((normalized_type, name) for name in names)
    return tuple(normalized)


def _normalize_repeat(raw_repeat: Any, decision_id: str, variant_id: str) -> int:
    if raw_repeat is None:
        return 1
    try:
        repeat = int(raw_repeat)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{decision_id}:{variant_id} repeat must be an integer"
        ) from exc
    if repeat < 1 or repeat > MAX_PROBE_REPEAT:
        raise ValueError(
            f"{decision_id}:{variant_id} repeat must be between 1 and {MAX_PROBE_REPEAT}"
        )
    return repeat


def _normalize_padding(
    raw_padding: Any, decision_id: str, variant_id: str
) -> ProbePadding | None:
    if raw_padding is None:
        return None
    if not isinstance(raw_padding, dict):
        raise TypeError(f"{decision_id}:{variant_id} padding must be a mapping")

    text = str(raw_padding.get("text") or "").strip()
    if not text:
        raise ValueError(f"{decision_id}:{variant_id} padding.text is required")
    repeat = _normalize_repeat(
        raw_padding.get("repeat"), decision_id, f"{variant_id} padding"
    )
    placement = str(raw_padding.get("placement") or "before").strip().lower()
    if placement not in PADDING_PLACEMENTS:
        supported = ", ".join(sorted(PADDING_PLACEMENTS))
        raise ValueError(
            f"{decision_id}:{variant_id} padding.placement must be one of: {supported}"
        )
    return ProbePadding(text=text, repeat=repeat, placement=placement)


def _normalize_messages(raw_messages: Any) -> tuple[dict[str, Any], ...]:

    return _normalize_objects(raw_messages, "messages")


def _normalize_objects(raw_items: Any, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_items, list):
        return ()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise TypeError(f"{label}[{index}] must be a mapping")
        normalized.append(item)
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
