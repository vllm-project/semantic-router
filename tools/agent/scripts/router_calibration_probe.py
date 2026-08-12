"""Probe schema types and normalization for router calibration manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PADDING_PLACEMENTS = frozenset({"before", "after", "around"})
PROBE_SCHEMA_VERSION = "v1"
MATCH_MODES = frozenset({"contains", "exact"})
MAX_PROBE_REPEAT = 10_000

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "description",
        "routing_assets",
        "router_eval_endpoint",
        "evaluation",
        "acceptance",
        "coverage",
        "decisions",
    }
)
ROUTING_ASSET_FIELDS = frozenset({"yaml", "dsl"})
EVALUATION_FIELDS = frozenset({"request_timeout_seconds", "concurrency"})
ACCEPTANCE_FIELDS = frozenset({"min_probe_pass_rate", "min_decision_pass_rate"})
DECISION_FIELDS = frozenset(
    {
        "id",
        "expected_decision",
        "model",
        "expected_recipe",
        "expected_algorithm",
        "expected_plugins",
        "forbidden_plugins",
        "plugin_match",
        "expected_alias",
        "expected_signals",
        "forbidden_signals",
        "signal_match",
        "robustness",
        "objective",
        "notes",
        "variants",
    }
)
VARIANT_FIELDS = frozenset(
    {
        "id",
        "query",
        "messages",
        "tools",
        "repeat",
        "padding",
        "tags",
        "notes",
        "expected_signals",
    }
)
PADDING_FIELDS = frozenset({"text", "repeat", "placement"})
ROBUSTNESS_FIELDS = frozenset({"min_pass_rate"})


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
    forbidden_plugins: tuple[str, ...] = ()
    plugin_match: str = "contains"
    expected_signals: tuple[tuple[str, str], ...] = ()
    forbidden_signals: tuple[tuple[str, str], ...] = ()
    signal_match: str = "contains"
    query: str | None = None
    repeat: int = 1
    padding: ProbePadding | None = None
    messages: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    expected_alias: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionDefaults:
    decision_id: str
    expected_decision: str
    model: str | None
    expected_recipe: str | None
    expected_algorithm: str | None
    expected_plugins: tuple[str, ...]
    forbidden_plugins: tuple[str, ...]
    plugin_match: str
    expected_alias: str | None
    expected_signals: tuple[tuple[str, str], ...]
    forbidden_signals: tuple[tuple[str, str], ...]
    signal_match: str
    notes: str | None


def load_grouped_probes(raw_decisions: list[Any], manifest_path: Path) -> list[Probe]:
    probes: list[Probe] = []
    decision_ids: set[str] = set()
    probe_ids: set[str] = set()
    for index, raw_decision in enumerate(raw_decisions):
        defaults, raw_variants = _load_decision_defaults(
            raw_decision,
            index,
            manifest_path,
            decision_ids,
        )
        for variant_index, raw_variant in enumerate(raw_variants):
            probe = _load_variant(
                raw_variant,
                variant_index,
                defaults,
                manifest_path,
                index,
            )
            if probe.probe_id in probe_ids:
                raise ValueError(
                    f"{manifest_path} contains duplicate probe id {probe.probe_id!r}"
                )
            probe_ids.add(probe.probe_id)
            probes.append(probe)
    return probes


def _load_decision_defaults(
    raw_decision: Any,
    index: int,
    manifest_path: Path,
    decision_ids: set[str],
) -> tuple[DecisionDefaults, list[Any]]:
    if not isinstance(raw_decision, dict):
        raise TypeError(f"decision[{index}] must be a mapping")
    label = f"{manifest_path} decisions[{index}]"
    reject_unknown_fields(raw_decision, DECISION_FIELDS, label)
    decision_id = str(
        raw_decision.get("id") or raw_decision.get("expected_decision") or ""
    ).strip()
    expected = str(raw_decision.get("expected_decision") or decision_id).strip()
    if not decision_id or not expected:
        raise ValueError(
            f"decision[{index}] must include non-empty id or expected_decision"
        )
    if decision_id in decision_ids:
        raise ValueError(
            f"{manifest_path} contains duplicate decision id {decision_id!r}"
        )
    decision_ids.add(decision_id)
    robustness = raw_decision.get("robustness", {})
    if not isinstance(robustness, dict):
        raise TypeError(f"{label}.robustness must be a mapping")
    reject_unknown_fields(robustness, ROBUSTNESS_FIELDS, f"{label}.robustness")
    variants = raw_decision.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"decision[{index}] must include a non-empty 'variants' list")
    return (
        DecisionDefaults(
            decision_id=decision_id,
            expected_decision=expected,
            model=_optional_string(raw_decision.get("model")),
            expected_recipe=_optional_string(raw_decision.get("expected_recipe")),
            expected_algorithm=_optional_string(raw_decision.get("expected_algorithm")),
            expected_plugins=_normalize_string_list(
                raw_decision.get("expected_plugins"), "expected_plugins"
            ),
            forbidden_plugins=_normalize_string_list(
                raw_decision.get("forbidden_plugins"), "forbidden_plugins"
            ),
            plugin_match=_normalize_match_mode(
                raw_decision.get("plugin_match"), f"{label}.plugin_match"
            ),
            expected_alias=_optional_string(raw_decision.get("expected_alias")),
            expected_signals=_normalize_expected_signals(
                raw_decision.get("expected_signals"), decision_id
            ),
            forbidden_signals=_normalize_expected_signals(
                raw_decision.get("forbidden_signals"),
                f"{decision_id}.forbidden_signals",
            ),
            signal_match=_normalize_match_mode(
                raw_decision.get("signal_match"), f"{label}.signal_match"
            ),
            notes=_optional_string(
                raw_decision.get("notes") or raw_decision.get("objective")
            ),
        ),
        variants,
    )


def _load_variant(
    raw_variant: Any,
    variant_index: int,
    defaults: DecisionDefaults,
    manifest_path: Path,
    decision_index: int,
) -> Probe:
    if not isinstance(raw_variant, dict):
        raise TypeError(
            f"decision[{decision_index}].variants[{variant_index}] must be a mapping"
        )
    label = f"{manifest_path} decisions[{decision_index}].variants[{variant_index}]"
    reject_unknown_fields(raw_variant, VARIANT_FIELDS, label)
    variant_id = str(raw_variant.get("id") or f"v{variant_index + 1}").strip()
    query = str(raw_variant.get("query") or "").strip()
    messages = _normalize_objects(raw_variant.get("messages"), "messages")
    if not variant_id or bool(query) == bool(messages):
        raise ValueError(
            f"{label} must include a non-empty id and exactly one of query or messages"
        )
    probe_id = f"{defaults.decision_id}:{variant_id}"
    return Probe(
        decision_id=defaults.decision_id,
        variant_id=variant_id,
        probe_id=probe_id,
        expected_decision=defaults.expected_decision,
        model=defaults.model,
        expected_recipe=defaults.expected_recipe,
        expected_algorithm=defaults.expected_algorithm,
        expected_plugins=defaults.expected_plugins,
        forbidden_plugins=defaults.forbidden_plugins,
        plugin_match=defaults.plugin_match,
        expected_signals=_normalize_expected_signals(
            raw_variant.get("expected_signals"),
            probe_id,
            default=defaults.expected_signals,
        ),
        forbidden_signals=defaults.forbidden_signals,
        signal_match=defaults.signal_match,
        query=query or None,
        repeat=_normalize_repeat(
            raw_variant.get("repeat"), defaults.decision_id, variant_id
        ),
        padding=_normalize_padding(
            raw_variant.get("padding"), defaults.decision_id, variant_id
        ),
        messages=messages,
        tools=_normalize_objects(raw_variant.get("tools"), "tools"),
        expected_alias=defaults.expected_alias,
        notes=_optional_string(raw_variant.get("notes")) or defaults.notes,
        tags=_normalize_tags(raw_variant.get("tags")),
    )


def reject_unknown_fields(
    value: dict[str, Any], allowed: frozenset[str], label: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")


def _optional_string(value: Any) -> str | None:
    return str(value or "").strip() or None


def _normalize_tags(raw_tags: Any) -> tuple[str, ...]:
    if not isinstance(raw_tags, list):
        return ()
    normalized = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    if len(normalized) != len(set(normalized)):
        raise ValueError("tags must not contain duplicates")
    return tuple(normalized)


def _normalize_match_mode(raw_mode: Any, label: str) -> str:
    mode = str(raw_mode or "contains").strip().lower()
    if mode not in MATCH_MODES:
        supported = ", ".join(sorted(MATCH_MODES))
        raise ValueError(f"{label} must be one of: {supported}")
    return mode


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
            f"{decision_id}:{variant_id} repeat must be between 1 and "
            f"{MAX_PROBE_REPEAT}"
        )
    return repeat


def _normalize_padding(
    raw_padding: Any, decision_id: str, variant_id: str
) -> ProbePadding | None:
    if raw_padding is None:
        return None
    if not isinstance(raw_padding, dict):
        raise TypeError(f"{decision_id}:{variant_id} padding must be a mapping")
    reject_unknown_fields(
        raw_padding, PADDING_FIELDS, f"{decision_id}:{variant_id} padding"
    )
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


def _normalize_objects(raw_items: Any, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_items, list):
        return ()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise TypeError(f"{label}[{index}] must be a mapping")
        normalized.append(item)
    return tuple(normalized)
