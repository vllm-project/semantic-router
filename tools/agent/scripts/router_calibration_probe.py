"""Probe schema types and normalization for router calibration manifests."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from router_calibration_image import (
    IMAGE_FIXTURE_MEDIA_TYPES,
    validate_image_fixture_payload,
)

PADDING_PLACEMENTS = frozenset({"before", "after", "around"})
PROBE_SCHEMA_VERSION = "v1"
MATCH_MODES = frozenset({"contains", "exact"})
MAX_PROBE_REPEAT = 10_000
MAX_GENERATED_TEXT_BYTES = 10 << 20
MAX_IMAGE_FIXTURE_BYTES = 4 << 20
MAX_IMAGE_FIXTURE_NAME_LENGTH = 128
MAX_IMAGE_FIXTURE_DESCRIPTION_LENGTH = 512
FIXTURE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

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
        "fixtures",
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
        "display_prompt",
        "playground",
        "messages",
        "tools",
        "repeat",
        "padding",
        "generated_text",
        "tags",
        "notes",
        "expected_signals",
    }
)
PADDING_FIELDS = frozenset({"text", "repeat", "placement"})
GENERATED_TEXT_FIELDS = frozenset(
    {"message_index", "content_index", "target_text_bytes", "character"}
)
PLAYGROUND_FIELDS = frozenset({"enabled", "reason"})
ROBUSTNESS_FIELDS = frozenset({"min_pass_rate"})
FIXTURES_FIELDS = frozenset({"images"})
IMAGE_FIXTURE_FIELDS = frozenset({"description", "media_type", "data_base64", "sha256"})


@dataclass(frozen=True)
class ProbePadding:
    text: str
    repeat: int
    placement: str


@dataclass(frozen=True)
class ProbeGeneratedText:
    """Declarative text inserted into one message content array at request time."""

    message_index: int
    content_index: int
    target_text_bytes: int
    character: str = "x"


@dataclass(frozen=True)
class ProbeImageFixture:
    name: str
    description: str
    media_type: str
    data_base64: str = field(repr=False)
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ProbePlaygroundPolicy:
    enabled: bool
    reason: str | None = None


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
    display_prompt: str | None = None
    playground: ProbePlaygroundPolicy = ProbePlaygroundPolicy(enabled=True)
    repeat: int = 1
    padding: ProbePadding | None = None
    generated_text: ProbeGeneratedText | None = None
    messages: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    expected_alias: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()
    image_fixtures: dict[str, ProbeImageFixture] = field(
        default_factory=dict, repr=False
    )


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


def load_grouped_probes(
    raw_decisions: list[Any],
    manifest_path: Path,
    image_fixtures: dict[str, ProbeImageFixture] | None = None,
) -> list[Probe]:
    image_fixtures = image_fixtures or {}
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
                image_fixtures,
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
    image_fixtures: dict[str, ProbeImageFixture],
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
    display_prompt = _optional_string(raw_variant.get("display_prompt"))
    playground = _normalize_playground_policy(
        raw_variant.get("playground"), defaults.decision_id, variant_id
    )
    if not playground.enabled and display_prompt is None:
        raise ValueError(
            f"{probe_id} display_prompt is required when Playground is disabled"
        )
    generated_text = _normalize_generated_text(
        raw_variant.get("generated_text"),
        defaults.decision_id,
        variant_id,
        messages,
    )
    _validate_image_fixture_references(messages, image_fixtures, label)
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
        display_prompt=display_prompt,
        playground=playground,
        repeat=_normalize_repeat(
            raw_variant.get("repeat"), defaults.decision_id, variant_id
        ),
        padding=_normalize_padding(
            raw_variant.get("padding"), defaults.decision_id, variant_id
        ),
        generated_text=generated_text,
        messages=messages,
        tools=_normalize_objects(raw_variant.get("tools"), "tools"),
        expected_alias=defaults.expected_alias,
        notes=_optional_string(raw_variant.get("notes")) or defaults.notes,
        tags=_normalize_tags(raw_variant.get("tags")),
        image_fixtures=image_fixtures,
    )


def normalize_image_fixtures(
    raw_fixtures: Any, manifest_path: Path
) -> dict[str, ProbeImageFixture]:
    if raw_fixtures is None:
        return {}
    label = f"{manifest_path} fixtures"
    if not isinstance(raw_fixtures, dict):
        raise TypeError(f"{label} must be a mapping")
    reject_unknown_fields(raw_fixtures, FIXTURES_FIELDS, label)
    raw_images = raw_fixtures.get("images")
    if not isinstance(raw_images, dict) or not raw_images:
        raise ValueError(f"{label}.images must be a non-empty mapping")
    return {
        str(raw_name): _normalize_image_fixture(raw_name, raw_fixture, label)
        for raw_name, raw_fixture in raw_images.items()
    }


def _normalize_image_fixture(
    raw_name: Any,
    raw_fixture: Any,
    fixtures_label: str,
) -> ProbeImageFixture:
    name = str(raw_name)
    label = f"{fixtures_label}.images.{name}"
    if len(name) > MAX_IMAGE_FIXTURE_NAME_LENGTH or not FIXTURE_NAME_PATTERN.fullmatch(
        name
    ):
        raise ValueError(f"{label} has an invalid fixture name")
    if not isinstance(raw_fixture, dict):
        raise TypeError(f"{label} must be a mapping")
    reject_unknown_fields(raw_fixture, IMAGE_FIXTURE_FIELDS, label)
    description = str(raw_fixture.get("description") or "").strip()
    if not description or len(description) > MAX_IMAGE_FIXTURE_DESCRIPTION_LENGTH:
        raise ValueError(
            f"{label}.description must contain between 1 and "
            f"{MAX_IMAGE_FIXTURE_DESCRIPTION_LENGTH} characters"
        )
    media_type = str(raw_fixture.get("media_type") or "").strip().lower()
    if media_type not in IMAGE_FIXTURE_MEDIA_TYPES:
        supported = ", ".join(sorted(IMAGE_FIXTURE_MEDIA_TYPES))
        raise ValueError(f"{label}.media_type must be one of: {supported}")
    raw_data = raw_fixture.get("data_base64")
    if not isinstance(raw_data, str) or not raw_data.strip():
        raise ValueError(f"{label}.data_base64 is required")
    data_base64 = "".join(raw_data.split())
    try:
        decoded = base64.b64decode(data_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{label}.data_base64 is invalid") from exc
    if base64.b64encode(decoded).decode("ascii") != data_base64:
        raise ValueError(f"{label}.data_base64 must use canonical padding")
    if not decoded or len(decoded) > MAX_IMAGE_FIXTURE_BYTES:
        raise ValueError(
            f"{label}.data_base64 must decode to between 1 and "
            f"{MAX_IMAGE_FIXTURE_BYTES} bytes"
        )
    actual_sha256 = f"sha256:{hashlib.sha256(decoded).hexdigest()}"
    expected_sha256 = str(raw_fixture.get("sha256") or "").strip().lower()
    if expected_sha256 != actual_sha256:
        raise ValueError(
            f"{label}.sha256 mismatch: expected {expected_sha256!r}, "
            f"actual {actual_sha256!r}"
        )
    validate_image_fixture_payload(decoded, media_type, label)
    return ProbeImageFixture(
        name=name,
        description=description,
        media_type=media_type,
        data_base64=data_base64,
        sha256=actual_sha256,
        bytes=len(decoded),
    )


def _validate_image_fixture_references(
    messages: tuple[dict[str, Any], ...],
    fixtures: dict[str, ProbeImageFixture],
    variant_label: str,
) -> None:
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            label = (
                f"{variant_label}.messages[{message_index}].content[{content_index}]"
            )
            raw_type = item.get("type")
            item_type = str(raw_type or "").strip().lower()
            if item_type in {"image_url", "input_image"}:
                raise ValueError(
                    f"{label} must use a declared image_fixture instead of an "
                    "inline or remote image"
                )
            if item_type != "image_fixture":
                continue
            if raw_type != "image_fixture":
                raise ValueError(f"{label}.type must be exactly 'image_fixture'")
            reject_unknown_fields(item, frozenset({"type", "fixture"}), label)
            fixture_name = item.get("fixture")
            if not isinstance(fixture_name, str) or not fixture_name.strip():
                raise TypeError(f"{label}.fixture must be a non-empty string")
            fixture_name = fixture_name.strip()
            if fixture_name not in fixtures:
                raise ValueError(
                    f"{label}.fixture references unknown image fixture {fixture_name!r}"
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


def _normalize_playground_policy(
    raw_policy: Any, decision_id: str, variant_id: str
) -> ProbePlaygroundPolicy:
    if raw_policy is None:
        return ProbePlaygroundPolicy(enabled=True)
    label = f"{decision_id}:{variant_id} playground"
    if not isinstance(raw_policy, dict):
        raise TypeError(f"{label} must be a mapping")
    reject_unknown_fields(raw_policy, PLAYGROUND_FIELDS, label)
    enabled = raw_policy.get("enabled")
    if not isinstance(enabled, bool):
        raise TypeError(f"{label}.enabled must be a boolean")
    reason = _optional_string(raw_policy.get("reason"))
    if not enabled and reason is None:
        raise ValueError(f"{label}.reason is required when disabled")
    return ProbePlaygroundPolicy(enabled=enabled, reason=reason)


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


def _normalize_generated_text(
    raw_generated_text: Any,
    decision_id: str,
    variant_id: str,
    messages: tuple[dict[str, Any], ...],
) -> ProbeGeneratedText | None:
    if raw_generated_text is None:
        return None
    label = f"{decision_id}:{variant_id} generated_text"
    if not isinstance(raw_generated_text, dict):
        raise TypeError(f"{label} must be a mapping")
    reject_unknown_fields(raw_generated_text, GENERATED_TEXT_FIELDS, label)
    if not messages:
        raise ValueError(f"{label} requires a messages probe")

    message_index = _required_non_negative_int(
        raw_generated_text.get("message_index"), f"{label}.message_index"
    )
    content_index = _required_non_negative_int(
        raw_generated_text.get("content_index"), f"{label}.content_index"
    )
    target_text_bytes = _required_positive_int(
        raw_generated_text.get("target_text_bytes"),
        f"{label}.target_text_bytes",
    )
    if target_text_bytes > MAX_GENERATED_TEXT_BYTES:
        raise ValueError(
            f"{label}.target_text_bytes must not exceed {MAX_GENERATED_TEXT_BYTES}"
        )
    character = raw_generated_text.get("character", "x")
    if (
        not isinstance(character, str)
        or len(character) != 1
        or not character.isascii()
        or not character.isprintable()
    ):
        raise ValueError(f"{label}.character must be one printable ASCII character")
    if message_index >= len(messages):
        raise ValueError(
            f"{label}.message_index must reference one of {len(messages)} messages"
        )
    content = messages[message_index].get("content")
    if not isinstance(content, list):
        raise ValueError(f"{label} requires the selected message content to be a list")
    if content_index > len(content):
        raise ValueError(f"{label}.content_index must be between 0 and {len(content)}")
    existing_text_bytes = message_content_text_bytes(content)
    if target_text_bytes <= existing_text_bytes:
        raise ValueError(
            f"{label}.target_text_bytes must exceed the selected message's "
            f"{existing_text_bytes} explicit text bytes"
        )
    return ProbeGeneratedText(
        message_index=message_index,
        content_index=content_index,
        target_text_bytes=target_text_bytes,
        character=character,
    )


def _required_non_negative_int(value: Any, label: str) -> int:
    parsed = _required_int(value, label)
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _required_positive_int(value: Any, label: str) -> int:
    parsed = _required_int(value, label)
    if parsed < 1:
        raise ValueError(f"{label} must be positive")
    return parsed


def _required_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def message_content_text_bytes(content: list[Any]) -> int:
    """Count UTF-8 bytes in text-bearing message content parts."""
    total = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        raw_kind = item.get("type")
        if raw_kind is not None and not isinstance(raw_kind, str):
            continue
        kind = str(raw_kind or "").strip().lower()
        if kind not in ("", "text", "input_text", "output_text"):
            continue
        text = item.get("text")
        if isinstance(text, str):
            total += len(text.encode("utf-8"))
    return total


def _normalize_objects(raw_items: Any, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_items, list):
        return ()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise TypeError(f"{label}[{index}] must be a mapping")
        normalized.append(item)
    return tuple(normalized)
