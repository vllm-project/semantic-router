"""Dashboard-facing canonical config helpers for the TD001 migration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

from cli.compat_blocks import dump_typed_compat_block, get_typed_compat_blocks
from cli.models import UserConfig
from cli.parser import parse_user_config, parse_user_config_data
from cli.validator import validate_user_config

_NAMED_TYPED_COMPAT_KEYS = (
    "api",
    "authz",
    "bert_model",
    "classifier",
    "feedback_detector",
    "hallucination_mitigation",
    "image_gen_backends",
    "looper",
    "modality_detector",
    "model_selection",
    "observability",
    "prompt_guard",
    "provider_profiles",
    "ratelimit",
    "response_api",
    "router_replay",
    "semantic_cache",
    "tools",
    "vector_store",
)
_ROUTER_OPTION_KEYS = (
    "auto_model_name",
    "include_config_models_in_list",
    "clear_route_cache",
    "streamed_body_mode",
    "max_streamed_body_bytes",
    "streamed_body_timeout_sec",
)
_RUNTIME_TOP_LEVEL_KEYS = (
    "config_source",
    "mom_registry",
    "strategy",
)
_DASHBOARD_CANONICAL_TOP_LEVEL_KEYS = frozenset(UserConfig.model_fields).union(
    _NAMED_TYPED_COMPAT_KEYS,
    _ROUTER_OPTION_KEYS,
    _RUNTIME_TOP_LEVEL_KEYS,
)


def load_dashboard_config(config_path: str) -> dict[str, Any]:
    """Load a config file into the dashboard canonical/nested shape."""

    user_config = parse_user_config(config_path)
    return dump_user_config_with_compat(user_config)


def render_dashboard_yaml(config_data: dict[str, Any]) -> str:
    """Validate dashboard config data through the CLI schema and serialize YAML."""

    user_config = parse_user_config_data(config_data)
    validation_errors = validate_user_config(user_config)
    if validation_errors:
        formatted = "\n".join(f"  • {error}" for error in validation_errors)
        raise ValueError(f"Configuration validation failed:\n{formatted}")

    normalized = dump_user_config_with_compat(user_config)
    return yaml.dump(normalized, default_flow_style=False, sort_keys=False)


def render_merged_dashboard_yaml(config_path: str, config_patch: dict[str, Any]) -> str:
    """Apply a canonical dashboard patch to an existing config and serialize YAML."""

    merged = merge_dashboard_config_patch(
        load_dashboard_config(config_path), config_patch
    )
    return render_dashboard_yaml(merged)


def merge_dashboard_config_patch(
    existing_config: dict[str, Any], config_patch: dict[str, Any]
) -> dict[str, Any]:
    """Merge a partial canonical dashboard patch into an existing canonical config."""

    if not config_patch or not isinstance(config_patch, dict):
        raise ValueError("Dashboard config patch must be a non-empty mapping")

    unsupported_keys = sorted(set(config_patch) - _DASHBOARD_CANONICAL_TOP_LEVEL_KEYS)
    if unsupported_keys:
        raise ValueError(
            "Unsupported canonical dashboard patch keys: " + ", ".join(unsupported_keys)
        )

    merged = deepcopy(existing_config)
    for key, value in config_patch.items():
        merged[key] = _merge_dashboard_patch_value(merged.get(key), value)
    return merged


def _merge_dashboard_patch_value(existing_value: Any, patch_value: Any) -> Any:
    """Merge dashboard patch values with map recursion and replace-on-write lists/scalars."""

    if patch_value is None:
        return None

    if isinstance(existing_value, dict) and isinstance(patch_value, dict):
        merged = deepcopy(existing_value)
        for key, value in patch_value.items():
            merged[key] = _merge_dashboard_patch_value(existing_value.get(key), value)
        return merged

    return deepcopy(patch_value)


def dump_user_config_with_compat(user_config: Any) -> dict[str, Any]:
    """Serialize UserConfig together with typed compatibility blocks."""

    data = user_config.model_dump(mode="python", exclude_none=True, by_alias=True)
    compat_blocks = get_typed_compat_blocks(user_config)

    for key in _NAMED_TYPED_COMPAT_KEYS:
        block = getattr(compat_blocks, key)
        if block is not None:
            data[key] = dump_typed_compat_block(block)

    if compat_blocks.router_options is not None:
        data.update(dump_typed_compat_block(compat_blocks.router_options))

    if compat_blocks.runtime_top_level is not None:
        data.update(dump_typed_compat_block(compat_blocks.runtime_top_level))

    return data
