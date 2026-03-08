"""Dashboard-facing canonical config helpers for the TD001 migration."""

from __future__ import annotations

from typing import Any

import yaml

from cli.compat_blocks import dump_typed_compat_block, get_typed_compat_blocks
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
