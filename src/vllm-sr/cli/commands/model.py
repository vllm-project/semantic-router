"""Model command implementations."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from cli.parser import ConfigParseError, parse_user_config
from cli.utils import get_logger

log = get_logger(__name__)

_SENSITIVE_QUERY_KEY_FRAGMENTS = ("key", "token", "secret", "password")


def model_list_command(config_path: str = "config.yaml") -> None:
    """Print the provider models and routing model cards from a config file.

    Args:
        config_path: Path to user config.yaml (default: config.yaml)
    """
    if not Path(config_path).exists():
        log.error(f"Config file not found: {config_path}")
        log.error(
            "Run 'vllm-sr serve' to bootstrap setup mode and create a config file"
        )
        sys.exit(1)

    try:
        user_config = parse_user_config(config_path)
    except ConfigParseError as e:
        log.error(f"Failed to parse configuration: {e}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("vLLM Semantic Router - Models")
    log.info("=" * 60)
    log.info(f"Config: {config_path}")
    log.info("")

    _print_provider_models(user_config)
    log.info("")
    _print_model_cards(user_config)


def _print_provider_models(user_config) -> None:
    providers = getattr(user_config, "providers", None)
    models = list(getattr(providers, "models", []) or []) if providers else []

    log.info(f"Provider models ({len(models)}):")
    if not models:
        log.info("  (none configured)")
        return

    default_model = None
    if providers and providers.defaults:
        default_model = providers.defaults.default_model

    for model in models:
        is_default = " [default]" if model.name == default_model else ""
        log.info(f"  - {model.name}{is_default}")
        if model.provider_model_id and model.provider_model_id != model.name:
            log.info(f"      provider_model_id: {model.provider_model_id}")
        if model.reasoning_family:
            log.info(f"      reasoning_family:  {model.reasoning_family}")
        if model.api_format:
            log.info(f"      api_format:        {model.api_format}")

        backends = list(model.backend_refs or [])
        if backends:
            log.info(f"      backends ({len(backends)}):")
            for ref in backends:
                # Never print api_key / api_key_env values; expose only the
                # transport identity so credentials cannot leak via CLI output.
                provider = ref.provider or "-"
                base_url = _redact_url(ref.base_url or ref.endpoint or "-")
                label = ref.name or "(unnamed)"
                log.info(
                    f"        * {label}  provider={provider}  base_url={base_url}  "
                    f"protocol={ref.protocol}  weight={ref.weight}"
                )


def _print_model_cards(user_config) -> None:
    routing = getattr(user_config, "routing", None)
    cards = list(getattr(routing, "model_cards", []) or []) if routing else []

    log.info(f"Model cards ({len(cards)}):")
    if not cards:
        log.info("  (none configured)")
        return

    for card in cards:
        log.info(f"  - {card.name}")
        if card.modality:
            log.info(f"      modality:        {card.modality}")
        if card.param_size:
            log.info(f"      param_size:      {card.param_size}")
        if card.context_window_size:
            log.info(f"      context_window:  {card.context_window_size}")
        if card.capabilities:
            log.info(f"      capabilities:    {', '.join(card.capabilities)}")
        if card.tags:
            log.info(f"      tags:            {', '.join(card.tags)}")
        if card.loras:
            lora_names = ", ".join(a.name for a in card.loras)
            log.info(f"      loras:           {lora_names}")


def _redact_url(value: str) -> str:
    """Return a display-safe URL without embedded credentials."""
    if value == "-":
        return value

    try:
        parsed = urlsplit(value)
    except ValueError:
        return value

    netloc = parsed.netloc
    if "@" in netloc:
        netloc = "***@" + netloc.rsplit("@", 1)[1]

    query_items = []
    for key, item_value in parse_qsl(parsed.query, keep_blank_values=True):
        if any(fragment in key.lower() for fragment in _SENSITIVE_QUERY_KEY_FRAGMENTS):
            query_items.append((key, "***"))
        else:
            query_items.append((key, item_value))

    redacted_query = urlencode(query_items, doseq=True, safe="*")
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, redacted_query, parsed.fragment)
    )
