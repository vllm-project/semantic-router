"""Validate command implementation."""

import sys

from cli.parser import ConfigParseError, parse_user_config
from cli.utils import get_logger
from cli.validator import print_validation_errors, validate_user_config

log = get_logger(__name__)


def validate_command(config_path: str):
    """
    Validate user configuration.

    Args:
        config_path: Path to user config.yaml
    """
    log.info("=" * 60)
    log.info("vLLM Semantic Router - Validate Configuration")
    log.info("=" * 60)
    log.info(f"Validating: {config_path}")
    log.info("")

    # Parse config
    try:
        user_config = parse_user_config(config_path)
    except ConfigParseError as e:
        log.error("\n❌ Configuration parsing failed:")
        log.error(f"{e}")
        sys.exit(1)

    # Validate config
    errors = validate_user_config(user_config)

    if errors:
        print_validation_errors(errors)
        sys.exit(1)

    log.info("=" * 60)
    log.info("Configuration is valid!")
    log.info("=" * 60)
    log.info("\nConfiguration summary:")
    log.info(f"  Version: {user_config.version}")
    log.info(f"  Listeners: {len(user_config.listeners)}")

    if user_config.signals:
        log.info(f"  Keyword signals: {len(user_config.signals.keywords)}")
        log.info(f"  Embedding signals: {len(user_config.signals.embeddings)}")
        if user_config.signals.fact_check:
            log.info(f"  Fact check signals: {len(user_config.signals.fact_check)}")
        log.info(f"  Domains: {len(user_config.signals.domains)}")
        if user_config.signals.structure:
            log.info(f"  Structure signals: {len(user_config.signals.structure)}")
    else:
        log.info(
            "  Signals: None (catch-all routing is supported; domain categories will auto-generate when needed)"
        )

    log.info(f"  Decisions: {len(user_config.decisions)}")

    # Count plugins
    total_plugins = 0
    plugins_by_type = {}
    decisions_with_plugins = 0
    for decision in user_config.decisions:
        if decision.plugins:
            decisions_with_plugins += 1
            for plugin in decision.plugins:
                total_plugins += 1
                plugin_type = (
                    plugin.type.value
                    if hasattr(plugin.type, "value")
                    else str(plugin.type)
                )
                plugins_by_type[plugin_type] = plugins_by_type.get(plugin_type, 0) + 1

    if total_plugins > 0:
        log.info(
            f"  Plugins: {total_plugins} total ({decisions_with_plugins} decisions)"
        )
        if len(plugins_by_type) > 0:
            plugin_types_str = ", ".join(
                f"{ptype}: {count}" for ptype, count in sorted(plugins_by_type.items())
            )
            log.info(f"    Types: {plugin_types_str}")

    log.info(f"  Models: {len(user_config.providers.models)}")
    log.info(f"  Default model: {user_config.providers.default_model}")
    log.info("")
