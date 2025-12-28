"""Validate command implementation."""

import sys

from cli.parser import parse_user_config, ConfigParseError
from cli.validator import validate_user_config, print_validation_errors
from cli.utils import getLogger

log = getLogger(__name__)


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
        log.error(f"\n❌ Configuration parsing failed:")
        log.error(f"{e}")
        sys.exit(1)

    # Validate config
    errors = validate_user_config(user_config)

    if errors:
        print_validation_errors(errors)
        sys.exit(1)

    log.info("=" * 60)
    log.info("✓ Configuration is valid!")
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
    else:
        log.info(f"  Signals: None (will auto-generate categories)")

    log.info(f"  Decisions: {len(user_config.decisions)}")
    log.info(f"  Models: {len(user_config.providers.models)}")
    log.info(f"  Default model: {user_config.providers.default_model}")
    log.info("")
