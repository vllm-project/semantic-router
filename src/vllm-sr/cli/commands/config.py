"""Config command implementation."""

from __future__ import annotations

import sys
import tempfile
from importlib import import_module
from pathlib import Path

from cli.config_generator import generate_envoy_config_from_user_config
from cli.config_import import import_config_command as run_import_config_command
from cli.parser import ConfigParseError, parse_user_config
from cli.terminal import echo
from cli.utils import get_logger
from cli.validator import (
    print_validation_errors,
    validate_user_config,
)

log = get_logger(__name__)


def config_command(config_type: str, config_path: str = "config.yaml"):
    """
    Print generated configuration.

    Args:
        config_type: Type of config to print ('envoy' or 'router')
        config_path: Path to user config.yaml (default: config.yaml)
    """
    if config_type not in ["envoy", "router"]:
        log.error(f"Invalid config type: {config_type}")
        log.error("Must be 'envoy' or 'router'")
        sys.exit(1)

    # Check if config file exists
    if not Path(config_path).exists():
        log.error(f"Config file not found: {config_path}")
        log.error("Run 'vllm-sr serve' to create a local Management workspace")
        log.error(
            "Or write a canonical v0.3 config.yaml using the documentation examples"
        )
        sys.exit(1)

    # Parse user config
    try:
        user_config = parse_user_config(config_path, log_summary=False)
    except ConfigParseError as e:
        log.error(f"Failed to parse configuration: {e}")
        sys.exit(1)

    # Validate user config
    errors = validate_user_config(user_config, log_summary=False)
    if errors:
        print_validation_errors(errors)
        sys.exit(1)

    if config_type == "router":
        # Router now reads canonical config.yaml directly.
        echo(Path(config_path).read_text(), nl=False)

    elif config_type == "envoy":
        # Generate envoy config
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                temp_path = f.name

            generate_envoy_config_from_user_config(
                user_config,
                temp_path,
                log_summary=False,
            )

            # Read and print
            with open(temp_path) as f:
                echo(f.read(), nl=False)

            # Clean up
            Path(temp_path).unlink()

        except Exception as e:
            log.error(f"Failed to generate Envoy config: {e}")
            sys.exit(1)


def import_config_from_source_command(
    from_type: str,
    source_path: str | None = None,
    target_path: str = "config.yaml",
    force: bool = False,
):
    """Import a supported external config source into canonical v0.3 YAML."""

    return run_import_config_command(
        from_type=from_type,
        source_path=source_path,
        target_path=target_path,
        force=force,
    )


def migrate_config_command(
    config_path: str = "config.yaml",
    output_path: str | None = None,
    force: bool = False,
):
    """Rewrite one previous-release v0.3 file into strict current v0.3."""

    # The previous-release decoder is an offline conversion boundary. Keep it
    # out of every runtime command's import graph so serve/validate remain one
    # strict current-v0.3 reader.
    migration_module = import_module("cli.config_migrate_command")

    return migration_module.migrate_config_command(
        config_path=config_path,
        output_path=output_path,
        force=force,
    )
