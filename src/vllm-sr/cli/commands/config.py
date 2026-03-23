"""Config command implementation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

from cli.config_generator import generate_envoy_config_from_user_config
from cli.config_import import import_config_command as run_import_config_command
from cli.config_migration import migrate_config_data
from cli.parser import ConfigParseError, load_config_file, parse_user_config
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
        log.error(
            "Run 'vllm-sr serve' to bootstrap setup mode and create a config file"
        )
        log.error(
            "Or write a canonical v0.3 config.yaml using the docs examples if you want to hand-author it directly"
        )
        sys.exit(1)

    # Parse user config
    try:
        user_config = parse_user_config(config_path)
    except ConfigParseError as e:
        log.error(f"Failed to parse configuration: {e}")
        sys.exit(1)

    # Validate user config
    errors = validate_user_config(user_config)
    if errors:
        log.error("Configuration validation failed:")
        print_validation_errors(errors)
        sys.exit(1)

    if config_type == "router":
        # Router now reads canonical config.yaml directly.
        print(Path(config_path).read_text())

    elif config_type == "envoy":
        # Generate envoy config
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                temp_path = f.name

            generate_envoy_config_from_user_config(user_config, temp_path)

            # Read and print
            with open(temp_path) as f:
                print(f.read())

            # Clean up
            Path(temp_path).unlink()

        except Exception as e:
            log.error(f"Failed to generate Envoy config: {e}")
            sys.exit(1)


def migrate_config_command(
    config_path: str = "config.yaml",
    output_path: str | None = None,
    force: bool = False,
) -> Path:
    """Migrate a legacy or mixed config file to canonical v0.3 YAML."""

    source_path = Path(config_path)
    if not source_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        data = load_config_file(config_path)
    except ConfigParseError as e:
        log.error(f"Failed to read configuration: {e}")
        sys.exit(1)

    migrated = migrate_config_data(data)

    destination = (
        Path(output_path)
        if output_path
        else source_path.with_name(f"{source_path.stem}.migrated{source_path.suffix}")
    )
    if destination.exists() and not force:
        log.error(f"Output file already exists: {destination}")
        log.error("Use --force to overwrite the destination")
        sys.exit(1)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(migrated, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    log.info("Migrated configuration written successfully")
    log.info(f"  Source: {source_path}")
    log.info(f"  Output: {destination}")
    return destination


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
