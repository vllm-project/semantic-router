"""Config command implementation."""

import sys
import yaml
from pathlib import Path

from cli.parser import parse_user_config, ConfigParseError
from cli.defaults import load_embedded_defaults, load_defaults
from cli.merger import merge_configs
from cli.validator import (
    validate_user_config,
    validate_merged_config,
    print_validation_errors,
)
from cli.config_generator import generate_envoy_config_from_user_config
from cli.utils import getLogger

log = getLogger(__name__)


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
        log.error("Run 'vllm-sr init' to create a config file")
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
        # Generate router config (use local defaults if available)
        defaults = load_defaults(".vllm-sr")
        merged = merge_configs(user_config, defaults)

        # Validate merged config
        errors = validate_merged_config(merged)
        if errors:
            log.error("Merged configuration validation failed:")
            print_validation_errors(errors)
            sys.exit(1)

        # Print router config as YAML
        print(yaml.dump(merged, default_flow_style=False, sort_keys=False))

    elif config_type == "envoy":
        # Generate envoy config
        try:
            # Generate to a temporary string
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                temp_path = f.name

            generate_envoy_config_from_user_config(user_config, temp_path)

            # Read and print
            with open(temp_path, "r") as f:
                print(f.read())

            # Clean up
            Path(temp_path).unlink()

        except Exception as e:
            log.error(f"Failed to generate Envoy config: {e}")
            sys.exit(1)
