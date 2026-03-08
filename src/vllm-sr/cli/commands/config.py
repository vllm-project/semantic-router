"""Config command implementation."""

import sys
import tempfile
from pathlib import Path

import yaml

from cli.authoring_projection import build_first_slice_authoring_config
from cli.config_generator import generate_envoy_config_from_user_config
from cli.defaults import load_defaults
from cli.merger import merge_configs
from cli.parser import ConfigParseError, parse_user_config
from cli.utils import getLogger
from cli.validator import (
    print_validation_errors,
    validate_merged_config,
    validate_user_config,
)

log = getLogger(__name__)


def config_command(config_type: str, config_path: str = "config.yaml"):
    """
    Print generated configuration.

    Args:
        config_type: Type of config to print ('envoy', 'router', or 'authoring')
        config_path: Path to user config.yaml (default: config.yaml)
    """
    if config_type not in ["envoy", "router", "authoring"]:
        log.error(f"Invalid config type: {config_type}")
        log.error("Must be 'envoy', 'router', or 'authoring'")
        sys.exit(1)

    # Check if config file exists
    if not Path(config_path).exists():
        log.error(f"Config file not found: {config_path}")
        log.error(
            "Run 'vllm-sr serve' to bootstrap setup mode and create a config file"
        )
        log.error(
            "Or run 'vllm-sr init' if you want an advanced YAML sample to edit directly"
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

    if config_type == "authoring":
        authoring_config = build_first_slice_authoring_config(user_config)
        print(yaml.dump(authoring_config, default_flow_style=False, sort_keys=False))

    elif config_type == "router":
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
