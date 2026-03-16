"""Serve command implementation."""

import sys
from pathlib import Path

from cli.config_generator import generate_envoy_config_from_user_config
from cli.parser import ConfigParseError, parse_user_config
from cli.utils import get_logger
from cli.validator import (
    print_validation_errors,
    validate_user_config,
)

log = get_logger(__name__)

DEFAULT_OUTPUT_DIR = ".vllm-sr"


def ensure_output_directory(output_dir: str) -> Path:
    """
    Ensure output directory exists.

    Args:
        output_dir: Output directory path

    Returns:
        Path: Output directory path object
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def generate_router_config(
    config_path: str, output_dir: str = DEFAULT_OUTPUT_DIR, force: bool = False
) -> Path:
    """
    Resolve the router configuration path from canonical config.

    Args:
        config_path: Path to user config.yaml
        output_dir: Output directory
        force: Unused compatibility flag

    Returns:
        Path: Path to canonical router config
    """
    try:
        user_config = parse_user_config(config_path)
    except ConfigParseError as e:
        log.error(f"Failed to parse configuration: {e}")
        sys.exit(1)

    # Validate user config
    errors = validate_user_config(user_config)
    if errors:
        print_validation_errors(errors)
        sys.exit(1)

    log.info(
        "Router reads canonical config directly; skipping router-config.yaml generation"
    )
    return Path(config_path)


def write_global_defaults_reference(output_dir: str) -> Path:
    """
    Write a read-only global defaults reference for inspection.

    Args:
        output_dir: Output directory

    Returns:
        Path: Path to defaults reference file
    """
    output_path = Path(output_dir)
    defaults_path = output_path / "global-defaults.yaml"
    if defaults_path.exists():
        return defaults_path
    defaults_path.write_text(
        "# Router-owned global defaults are now built into the Go router.\n"
        "# Use the dashboard or `vllm-sr config router` to inspect effective config.\n"
    )

    return defaults_path


def serve_command(
    config_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    regenerate: bool = False,
    router_config: str | None = None,
    envoy_config: str | None = None,
):
    """
    Start vLLM Semantic Router service.

    Args:
        config_path: Path to user config.yaml
        output_dir: Output directory for generated configs
        regenerate: Force regenerate router config
        router_config: Custom router config path (bypasses generation)
        envoy_config: Custom Envoy config path (bypasses generation)
    """
    log.info("=" * 60)
    log.info("vLLM Semantic Router - Starting Service")
    log.info("=" * 60)

    # Ensure output directory exists
    output_path = ensure_output_directory(output_dir)

    # Parse user config (needed for Envoy generation)
    try:
        user_config = parse_user_config(config_path)
    except ConfigParseError as e:
        log.error(f"Failed to parse configuration: {e}")
        sys.exit(1)

    # Resolve or use router config
    if router_config:
        log.info(f"Using custom router config: {router_config}")
        router_config_path = Path(router_config)
    else:
        router_config_path = generate_router_config(
            config_path, output_dir, force=regenerate
        )

    # Write defaults note for reference
    write_global_defaults_reference(output_dir)

    # Generate Envoy config
    envoy_config_path = None
    if envoy_config:
        log.info(f"Using custom Envoy config: {envoy_config}")
        envoy_config_path = Path(envoy_config)
    else:
        try:
            envoy_output = output_path / "envoy-config.yaml"
            envoy_config_path = generate_envoy_config_from_user_config(
                user_config, str(envoy_output)
            )
        except Exception as e:
            log.warning(f"Failed to generate Envoy config: {e}")
            log.warning("Continuing without Envoy config...")

    # TODO: Start services

    log.info("=" * 60)
    log.info("Configuration generated successfully")
    log.info(f"  Router config: {router_config_path}")
    if envoy_config_path:
        log.info(f"  Envoy config: {envoy_config_path}")
    log.info(f"  Output directory: {output_dir}")
    log.info("=" * 60)

    # For now, just show what would happen
    log.info("\nNext steps (to be implemented):")
    log.info("  1. Start Envoy proxy")
    log.info("  2. Start Router service")
    log.info("  3. Wait for health checks")
