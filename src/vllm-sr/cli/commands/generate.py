"""Generate command implementation."""

import sys
from pathlib import Path

from cli.commands.serve import (
    generate_router_config,
    copy_defaults_reference,
    ensure_output_directory,
    DEFAULT_OUTPUT_DIR,
)
from cli.parser import parse_user_config, ConfigParseError
from cli.config_generator import generate_envoy_config_from_user_config
from cli.utils import getLogger

log = getLogger(__name__)


def generate_command(
    config_path: str, output_dir: str = DEFAULT_OUTPUT_DIR, force: bool = False
):
    """
    Generate configurations without starting services.

    Args:
        config_path: Path to user config.yaml
        output_dir: Output directory for generated configs
        force: Force overwrite existing files
    """
    log.info("=" * 60)
    log.info("vLLM Semantic Router - Generate Configurations")
    log.info("=" * 60)

    # Ensure output directory exists
    output_path = ensure_output_directory(output_dir)

    # Parse user config (needed for Envoy generation)
    try:
        user_config = parse_user_config(config_path)
    except ConfigParseError as e:
        log.error(f"Failed to parse configuration: {e}")
        sys.exit(1)

    # Generate router config
    router_config_path = generate_router_config(config_path, output_dir, force=force)

    # Copy defaults for reference
    defaults_path = copy_defaults_reference(output_dir)

    # Generate Envoy config
    envoy_config_path = None
    try:
        envoy_output = output_path / "envoy-config.yaml"
        envoy_config_path = generate_envoy_config_from_user_config(
            user_config, str(envoy_output)
        )
    except Exception as e:
        log.warning(f"Failed to generate Envoy config: {e}")
        log.warning("Continuing without Envoy config...")

    log.info("=" * 60)
    log.info("✓ Configurations generated successfully")
    log.info(f"  Output directory: {output_path.absolute()}")
    log.info(f"  Router config: {router_config_path.name}")
    if envoy_config_path:
        log.info(f"  Envoy config: {envoy_config_path.name}")
    log.info(f"  Defaults reference: {defaults_path.name}")
    log.info("=" * 60)
    log.info("\nYou can now:")
    log.info(f"  • Edit {router_config_path} to customize system settings")
    if envoy_config_path:
        log.info(f"  • Edit {envoy_config_path} to customize Envoy settings")
    log.info(f"  • Start service with: vllm-sr serve {config_path}")
