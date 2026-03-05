"""Serve command implementation."""

import os
import sys
import yaml
from pathlib import Path

from cli.parser import parse_user_config, ConfigParseError
from cli.defaults import load_embedded_defaults, get_defaults_yaml, load_defaults
from cli.merger import merge_configs
from cli.validator import (
    validate_user_config,
    validate_merged_config,
    print_validation_errors,
)
from cli.config_generator import generate_envoy_config_from_user_config
from cli.utils import getLogger

log = getLogger(__name__)

DEFAULT_OUTPUT_DIR = ".vllm-sr"


def _normalize_platform(value: str) -> str:
    """Normalize platform value for comparisons."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _set_use_cpu_false_for_amd(config_node, path: str, changed_paths: list):
    """Recursively set use_cpu=true to false for AMD GPU defaults."""
    if isinstance(config_node, dict):
        for key, value in config_node.items():
            current_path = f"{path}.{key}" if path else key
            if key == "use_cpu" and value is True:
                config_node[key] = False
                changed_paths.append(current_path)
            else:
                _set_use_cpu_false_for_amd(value, current_path, changed_paths)
        return

    if isinstance(config_node, list):
        for index, item in enumerate(config_node):
            _set_use_cpu_false_for_amd(item, f"{path}[{index}]", changed_paths)


def apply_platform_gpu_defaults(merged_config: dict) -> None:
    """
    Apply platform-specific GPU defaults.

    For AMD platform, default all `use_cpu` flags to false so inference prefers GPU.
    Can be disabled by setting VLLM_SR_AMD_FORCE_GPU=0/false/no/off.
    """
    platform = _normalize_platform(
        os.getenv("VLLM_SR_PLATFORM") or os.getenv("DASHBOARD_PLATFORM")
    )
    if platform != "amd":
        return

    force_gpu = os.getenv("VLLM_SR_AMD_FORCE_GPU", "1").strip().lower()
    if force_gpu in {"0", "false", "no", "off"}:
        log.info(
            "Platform amd detected but GPU default override disabled by VLLM_SR_AMD_FORCE_GPU"
        )
        return

    changed_paths = []
    _set_use_cpu_false_for_amd(merged_config, "", changed_paths)
    if not changed_paths:
        log.info("Platform amd detected: no use_cpu flags found to override")
        return

    preview = ", ".join(changed_paths[:8])
    if len(changed_paths) > 8:
        preview = f"{preview}, ..."
    log.info(
        f"Platform amd detected: set {len(changed_paths)} use_cpu flag(s) to false for GPU default ({preview})"
    )


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
    Generate router configuration from user config.

    Args:
        config_path: Path to user config.yaml
        output_dir: Output directory
        force: Force regeneration even if file exists

    Returns:
        Path: Path to generated router config
    """
    output_path = ensure_output_directory(output_dir)
    router_config_path = output_path / "router-config.yaml"

    # Check if router config already exists
    if router_config_path.exists() and not force:
        log.info(f"Using existing {router_config_path}")
        log.info(f"  (Use --regenerate to recreate from {config_path})")
        return router_config_path

    log.info(f"Generating {router_config_path}...")

    # Parse user config
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

    # Load defaults (prefer local router-defaults.yaml if it exists)
    defaults = load_defaults(output_dir)

    # Log which defaults were used
    local_defaults_path = Path(output_dir) / "router-defaults.yaml"
    if local_defaults_path.exists():
        log.info(f"  Using local defaults: {local_defaults_path}")
    else:
        log.info(f"  Using embedded defaults")

    # Merge configs
    merged = merge_configs(user_config, defaults)
    apply_platform_gpu_defaults(merged)

    # Validate merged config
    errors = validate_merged_config(merged)
    if errors:
        print_validation_errors(errors)
        sys.exit(1)

    # Write router config
    with open(router_config_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    log.info(f"Generated {router_config_path}")

    return router_config_path


def copy_defaults_reference(output_dir: str) -> Path:
    """
    Copy embedded defaults to output directory for reference.

    Will NOT overwrite existing router-defaults.yaml to preserve user modifications.

    Args:
        output_dir: Output directory

    Returns:
        Path: Path to defaults reference file
    """
    output_path = Path(output_dir)
    defaults_path = output_path / "router-defaults.yaml"

    if defaults_path.exists():
        log.info(f"Using existing {defaults_path} (preserving user modifications)")
        return defaults_path

    log.info(f"Copying router-defaults.yaml (for reference)...")

    with open(defaults_path, "w") as f:
        f.write(get_defaults_yaml())

    log.info(f"Copied {defaults_path}")

    return defaults_path


def serve_command(
    config_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    regenerate: bool = False,
    router_config: str = None,
    envoy_config: str = None,
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

    # Generate or use existing router config
    if router_config:
        log.info(f"Using custom router config: {router_config}")
        router_config_path = Path(router_config)
    else:
        router_config_path = generate_router_config(
            config_path, output_dir, force=regenerate
        )

    # Copy defaults for reference
    copy_defaults_reference(output_dir)

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
