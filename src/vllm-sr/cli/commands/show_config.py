"""Show config command implementation."""

import sys
from pathlib import Path

import yaml

from cli.parser import ConfigParseError, parse_user_config
from cli.utils import getLogger
from cli.validator import print_validation_errors, validate_user_config

log = getLogger(__name__)


def print_section_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_yaml_section(data: dict, max_lines: int | None = None):
    """Print YAML data with optional line limit."""
    yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
    lines = yaml_str.split("\n")

    if max_lines and len(lines) > max_lines:
        print("\n".join(lines[:max_lines]))
        print(f"\n... ({len(lines) - max_lines} more lines)")
    else:
        print(yaml_str)


def show_config_command(
    config_path: str,
    output_dir: str = ".vllm-sr",
    show_user: bool = True,
    show_router: bool = True,
    show_envoy: bool = True,
    full: bool = False,
):
    """
    Show all configurations (user, router, envoy).

    Args:
        config_path: Path to user config.yaml
        output_dir: Output directory for generated configs
        show_user: Show user configuration
        show_router: Show router configuration
        show_envoy: Show Envoy configuration
        full: Show full configuration without truncation
    """
    log.info("=" * 80)
    log.info("vLLM Semantic Router - Show Configurations")
    log.info("=" * 80)
    log.info(f"Config file: {config_path}")
    log.info(f"Output directory: {output_dir}")
    log.info("")

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

    # Show user configuration
    if show_user:
        print_section_header("USER CONFIGURATION (config.yaml)")
        user_dict = user_config.model_dump()
        if full:
            print_yaml_section(user_dict)
        else:
            print_yaml_section(user_dict, max_lines=50)

    # Generate router configuration
    if show_router:
        print_section_header("ROUTER CONFIGURATION (config.yaml)")
        router_config_path = Path(config_path)
        log.info("Router now reads canonical config directly")
        router_config = user_config.model_dump(by_alias=True, exclude_none=True)

        if full:
            print_yaml_section(router_config)
        else:
            print_yaml_section(router_config, max_lines=50)

    # Show Envoy configuration
    if show_envoy:
        print_section_header("ENVOY CONFIGURATION (envoy-config.yaml)")

        envoy_config_path = Path(output_dir) / "envoy-config.yaml"
        if envoy_config_path.exists():
            log.info(f"Loading existing: {envoy_config_path}")
            with open(envoy_config_path) as f:
                envoy_config = yaml.safe_load(f)

            if full:
                print_yaml_section(envoy_config)
            else:
                print_yaml_section(envoy_config, max_lines=50)
        else:
            print("\n⚠️  Envoy configuration not generated yet")
            print("   Run 'vllm-sr generate' or 'vllm-sr serve' to generate it")

    # Summary
    print("\n" + "=" * 80)
    log.info("Configuration Summary")
    print("=" * 80)

    if show_user:
        print(f"User config: {config_path}")

    if show_router:
        print(f"Router config: {router_config_path}")

    if show_envoy:
        if envoy_config_path.exists():
            print(f"Envoy config: {envoy_config_path}")
        else:
            print("⚠ Envoy config: Not generated yet")

    print("\nOptions:")
    print("  --full          Show full configuration without truncation")
    print("  --user-only     Show only user configuration")
    print("  --router-only   Show only router configuration")
    print("  --envoy-only    Show only Envoy configuration")
    print("")
