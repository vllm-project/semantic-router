"""Envoy configuration generator for vLLM Semantic Router."""

import ipaddress
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from cli.consts import DEFAULT_LISTENER_PORT
from cli.managed_envoy_contract import (
    INTERNAL_REQUEST_HEADERS,
    resolve_backend_dispatch_endpoint,
)
from cli.models import UserConfig
from cli.utils import get_logger

log = get_logger(__name__)


def _is_ip_address(host: str) -> bool:
    """
    Check if a host string is an IP address (IPv4 or IPv6).

    Args:
        host: Host string to check

    Returns:
        bool: True if host is an IP address, False if it's a domain name
    """
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def generate_envoy_config_from_user_config(
    user_config: UserConfig,
    output_file: str,
    template_file: str | None = None,
    template_root: str | None = None,
    *,
    log_summary: bool = True,
) -> Path:
    """
    Generate Envoy configuration from user config.

    Args:
        user_config: Parsed user configuration
        output_file: Output file path for Envoy config
        template_file: Path to Envoy template (optional)
        template_root: Template root directory (optional)
        log_summary: Emit the human-readable generation summary. Machine-readable
            callers disable this so their output streams remain valid documents.

    Returns:
        Path: Path to generated Envoy config
    """
    # Default template paths - templates are now in cli/templates/
    if template_file is None:
        template_file = os.getenv("ENVOY_TEMPLATE_FILE", "envoy.template.yaml")
    if template_root is None:
        # Default to templates directory in cli package
        cli_dir = Path(__file__).parent  # cli/config_generator.py -> cli/
        default_template_root = cli_dir / "templates"
        template_root = os.getenv("TEMPLATE_ROOT", str(default_template_root))

    if log_summary:
        log.info("Generating Envoy config...")

    backend_dispatch = resolve_backend_dispatch_endpoint(user_config)

    # Extract all listeners
    listeners = []
    if user_config.listeners:
        for listener in user_config.listeners:
            listeners.append(
                {
                    "name": listener.name,
                    "address": listener.address,
                    "port": listener.port,
                    "timeout": (
                        listener.timeout if hasattr(listener, "timeout") else "300s"
                    ),
                }
            )
    else:
        # Default listener if none configured
        listeners.append(
            {
                "name": "listener_0",
                "address": "0.0.0.0",
                "port": DEFAULT_LISTENER_PORT,
                "timeout": "300s",
            }
        )

    extproc_host = os.getenv("ENVOY_EXTPROC_ADDRESS", "127.0.0.1")
    extproc_host_is_domain = not _is_ip_address(extproc_host)

    # Prepare template data
    template_data = {
        "listeners": listeners,
        "extproc_host": extproc_host,
        "extproc_port": 50051,
        "extproc_cluster_type": "LOGICAL_DNS" if extproc_host_is_domain else "STATIC",
        "extproc_host_is_domain": extproc_host_is_domain,
        "backend_dispatch": backend_dispatch,
        "internal_request_headers": INTERNAL_REQUEST_HEADERS,
    }

    if log_summary:
        log.info("  Listeners:")
        for listener in listeners:
            log.info(
                f"    - {listener['name']}: {listener['address']}:{listener['port']}"
            )
        log.info(
            "  Backend dispatch: " f"{backend_dispatch.address}:{backend_dispatch.port}"
        )

    # Check if template exists
    template_path = Path(template_root) / template_file
    if not template_path.exists():
        log.warning(f"Template not found: {template_path}")
        log.warning("Skipping Envoy config generation")
        log.warning("To generate Envoy config, provide envoy.template.yaml")
        return None

    # Render template
    try:
        env = Environment(loader=FileSystemLoader(template_root))
        template = env.get_template(template_file)
        rendered = template.render(template_data)
    except Exception as e:
        log.error(f"Failed to render template: {e}")
        raise

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    try:
        with open(output_path, "w") as f:
            f.write(rendered)
        if log_summary:
            log.info(f"Generated Envoy config: {output_path}")
    except Exception as e:
        log.error(f"Failed to write Envoy config: {e}")
        raise

    return output_path


if __name__ == "__main__":
    """Entry point when run as: python -m cli.config_generator"""
    import sys

    from cli.parser import parse_user_config

    minimum_args = 3
    if len(sys.argv) < minimum_args:
        print("Usage: python -m cli.config_generator <config.yaml> <output_envoy.yaml>")
        print("  Generates Envoy configuration from user config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # Parse user config
        user_config = parse_user_config(config_file)

        # Generate Envoy config from user config
        generate_envoy_config_from_user_config(user_config, output_file)

        log.info(f"Envoy configuration generated: {output_file}")
    except Exception as e:
        log.error(f"Config generation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
