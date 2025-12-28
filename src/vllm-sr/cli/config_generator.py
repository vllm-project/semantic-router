"""Envoy configuration generator for vLLM Semantic Router."""

import os
import yaml
import ipaddress
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader
from cli.utils import getLogger
from cli.models import UserConfig

log = getLogger(__name__)


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
    template_file: str = None,
    template_root: str = None,
) -> Path:
    """
    Generate Envoy configuration from user config.

    Args:
        user_config: Parsed user configuration
        output_file: Output file path for Envoy config
        template_file: Path to Envoy template (optional)
        template_root: Template root directory (optional)

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

    log.info(f"Generating Envoy config...")

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
                "port": 8000,
                "timeout": "300s",
            }
        )

    # Extract models and their endpoints
    # Group endpoints by model for cluster creation
    models = []
    for model in user_config.providers.models:
        endpoints = []
        has_https = False
        uses_dns = False

        for endpoint in model.endpoints:
            # Parse endpoint (host:port or just host)
            if ":" in endpoint.endpoint:
                host, port = endpoint.endpoint.split(":", 1)
                port = int(port)
            else:
                host = endpoint.endpoint
                # Default port based on protocol
                port = 443 if endpoint.protocol == "https" else 80

            # Check if this is HTTPS (for transport_socket)
            is_https = endpoint.protocol == "https"
            if is_https:
                has_https = True

            # Check if host is a domain name (for cluster type)
            # Simple heuristic: if it contains letters or dots in non-IP pattern, it's a domain
            is_domain = not _is_ip_address(host)
            if is_domain:
                uses_dns = True

            endpoints.append(
                {
                    "name": endpoint.name,
                    "address": host,
                    "port": int(port),
                    "weight": endpoint.weight,
                    "protocol": endpoint.protocol,
                    "is_https": is_https,
                    "is_domain": is_domain,
                }
            )

        # Sanitize model name for cluster name (replace / with _)
        cluster_name = model.name.replace("/", "_").replace("-", "_")

        # Determine cluster type based on whether endpoints use domain names
        # Domain names → LOGICAL_DNS, IP addresses → STATIC
        cluster_type = "LOGICAL_DNS" if uses_dns else "STATIC"

        models.append(
            {
                "name": model.name,
                "cluster_name": cluster_name,
                "endpoints": endpoints,
                "cluster_type": cluster_type,
                "has_https": has_https,
            }
        )

    # Prepare template data
    template_data = {
        "listeners": listeners,
        "extproc_port": 50051,
        "models": models,
        "use_original_dst": False,  # Use static clusters for now
    }

    log.info(f"  Listeners:")
    for listener in listeners:
        log.info(f"    - {listener['name']}: {listener['address']}:{listener['port']}")
    log.info(f"  Found {len(models)} model(s):")
    for model in models:
        log.info(f"    - {model['name']} (cluster: {model['cluster_name']})")
        for ep in model["endpoints"]:
            log.info(
                f"        - {ep['name']}: {ep['address']}:{ep['port']} (weight: {ep['weight']})"
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
        log.info(f"✓ Generated Envoy config: {output_path}")
    except Exception as e:
        log.error(f"Failed to write Envoy config: {e}")
        raise

    return output_path


def generate_envoy_config_from_router_config(
    router_config_file: str,
    output_file: str,
    template_file: str = None,
    template_root: str = None,
) -> Path:
    """
    Generate Envoy configuration from router config file.

    Args:
        router_config_file: Path to router config YAML
        output_file: Output file path for Envoy config
        template_file: Path to Envoy template (optional)
        template_root: Template root directory (optional)

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

    log.info(f"Generating Envoy config from {router_config_file}")

    # Load router config
    try:
        with open(router_config_file, "r") as f:
            router_config = yaml.safe_load(f)
    except Exception as e:
        log.error(f"Failed to load router config: {e}")
        raise

    # Extract configuration
    vllm_endpoints = router_config.get("vllm_endpoints", [])

    # Extract listeners from router config or use default
    config_listeners = router_config.get("listeners", [])
    listeners = []
    if config_listeners:
        for listener in config_listeners:
            listeners.append(
                {
                    "name": listener.get("name", "listener_0"),
                    "address": listener.get("address", "0.0.0.0"),
                    "port": listener.get("port", 8000),
                    "timeout": listener.get("timeout", "300s"),
                }
            )
    else:
        # Default listener
        listeners.append(
            {
                "name": "listener_0",
                "address": "0.0.0.0",
                "port": 8000,
                "timeout": "300s",
            }
        )

    # Prepare template data
    template_data = {
        "listeners": listeners,
        "extproc_port": 50051,
        "vllm_endpoints": vllm_endpoints,
        "use_original_dst": False,  # Use static clusters for now
    }

    log.info(f"Listeners:")
    for listener in listeners:
        log.info(f"  - {listener['name']}: {listener['address']}:{listener['port']}")
    log.info(f"Found {len(vllm_endpoints)} vLLM endpoints")
    for endpoint in vllm_endpoints:
        log.info(f"  - {endpoint['name']}: {endpoint['address']}:{endpoint['port']}")

    # Check if template exists
    template_path = Path(template_root) / template_file
    if not template_path.exists():
        log.warning(f"Template not found: {template_path}")
        log.warning("Skipping Envoy config generation")
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
        log.info(f"✓ Generated Envoy config: {output_path}")
    except Exception as e:
        log.error(f"Failed to write Envoy config: {e}")
        raise

    return output_path


if __name__ == "__main__":
    """Entry point when run as: python -m cli.config_generator"""
    import sys
    from cli.parser import parse_user_config

    if len(sys.argv) < 3:
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

        log.info(f"✓ Envoy configuration generated: {output_file}")
    except Exception as e:
        log.error(f"Config generation failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
