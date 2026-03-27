"""Envoy configuration generator for vLLM Semantic Router."""

import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader

from cli.consts import DEFAULT_LISTENER_PORT, EXTERNAL_API_MODEL_FORMATS
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

    log.info("Generating Envoy config...")

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

    # Extract models and their endpoints
    # Group endpoints by model for cluster creation
    models = []
    anthropic_models = []  # Anthropic models use a shared cluster

    for model in user_config.providers.models:
        # Handle external API models (e.g., Anthropic) - they use shared clusters
        if model.api_format and model.api_format in EXTERNAL_API_MODEL_FORMATS:
            if model.api_format == "anthropic":
                anthropic_models.append({"name": model.name})
                log.info(
                    f"  Anthropic model: {model.name} (will use shared anthropic_api_cluster)"
                )
            continue

        endpoints = []
        has_https = False
        uses_dns = False

        backend_refs = model.backend_refs
        for index, backend in enumerate(backend_refs):
            # Parse endpoint: can be "host", "host:port", or "host/path" or "host:port/path"
            endpoint_str = backend.endpoint or backend.base_url or ""
            if not endpoint_str:
                continue
            path = ""

            if "://" in endpoint_str:
                parsed = urlparse(endpoint_str)
                host = parsed.netloc
                path = parsed.path.rstrip("/")
                protocol = parsed.scheme or backend.protocol
                port = parsed.port or (443 if protocol == "https" else 80)
                if parsed.port is None and ":" in host:
                    host = parsed.hostname or host
            else:
                protocol = backend.protocol

                # Extract path if present (e.g., "host/path" or "host:port/path")
                if "/" in endpoint_str:
                    # Split by first "/" to separate host[:port] from path
                    parts = endpoint_str.split("/", 1)
                    endpoint_str = parts[0]  # host or host:port
                    path = "/" + parts[1]  # /path

                # Parse host and port
                if ":" in endpoint_str:
                    host, port = endpoint_str.split(":", 1)
                    port = int(port)
                else:
                    host = endpoint_str
                    # Default port based on protocol
                    port = 443 if protocol == "https" else 80

            # Check if this is HTTPS (for transport_socket)
            is_https = protocol == "https"
            if is_https:
                has_https = True

            # Check if host is a domain name (for cluster type)
            # Simple heuristic: if it contains letters or dots in non-IP pattern, it's a domain
            is_domain = not _is_ip_address(host)
            if is_domain:
                uses_dns = True

            endpoints.append(
                {
                    "name": backend.name or f"backend-{index + 1}",
                    "address": host,
                    "port": int(port),
                    "path": path,
                    "weight": backend.weight,
                    "protocol": protocol,
                    "is_https": is_https,
                    "is_domain": is_domain,
                }
            )

        # Sanitize model name for cluster name (replace / with _)
        if not endpoints:
            continue

        cluster_name = model.name.replace("/", "_").replace("-", "_")

        # Determine cluster type based on whether endpoints use domain names
        # Domain names → LOGICAL_DNS, IP addresses → STATIC
        cluster_type = "LOGICAL_DNS" if uses_dns else "STATIC"

        # Determine path prefix - use the first endpoint's path if all endpoints have the same path
        path_prefix = ""
        if endpoints:
            first_path = endpoints[0].get("path", "")
            if first_path and all(ep.get("path", "") == first_path for ep in endpoints):
                path_prefix = first_path

        models.append(
            {
                "name": model.name,
                "cluster_name": cluster_name,
                "endpoints": endpoints,
                "cluster_type": cluster_type,
                "has_https": has_https,
                "path_prefix": path_prefix,
            }
        )

    extproc_host = os.getenv("ENVOY_EXTPROC_ADDRESS", "127.0.0.1")
    router_api_host = os.getenv("ENVOY_ROUTER_API_ADDRESS", "127.0.0.1")
    extproc_host_is_domain = not _is_ip_address(extproc_host)
    router_api_host_is_domain = not _is_ip_address(router_api_host)

    # Prepare template data
    template_data = {
        "listeners": listeners,
        "extproc_host": extproc_host,
        "extproc_port": 50051,
        "extproc_cluster_type": "LOGICAL_DNS" if extproc_host_is_domain else "STATIC",
        "extproc_host_is_domain": extproc_host_is_domain,
        "router_api_host": router_api_host,
        "router_api_port": 8080,
        "router_api_cluster_type": (
            "LOGICAL_DNS" if router_api_host_is_domain else "STATIC"
        ),
        "router_api_host_is_domain": router_api_host_is_domain,
        "models": models,
        "anthropic_models": anthropic_models,  # Anthropic models for shared cluster
        "use_original_dst": False,  # Use static clusters for now
    }

    log.info("  Listeners:")
    for listener in listeners:
        log.info(f"    - {listener['name']}: {listener['address']}:{listener['port']}")
    log.info(f"  Found {len(models)} vLLM model(s):")
    for model in models:
        log.info(f"    - {model['name']} (cluster: {model['cluster_name']})")
        for ep in model["endpoints"]:
            log.info(
                f"        - {ep['name']}: {ep['address']}:{ep['port']} (weight: {ep['weight']})"
            )
    if anthropic_models:
        log.info(f"  Found {len(anthropic_models)} Anthropic model(s):")
        for model in anthropic_models:
            log.info(f"    - {model['name']} (cluster: anthropic_api_cluster)")

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
