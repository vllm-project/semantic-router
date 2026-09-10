"""Envoy configuration generator for vLLM Semantic Router."""

import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from cli.catalog_provider_projection import project_provider_models_for_envoy
from cli.consts import DEFAULT_LISTENER_PORT
from cli.envoy_backend_pool import is_ip_address, project_envoy_backend_group
from cli.models import UserConfig
from cli.utils import get_logger

log = get_logger(__name__)


def _route_request_headers(endpoint: dict) -> list[dict[str, str]]:
    headers: dict[str, str] = {}
    for key, value in (endpoint.get("extra_headers") or {}).items():
        if key and value is not None:
            headers[str(key)] = str(value)
    return [{"key": key, "value": value} for key, value in sorted(headers.items())]


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
                    "api_keys": (
                        list(listener.api_keys)
                        if getattr(listener, "api_keys", None)
                        else []
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
                "api_keys": [],
            }
        )

    # Extract models and their endpoints
    # Group endpoints by model for cluster creation
    models = []
    for model in project_provider_models_for_envoy(user_config):
        backend_group = project_envoy_backend_group(model)
        endpoints = list(backend_group.endpoints)
        has_https = backend_group.has_https
        uses_dns = backend_group.uses_dns

        # Sanitize model name for cluster name (replace / with _)
        if not endpoints:
            continue

        cluster_name = model.name.replace("/", "_").replace("-", "_")

        # Determine cluster type based on whether endpoints use domain names
        # Domain names → LOGICAL_DNS, IP addresses → STATIC
        cluster_type = (
            "STRICT_DNS"
            if uses_dns and len(endpoints) > 1
            else "LOGICAL_DNS" if uses_dns else "STATIC"
        )

        # Determine path prefix - use the first endpoint's path if all endpoints have the same path
        path_prefix = ""
        route_request_headers = []
        if endpoints:
            first_path = endpoints[0].get("path", "")
            if first_path:
                path_prefix = first_path
            route_request_headers = _route_request_headers(endpoints[0])

        auto_host_rewrite = (
            len({endpoint["host_authority"] for endpoint in endpoints}) > 1
        )

        models.append(
            {
                "name": model.name,
                "cluster_name": cluster_name,
                "endpoints": endpoints,
                "cluster_type": cluster_type,
                "has_https": has_https,
                "path_prefix": path_prefix,
                "route_request_headers": route_request_headers,
                "auto_host_rewrite": auto_host_rewrite,
                "reliability": (
                    model.reliability.model_dump()
                    if model.reliability is not None
                    else {}
                ),
            }
        )

    extproc_host = os.getenv("ENVOY_EXTPROC_ADDRESS", "127.0.0.1")
    router_api_host = os.getenv("ENVOY_ROUTER_API_ADDRESS", "127.0.0.1")
    extproc_host_is_domain = not is_ip_address(extproc_host)
    router_api_host_is_domain = not is_ip_address(router_api_host)

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
        "use_original_dst": False,  # Use static clusters for now
    }

    if log_summary:
        log.info("  Listeners:")
        for listener in listeners:
            log.info(
                f"    - {listener['name']}: {listener['address']}:{listener['port']}"
            )
        log.info(f"  Found {len(models)} vLLM model(s):")
        for model in models:
            log.info(f"    - {model['name']} (cluster: {model['cluster_name']})")
            for ep in model["endpoints"]:
                log.info(
                    f"        - {ep['name']}: {ep['address']}:{ep['port']} "
                    f"(weight: {ep['weight']})"
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
