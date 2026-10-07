"""General Click command entrypoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from cli import router_validation
from cli.commands.common import exit_with_logged_error
from cli.commands.config import (
    config_command,
    config_schema_command,
    init_config_command,
    migrate_config_command,
)
from cli.commands.config_management import CONFIG_MANAGEMENT_COMMANDS
from cli.commands.runtime_paths import resolve_state_root_dir
from cli.commands.validate import validate_command
from cli.config_proposal import propose_config_command
from cli.gateway_mode import GATEWAY_ENV, VALID_GATEWAYS, resolve_gateway
from cli.router_management_client import RouterManagementClient
from cli.utils import get_logger

log = get_logger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
@exit_with_logged_error(log)
def config(ctx: click.Context) -> None:
    """
    Print generated configuration or run config subcommands.

    Examples:
        vllm-sr config init --output config.yaml
        vllm-sr config validate --config config.yaml
        vllm-sr config apply --config config.yaml
        vllm-sr config router
        vllm-sr config migrate --config old.yaml
        vllm-sr config envoy    # with --gateway extproc
    """
    if ctx.invoked_subcommand is not None:
        return
    click.echo(ctx.get_help())


@config.command("init")
@click.option(
    "--output",
    default="config.yaml",
    show_default=True,
    help="Path for the new canonical configuration template.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite the output file if it already exists.",
)
@exit_with_logged_error(log)
def config_init(output: str, force: bool) -> None:
    """Create a minimal canonical configuration template."""

    init_config_command(output, force=force)


@config.command("envoy")
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
@exit_with_logged_error(log)
def config_envoy(config_path: str) -> None:
    """Print the generated Envoy configuration."""

    config_command("envoy", config_path)


@config.command("router")
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
@exit_with_logged_error(log)
def config_router(config_path: str) -> None:
    """Print the canonical router configuration."""

    config_command("router", config_path)


@config.command("schema")
@click.option(
    "--endpoint",
    help=(
        "Read the contract from a running Router management origin instead of "
        "the schema bundled with this CLI."
    ),
)
@click.option("--timeout", type=float, default=15, show_default=True)
@click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)
@click.option(
    "--full",
    is_flag=True,
    help="Print the complete JSON Schema instead of the compact index.",
)
@click.option(
    "--section",
    metavar="PATH",
    help="Print one config path and only its referenced definitions.",
)
@click.option(
    "--surface",
    metavar="KIND:NAME",
    help="Print one signal, algorithm, plugin, or projection contract.",
)
@click.option(
    "--expanded",
    is_flag=True,
    help="Include the selected section's self-contained JSON Schema.",
)
@exit_with_logged_error(log)
def config_schema(
    endpoint: str | None,
    timeout: float,
    token_env: str,
    full: bool,
    section: str | None,
    surface: str | None,
    expanded: bool,
) -> None:
    """Discover the canonical config contract progressively."""

    config_schema_command(
        endpoint,
        full=full,
        section=section,
        surface=surface,
        expanded=expanded,
        timeout=timeout,
        token_env=token_env,
    )


@config.command("migrate")
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    help="Path to source config file (default: config.yaml)",
)
@click.option(
    "--output",
    help="Path for migrated canonical config (default: <config>.migrated.yaml)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite the output file if it already exists.",
)
@exit_with_logged_error(log)
def config_migrate(config_path: str, output: str | None, force: bool) -> None:
    """Migrate a legacy or mixed config file to canonical v0.3 YAML."""

    migrate_config_command(config_path=config_path, output_path=output, force=force)


@config.command("propose")
@click.option(
    "--config",
    "config_path",
    required=True,
    help="Path to a canonical v0.3 config file. The file is not modified.",
)
@click.option(
    "--intent",
    required=True,
    help="Maintained proposal intent id, for example selection.latency-aware.",
)
@click.option(
    "--decision",
    default=None,
    help="Name of an existing routing decision. Used by decision intents.",
)
@click.option(
    "--recipe",
    default=None,
    help="Name of an existing recipe. Used by recipe intents.",
)
@exit_with_logged_error(log)
def config_propose(
    config_path: str, intent: str, decision: str | None, recipe: str | None
) -> None:
    """Print a reviewable config proposal. This command does not apply it.

    Examples:
        vllm-sr config propose --config config.yaml \\
            --intent selection.latency-aware --decision default-route
        vllm-sr config propose --config config.yaml \\
            --intent recipe.privacy --recipe privacy-lane
    """

    if not propose_config_command(config_path, intent, decision, recipe):
        sys.exit(1)


@config.command("validate")
@click.option(
    "--config",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
@click.option(
    "--endpoint",
    default=None,
    help="Validate with this running Router instead of the local Router image.",
)
@click.option(
    "--image",
    default=None,
    help="Router image whose own validation to run (default: the stack's, if present).",
)
@click.option(
    "--gateway",
    type=click.Choice(VALID_GATEWAYS),
    default=None,
    help="The gateway mode the configuration is served in (default: standalone).",
)
@click.option(
    "--offline",
    is_flag=True,
    help="Run only the CLI's own checks, without the Router's validation.",
)
@click.option("--timeout", type=float, default=15, show_default=True)
@click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)
@exit_with_logged_error(log)
def config_validate(
    config: str,
    endpoint: str | None,
    image: str | None,
    gateway: str | None,
    offline: bool,
    timeout: float,
    token_env: str,
) -> None:
    """
    Validate configuration file.

    The CLI's own checks run first. The Router's validation then decides, as
    it does when `vllm-sr serve` or `vllm-sr config apply` loads the file: the
    Router in its local image (never pulled), or the running Router --endpoint
    names.

    Examples:
        vllm-sr config validate
        vllm-sr config validate --config my-config.yaml
        vllm-sr config validate --endpoint http://localhost:8080
    """
    config_path = Path(config)
    if offline:
        router_verdict = None
    elif endpoint:
        client = RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)

        def router_verdict():
            return router_validation.validate_with_endpoint(config_path, client)

    else:

        def router_verdict():
            return router_validation.validate_with_image(
                config_path,
                router_validation.validation_image(image),
                gateway=resolve_gateway(gateway or os.getenv(GATEWAY_ENV)),
                models_dir=Path(resolve_state_root_dir(config)) / "models",
            )

    validate_command(config, router_verdict=router_verdict)


for command in CONFIG_MANAGEMENT_COMMANDS:
    config.add_command(command)
