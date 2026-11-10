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
GATEWAY_OPTION_HELP = (
    f"The gateway mode the configuration is served in (default: ${GATEWAY_ENV} "
    "or standalone)."
)


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
    "--gateway",
    type=click.Choice(VALID_GATEWAYS),
    default=None,
    help=GATEWAY_OPTION_HELP,
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite the output file if it already exists.",
)
@exit_with_logged_error(log)
def config_migrate(
    config_path: str, output: str | None, gateway: str | None, force: bool
) -> None:
    """Migrate a config file to canonical v0.5 YAML.

    The Router rewrites the file: a local Router binary
    ($VLLM_SR_ROUTER_BINARY) or, present here and never pulled, the Router
    image `vllm-sr serve` runs by default ($VLLM_SR_ROUTER_IMAGE,
    $VLLM_SR_IMAGE or this CLI's own); a running stack's Router is not asked.
    The output is written only when the CLI's structural checks and the
    Router's own validation accept it, in the gateway mode --gateway names.
    """

    migrate_config_command(
        config_path=config_path, output_path=output, force=force, gateway=gateway
    )


@config.command("propose")
@click.option(
    "--config",
    "config_path",
    required=True,
    help="Path to a canonical v0.5 config file. The file is not modified.",
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
    help="Validate with this running Router instead of a local Router.",
)
@click.option(
    "--router-binary",
    default=None,
    metavar="PATH",
    help=(
        "Local Router executable whose own validation to run, preferred over "
        f"the image (default: ${router_validation.ROUTER_BINARY_ENV})."
    ),
)
@click.option(
    "--image",
    default=None,
    help=(
        "Router image whose own validation to run (default: the one `vllm-sr "
        "serve` runs without --image, if present)."
    ),
)
@click.option(
    "--gateway",
    type=click.Choice(VALID_GATEWAYS),
    default=None,
    help=GATEWAY_OPTION_HELP,
)
@click.option(
    "--offline",
    is_flag=True,
    help="Run only the CLI's structural checks, without the Router's validation.",
)
@click.option("--timeout", type=float, default=15, show_default=True)
@click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)
@exit_with_logged_error(log)
def config_validate(
    config: str,
    endpoint: str | None,
    router_binary: str | None,
    image: str | None,
    gateway: str | None,
    offline: bool,
    timeout: float,
    token_env: str,
) -> None:
    """
    Validate configuration file.

    The Router's own validation decides, as it does when `vllm-sr serve` or
    `vllm-sr config apply` loads the file: a local Router binary
    (--router-binary or $VLLM_SR_ROUTER_BINARY), a Router image present here
    and never pulled (--image, else the one `vllm-sr serve` runs by default:
    $VLLM_SR_ROUTER_IMAGE, $VLLM_SR_IMAGE or this CLI's own), or the running
    Router --endpoint names. A running stack's Router is asked only when named.
    The CLI checks the file's structure and the rules it owns as well. With
    --offline, or when no Router can run, only the CLI's structural checks run.

    Examples:
        vllm-sr config validate
        vllm-sr config validate --config my-config.yaml
        vllm-sr config validate --router-binary bin/router
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
            return router_validation.local_router_verdict(
                config_path,
                gateway=resolve_gateway(gateway or os.getenv(GATEWAY_ENV)),
                models_dir=Path(resolve_state_root_dir(config)) / "models",
                image=image,
                binary=router_binary,
            )

    validate_command(config, router_verdict=router_verdict)


for command in CONFIG_MANAGEMENT_COMMANDS:
    config.add_command(command)
