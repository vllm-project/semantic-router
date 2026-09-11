"""General Click command entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import click

from cli.commands.common import exit_with_logged_error
from cli.commands.config import (
    config_command,
    init_config_command,
    config_schema_command,
    import_config_from_source_command,
    migrate_config_command,
)
from cli.commands.config_management import CONFIG_MANAGEMENT_COMMANDS
from cli.commands.validate import validate_command
from cli.router_management_client import RouterManagementClient
from cli.terminal import echo
from cli.utils import get_logger

log = get_logger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
@exit_with_logged_error(log)
def config(ctx: click.Context) -> None:
    """
    Print generated configuration or run config subcommands.

    Examples:
        vllm-sr config envoy
        vllm-sr config router
        vllm-sr config init --output config.yaml
        vllm-sr config envoy --config my-config.yaml
        vllm-sr config migrate --config old.yaml
        vllm-sr config import --from openclaw --source openclaw.json
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
@exit_with_logged_error(log)
def config_schema(
    endpoint: str | None,
    timeout: float,
    token_env: str,
    full: bool,
    section: str | None,
    surface: str | None,
) -> None:
    """Discover the canonical config contract progressively."""

    config_schema_command(
        endpoint,
        full=full,
        section=section,
        surface=surface,
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


@config.command("import")
@click.option(
    "--from",
    "from_type",
    required=True,
    type=click.Choice(["openclaw"], case_sensitive=False),
    help="Import source type.",
)
@click.option(
    "--source",
    "source_path",
    help="Path to the source config file. Defaults to OpenClaw discovery order.",
)
@click.option(
    "--target",
    "target_path",
    default="config.yaml",
    show_default=True,
    help="Path to the target canonical config file.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing backup files for the source or target paths.",
)
@exit_with_logged_error(log)
def config_import(
    from_type: str,
    source_path: str | None,
    target_path: str,
    force: bool,
) -> None:
    """Import a supported external config source into canonical v0.3 YAML."""

    import_config_from_source_command(
        from_type=from_type,
        source_path=source_path,
        target_path=target_path,
        force=force,
    )


@config.command("validate")
@click.option(
    "--config",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
@click.option(
    "--endpoint",
    default=None,
    help="Also validate with this running Router's authoritative parser.",
)
@click.option("--timeout", type=float, default=15, show_default=True)
@click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)
@exit_with_logged_error(log)
def config_validate(
    config: str,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """
    Validate configuration file.

    Examples:
        vllm-sr config validate
        vllm-sr config validate --config my-config.yaml
    """
    validate_command(config)
    if endpoint:
        response = RouterManagementClient(
            endpoint,
            timeout=timeout,
            token_env=token_env,
        ).validate_config(Path(config).read_text(encoding="utf-8"))
        echo(json.dumps(response.payload, indent=2, sort_keys=True))


for command in CONFIG_MANAGEMENT_COMMANDS:
    config.add_command(command)
