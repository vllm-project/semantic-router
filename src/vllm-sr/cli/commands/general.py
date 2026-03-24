"""General Click command entrypoints."""

from __future__ import annotations

import click

from cli.commands.common import exit_with_logged_error
from cli.commands.config import (
    config_command,
    import_config_from_source_command,
    migrate_config_command,
)
from cli.commands.validate import validate_command
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
        vllm-sr config envoy --config my-config.yaml
        vllm-sr config migrate --config old.yaml
        vllm-sr config import --from openclaw --source openclaw.json
    """
    if ctx.invoked_subcommand is not None:
        return
    click.echo(ctx.get_help())


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


@click.command()
@click.option(
    "--config",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
@exit_with_logged_error(log)
def validate(config: str) -> None:
    """
    Validate configuration file.

    Examples:
        vllm-sr validate                    # Uses config.yaml
        vllm-sr validate --config my-config.yaml  # Uses my-config.yaml
    """
    validate_command(config)
