"""General Click command entrypoints."""

from __future__ import annotations

import click

from cli.commands.common import exit_with_logged_error
from cli.commands.config import (
    config_command,
    import_config_from_source_command,
    migrate_config_command,
)
from cli.commands.model import (
    model_fork_command,
    model_list_command,
    model_show_command,
    model_validate_command,
)
from cli.commands.rag import rag_list_command
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


@click.group(invoke_without_command=True)
@click.pass_context
@exit_with_logged_error(log)
def model(ctx: click.Context) -> None:
    """
    Discover bundled virtual models or inspect configured provider models.

    Examples:
        vllm-sr model list
        vllm-sr model show vllm-sr/mom-v1-blend
        vllm-sr model fork vllm-sr/mom-v1-blend mom-v1.yaml
        vllm-sr model list --config my-config.yaml
    """
    if ctx.invoked_subcommand is not None:
        return
    click.echo(ctx.get_help())


@model.command("list")
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Inspect one explicit runtime config instead of the installed catalog.",
)
@click.option("--catalog-version", default="latest", show_default=True)
@click.option(
    "--all-versions", is_flag=True, help="Include immutable release snapshots."
)
@click.option(
    "--all",
    "include_incompatible",
    is_flag=True,
    help="Include incompatible models with their reason.",
)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", show_default=True
)
@exit_with_logged_error(log)
def model_list(
    config_path: str | None,
    catalog_version: str,
    all_versions: bool,
    include_incompatible: bool,
    output: str,
) -> None:
    """List installed virtual models or inspect one explicit config."""

    model_list_command(
        config_path,
        catalog_version=catalog_version,
        include_all_versions=all_versions,
        include_incompatible=include_incompatible,
        output=output,
    )


@model.command("show")
@click.argument("model_id")
@click.option("--catalog-version", default="latest", show_default=True)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", show_default=True
)
@exit_with_logged_error(log)
def model_show(model_id: str, catalog_version: str, output: str) -> None:
    """Show requirements for one installed virtual model."""

    model_show_command(model_id, catalog_version=catalog_version, output=output)


@model.command("fork")
@click.argument("model_id")
@click.argument("destination")
@click.option(
    "--enable",
    "additional_models",
    multiple=True,
    help="Enable another compatible virtual model; catalog assets merge fail-closed.",
)
@click.option("--catalog-version", default="latest", show_default=True)
@click.option(
    "--default", "default_model", default=None, help="Default enabled virtual model."
)
@exit_with_logged_error(log)
def model_fork(
    model_id: str,
    destination: str,
    additional_models: tuple[str, ...],
    catalog_version: str,
    default_model: str | None,
) -> None:
    """Fork one or more compatible built-ins to canonical YAML."""

    model_fork_command(
        (model_id, *additional_models),
        destination,
        catalog_version=catalog_version,
        default_model=default_model,
    )


@model.command("validate")
@click.argument("config_path")
@click.option("--catalog-version", default="latest", show_default=True)
@exit_with_logged_error(log)
def model_validate(config_path: str, catalog_version: str) -> None:
    """Validate canonical YAML and report catalog verification status."""

    model_validate_command(config_path, catalog_version=catalog_version)


@click.group(invoke_without_command=True)
@click.pass_context
@exit_with_logged_error(log)
def rag(ctx: click.Context) -> None:
    """
    Inspect the RAG ingestion pipeline's vector stores.

    Examples:
        vllm-sr rag list
        vllm-sr rag list --endpoint http://localhost:8080
    """
    if ctx.invoked_subcommand is not None:
        return
    click.echo(ctx.get_help())


@rag.command("list")
@click.option(
    "--endpoint",
    default=None,
    help=(
        "Router base URL or full /v1/vector_stores endpoint. "
        "Defaults to the local router API port (8080 + VLLM_SR_PORT_OFFSET)."
    ),
)
@click.option(
    "--timeout",
    default=15,
    show_default=True,
    help="HTTP request timeout in seconds.",
)
@exit_with_logged_error(log)
def rag_list(endpoint: str | None, timeout: int) -> None:
    """List the vector stores created through the RAG ingestion pipeline."""

    rag_list_command(endpoint=endpoint, timeout=timeout)
