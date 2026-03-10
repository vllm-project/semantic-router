"""General Click command entrypoints."""

from __future__ import annotations

import click

from cli.commands.common import exit_with_logged_error
from cli.commands.config import config_command
from cli.commands.init import init_command
from cli.commands.validate import validate_command
from cli.utils import getLogger

log = getLogger(__name__)


@click.command()
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing config.yaml and .vllm-sr without prompting.",
)
@exit_with_logged_error(log)
def init(force: bool) -> None:
    """
    Generate an advanced YAML sample for vLLM Semantic Router.

    Creates config.yaml and .vllm-sr/ with advanced sample files for YAML-first users.
    Most users can start directly with `vllm-sr serve`.

    Examples:
        vllm-sr init
        vllm-sr init --force
    """
    init_command(force=force)


@click.command()
@click.argument("config_type", type=click.Choice(["envoy", "router"]))
@click.option(
    "--config",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
@exit_with_logged_error(log)
def config(config_type: str, config: str) -> None:
    """
    Print generated configuration.

    Examples:
        vllm-sr config envoy
        vllm-sr config router
        vllm-sr config envoy --config my-config.yaml
    """
    config_command(config_type, config)


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
