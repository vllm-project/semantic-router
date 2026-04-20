"""vLLM Semantic Router CLI main entry point."""

from __future__ import annotations

import click

from cli import __version__
from cli.commands.chat import chat
from cli.commands.eval import eval
from cli.commands.general import config, validate
from cli.commands.runtime import dashboard, logs, serve, status, stop

logo = r"""
       _ _     __  __       ____  ____
__   _| | |_ _|  \/  |     / ___||  _ \
\ \ / / | | | | |\/| |_____\___ \| |_) |
 \ V /| | | |_| | |  |_____|___) |  _ <
  \_/ |_|_|\__,_|_|  |     |____/|_| \_\

vLLM Semantic Router - Intelligent routing for vLLM
"""

REGISTERED_COMMANDS = (
    serve,
    config,
    validate,
    eval,
    status,
    logs,
    stop,
    dashboard,
    chat,
)


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit.")
@click.pass_context
def main(ctx: click.Context, version: bool) -> None:
    """vLLM Semantic Router CLI - Intelligent routing and caching for vLLM endpoints."""
    if version:
        click.echo(f"vllm-sr version: {__version__}")
        ctx.exit()

    if ctx.invoked_subcommand is None:
        click.echo(logo)
        click.echo(ctx.get_help())


for command in REGISTERED_COMMANDS:
    main.add_command(command)


if __name__ == "__main__":
    main()
