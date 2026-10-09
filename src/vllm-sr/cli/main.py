"""vLLM Semantic Router CLI main entry point."""

from __future__ import annotations

import click

from cli import __version__
from cli.commands.benchmark import benchmark
from cli.commands.completion import completion
from cli.commands.general import config
from cli.commands.instance import instance
from cli.commands.optimize import optimize
from cli.commands.recipe import recipe
from cli.commands.request import request
from cli.commands.route import route
from cli.commands.runtime import dashboard, logs, serve, status, stop
from cli.commands.storage import storage
from cli.terminal import brand

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
    route,
    request,
    benchmark,
    optimize,
    status,
    logs,
    stop,
    dashboard,
    completion,
    recipe,
    storage,
    instance,
)


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit.")
@click.pass_context
def main(ctx: click.Context, version: bool) -> None:
    """vLLM Semantic Router CLI - Signal-driven routing across LLM providers, with a built-in model runtime."""
    if version:
        click.echo(f"vllm-sr version: {__version__}")
        ctx.exit()

    if ctx.invoked_subcommand is None:
        brand(logo)
        click.echo(ctx.get_help())


for command in REGISTERED_COMMANDS:
    main.add_command(command)


if __name__ == "__main__":
    main()
