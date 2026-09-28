"""OpenAI-compatible data-plane request commands."""

from __future__ import annotations

import click

from cli.commands.chat import chat


@click.group()
def request() -> None:
    """Send requests through an Envoy listener."""


request.add_command(chat)
