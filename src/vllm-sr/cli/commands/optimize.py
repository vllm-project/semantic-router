"""Offline and online routing optimization commands."""

from __future__ import annotations

import click

from cli.commands.recipe_learning import recipe_learning


@click.group()
def optimize() -> None:
    """Analyze routing evidence and produce candidate recipe changes."""


optimize.add_command(recipe_learning)
