"""Offline Recipe packaging commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from cli.recipe_package import RecipePackageError, pack_recipe


@click.group()
def recipe() -> None:
    """Package and inspect offline Recipe bundles."""


@recipe.command("pack")
@click.argument(
    "recipe_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Archive path or output directory (default: the Recipe parent directory).",
)
def pack(recipe_dir: Path, output: Path | None) -> None:
    """Create a deterministic ZIP from an exact five-file RECIPE_DIR."""

    try:
        result = pack_recipe(recipe_dir, output)
    except (OSError, RecipePackageError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(json.dumps(result.as_dict(), sort_keys=True))
