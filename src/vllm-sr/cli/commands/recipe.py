"""Managed Recipe packaging commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from cli.model_catalog import ModelCatalogError
from cli.recipe_package import RecipePackageError, pack_recipe
from cli.recipe_scaffold import RecipeScaffoldError, scaffold_recipe


@click.group()
def recipe() -> None:
    """Package and inspect managed Recipes."""


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


@recipe.command("scaffold")
@click.option("--name", required=True, help="Recipe directory id (lowercase slug).")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: config/recipes/<name>/).",
)
@click.option(
    "--from",
    "from_model",
    default=None,
    help="Fork from a built-in catalog model such as vllm-sr/mom-v1-blend.",
)
@click.option(
    "--from-recipe",
    default=None,
    help="Fork from an existing maintained recipe directory name.",
)
@click.option(
    "--multi-profile",
    is_flag=True,
    help="Generate a multi-profile config with entrypoints and recipes[].",
)
def scaffold(
    name: str,
    output: Path | None,
    from_model: str | None,
    from_recipe: str | None,
    multi_profile: bool,
) -> None:
    """Create a maintained five-file recipe directory."""

    if from_model and from_recipe:
        raise click.ClickException("use only one of --from or --from-recipe")

    try:
        result = scaffold_recipe(
            name,
            output=output,
            from_recipe=from_recipe,
            from_model=from_model,
            multi_profile=multi_profile,
        )
    except (ModelCatalogError, OSError, RecipeScaffoldError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(json.dumps(result, sort_keys=True))
