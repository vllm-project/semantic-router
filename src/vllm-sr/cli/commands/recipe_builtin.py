"""User-facing offline built-in Recipe commands."""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path

import click
import yaml

from cli.builtin_recipes import (
    export_builtin_bundle,
    initialize_builtin_recipe,
    list_builtin_recipes,
)
from cli.model_catalog import DEFAULT_CHANNEL
from cli.parser import ConfigParseError


def _command(function):
    @wraps(function)
    def run(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except (OSError, ValueError, ConfigParseError, yaml.YAMLError) as error:
            raise click.ClickException(str(error)) from error
        click.echo(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

    return click.option(
        "--catalog-version", default=DEFAULT_CHANNEL, show_default=True
    )(run)


@click.group()
def builtin() -> None:
    """Discover, export, or bind installed Recipes without a running Router."""


@builtin.command("list")
@_command
def list_recipes(catalog_version: str):
    """List installed bundles, recipe names, and required candidate pools as JSON."""
    return list_builtin_recipes(catalog_version)


@builtin.command("export")
@click.argument("bundle")
@click.option(
    "--output-dir", required=True, type=click.Path(path_type=Path, file_okay=False)
)
@_command
def export(bundle: str, output_dir: Path, catalog_version: str):
    """Export an exact five-file BUNDLE to a new directory."""
    return export_builtin_bundle(bundle, output_dir, catalog_version)


@builtin.command("init")
@click.argument("name")
@click.option("--bundle", required=True, help="Bundle from recipe builtin list.")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Existing canonical provider config to preserve.",
)
@click.option(
    "--bindings",
    "bindings_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="YAML mapping each decision that calls a backend to modelRefs; omit immediate responses.",
)
@click.option("--model-name", required=True, help="Explicit public entrypoint name.")
@click.option(
    "--exclude-decision",
    "excluded_decisions",
    multiple=True,
    help="Explicitly omit a named lane for an unavailable capability; creates a recipe derivative. Repeatable.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="New config path; existing files are never replaced. Keep beside --config when it references local KB assets.",
)
@_command
def initialize(
    name: str,
    bundle: str,
    config_path: Path,
    bindings_path: Path,
    model_name: str,
    output: Path,
    catalog_version: str,
    excluded_decisions: tuple[str, ...],
):
    """Bind one installed recipe to explicit models and validate the new config.

    No model, endpoint, capability, or candidate pool is invented. Every recipe
    decision must meet its declared minimum_candidates before this succeeds.
    """
    return initialize_builtin_recipe(
        name,
        bundle=bundle,
        config_path=config_path,
        bindings_path=bindings_path,
        model_name=model_name,
        output=output,
        version=catalog_version,
        excluded_decisions=excluded_decisions,
    )
