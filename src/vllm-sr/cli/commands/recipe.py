"""Managed Recipe packaging commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import click
import yaml

from cli.recipe_package import RecipePackageError, pack_recipe
from cli.router_management_client import RouterManagementClient


@click.group()
def recipe() -> None:
    """Validate, plan, apply, inspect, or package routing recipes."""


def _connection_options(command):
    command = click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)(
        command
    )
    command = click.option("--timeout", type=float, default=15, show_default=True)(
        command
    )
    return click.option(
        "--endpoint",
        default=None,
        help="Router management base URL; defaults to the local Router API port.",
    )(command)


def _client(
    endpoint: str | None, timeout: float, token_env: str
) -> RouterManagementClient:
    return RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)


def _dump(value: Any) -> None:
    click.echo(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def _user_errors(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except click.ClickException:
            raise
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper


def _load_recipe(path: Path) -> tuple[str, dict[str, Any]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise click.ClickException("Recipe file must contain a YAML object")
    name = str(value.get("name") or "").strip()
    if not name:
        raise click.ClickException("Recipe file must declare name")
    if not isinstance(value.get("routing"), dict):
        raise click.ClickException("Recipe file must declare a routing object")
    return name, value


@recipe.command("list")
@_connection_options
@_user_errors
def list_recipes(endpoint: str | None, timeout: float, token_env: str) -> None:
    """List recipes and the collection ETag."""

    response = _client(endpoint, timeout, token_env).list_recipes()
    if not isinstance(response.payload, dict) or not isinstance(
        response.payload.get("recipes"), list
    ):
        raise ValueError("Router returned an invalid recipe collection")
    _dump({"etag": response.etag, "recipes": response.payload["recipes"]})


@recipe.command("get")
@click.argument("name")
@_connection_options
@_user_errors
def get_recipe(
    name: str,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Read one managed recipe."""

    response = _client(endpoint, timeout, token_env).get_recipe(name)
    _dump({"etag": response.etag, "recipe": response.payload})


@recipe.command("validate")
@click.argument(
    "recipe_file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@_connection_options
@_user_errors
def validate_recipe(
    recipe_file: Path,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Validate a recipe against the running Router without changing config."""

    _name, payload = _load_recipe(recipe_file)
    _dump(_client(endpoint, timeout, token_env).validate_recipe(payload).payload)


@recipe.command("plan")
@click.argument(
    "recipe_file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@_connection_options
@_user_errors
def plan_recipe(
    recipe_file: Path,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Validate a recipe and bind the plan to the current config ETag."""

    name, payload = _load_recipe(recipe_file)
    client = _client(endpoint, timeout, token_env)
    collection = client.list_recipes()
    validation = client.validate_recipe(payload)
    _dump(
        {
            "valid": True,
            "name": name,
            "current_etag": collection.etag,
            "action": (
                validation.payload.get("action")
                if isinstance(validation.payload, dict)
                else None
            ),
        }
    )


@recipe.command("apply")
@click.argument(
    "recipe_file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@_connection_options
@_user_errors
def apply_recipe(
    recipe_file: Path,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Validate then compare-and-swap one recipe into the active config."""

    name, payload = _load_recipe(recipe_file)
    client = _client(endpoint, timeout, token_env)
    collection = client.list_recipes()
    validation = client.validate_recipe(payload).payload
    result = client.apply_recipe(name, payload, collection.etag)
    _dump({"plan": validation, "result": result.payload})


@recipe.command("delete")
@click.argument("name")
@_connection_options
@_user_errors
def delete_recipe(
    name: str,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Compare-and-swap deletion of an unreferenced named recipe."""

    client = _client(endpoint, timeout, token_env)
    collection = client.list_recipes()
    _dump(client.delete_recipe(name, collection.etag).payload)


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
