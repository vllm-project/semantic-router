"""Agent-facing Router configuration lifecycle commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import click
import yaml

from cli.router_management_client import RouterManagementClient


def _client(
    endpoint: str | None, timeout: float, token_env: str
) -> RouterManagementClient:
    return RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json(value: Any) -> None:
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


def _connection_options(command):
    command = click.option(
        "--token-env",
        default="VSR_MGMT_TOKEN",
        show_default=True,
        help="Environment variable containing the Router management bearer token.",
    )(command)
    command = click.option("--timeout", type=float, default=15, show_default=True)(
        command
    )
    return click.option(
        "--endpoint",
        default=None,
        help="Router management base URL; defaults to the local Router API port.",
    )(command)


@click.command("get")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml"]),
    default="yaml",
    show_default=True,
)
@_connection_options
@_user_errors
def config_get(
    output_format: str,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Read the active canonical configuration from a Router."""

    response = _client(endpoint, timeout, token_env).get_config()
    if output_format == "json":
        _json({"etag": response.etag, "config": response.payload})
        return
    click.echo(yaml.safe_dump(response.payload, sort_keys=False), nl=False)


@click.command("plan")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("config.yaml"),
    show_default=True,
)
@click.option(
    "--mode",
    type=click.Choice(["replace", "merge"]),
    default="replace",
    show_default=True,
)
@_connection_options
@_user_errors
def config_plan(
    config_path: Path,
    mode: str,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Validate and plan an exact remote mutation without changing the Router."""

    result = _client(endpoint, timeout, token_env).plan_config(_read(config_path), mode)
    _json(result.payload)


@click.command("apply")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("config.yaml"),
    show_default=True,
)
@click.option(
    "--mode",
    type=click.Choice(["replace", "merge"]),
    default="replace",
    show_default=True,
)
@_connection_options
@_user_errors
def config_apply(
    config_path: Path,
    mode: str,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Plan, compare-and-swap, persist, and hot-reload a configuration."""

    client = _client(endpoint, timeout, token_env)
    yaml_text = _read(config_path)
    plan = client.plan_config(yaml_text, mode).payload
    if not isinstance(plan, dict):
        raise click.ClickException("Router returned an invalid config plan")
    if not plan.get("changed"):
        _json({"applied": False, "reason": "unchanged", "plan": plan})
        return
    current_etag = plan.get("current_etag")
    if not isinstance(current_etag, str) or not current_etag.strip():
        raise click.ClickException("Router config plan did not return current_etag")
    mutation = client.apply_config(yaml_text, mode, current_etag)
    _json({"applied": True, "plan": plan, "result": mutation.payload})


@click.command("versions")
@_connection_options
@_user_errors
def config_versions(endpoint: str | None, timeout: float, token_env: str) -> None:
    """List immutable configuration backup versions."""

    _json(_client(endpoint, timeout, token_env).config_versions().payload)


@click.command("rollback")
@click.argument("version")
@_connection_options
@_user_errors
def config_rollback(
    version: str,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Compare-and-swap the active configuration to a backup version."""

    client = _client(endpoint, timeout, token_env)
    etag = client.get_config().etag
    _json(client.rollback_config(version, etag).payload)


CONFIG_MANAGEMENT_COMMANDS = (
    config_get,
    config_plan,
    config_apply,
    config_versions,
    config_rollback,
)
