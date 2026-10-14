"""Agent-facing Router configuration lifecycle commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import click
import yaml

from cli.local_stack_config import (
    LocalStack,
    local_stack,
    save_pending_restart,
    stack_document,
)
from cli.router_management_client import (
    RouterManagementClient,
    RouterManagementError,
    default_management_base_url,
)

# A mutation answers once the Router has activated the change, which includes
# loading any model it adds; on CPU that can take minutes.
MUTATION_TIMEOUT_SECONDS = 120
RESTART_REQUIRED = "RESTART_REQUIRED"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _client(
    endpoint: str | None, timeout: float, token_env: str
) -> RouterManagementClient:
    return RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _document_for(
    config_path: Path, mode: str, endpoint: str | None
) -> tuple[str, LocalStack | None]:
    """The document to send, and the local stack it goes to, if any.

    A replacement for the local stack carries the wiring `vllm-sr serve` adds,
    so the Router keeps its management listener and service endpoints.
    """

    if not _targets_local_stack(endpoint) or mode != "replace":
        return _read(config_path), None
    stack = local_stack()
    if stack is None:
        return _read(config_path), None
    if not stack.owns(config_path):
        raise click.ClickException(
            f"{config_path} is not in the directory of the running stack, whose "
            f"state is {stack.state_dir}. Put the file there, or make it the "
            f"stack's source with `vllm-sr serve --config {config_path} "
            "--replace-active-config`."
        )
    return stack_document(stack, config_path), stack


def _targets_local_stack(endpoint: str | None) -> bool:
    """Whether the request goes to the local stack's management API."""

    if endpoint is None:
        return True
    target, local = urlsplit(endpoint), urlsplit(default_management_base_url())
    return target.hostname in _LOOPBACK_HOSTS and target.port == local.port


def _timeout_message(error: RouterManagementError, timeout: float) -> str:
    return (
        f"{error}. The change may still activate: the Router keeps loading what "
        "it needs, and a new model can take minutes on CPU. Check with "
        "`vllm-sr config versions`, or pass a larger --timeout "
        f"(this one was {timeout:g}s) to wait for it."
    )


def _restart_required(
    error: RouterManagementError,
    stack: LocalStack | None,
    config_path: Path,
    plan: Any,
) -> None:
    reason = error.detail or str(error)
    if stack is None:
        raise click.ClickException(
            f"Restart required: {reason}. The running Router can't take this "
            "change; restart it with the new configuration (on Kubernetes, roll "
            "it out with `helm upgrade`)."
        )
    runtime_config = save_pending_restart(stack, config_path, reason)
    message = "Restart required: run `vllm-sr serve` to apply."
    click.echo(f"{message} ({reason})", err=True)
    _json(
        {
            "applied": False,
            "status": "restart_required",
            "message": message,
            "reason": reason,
            "saved": str(runtime_config),
            "plan": plan,
        }
    )


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


def _connection_options(command=None, *, timeout: float = 15):
    def decorate(command):
        command = click.option(
            "--token-env",
            default="VSR_MGMT_TOKEN",
            show_default=True,
            help="Environment variable containing the Router management bearer token.",
        )(command)
        command = click.option(
            "--timeout", type=float, default=timeout, show_default=True
        )(command)
        return click.option(
            "--endpoint",
            default=None,
            help="Router management base URL; defaults to the local Router API port.",
        )(command)

    return decorate(command) if command is not None else decorate


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

    document, _stack = _document_for(config_path, mode, endpoint)
    try:
        result = _client(endpoint, timeout, token_env).plan_config(document, mode)
    except RouterManagementError as error:
        if error.code == RESTART_REQUIRED:
            raise click.ClickException(
                f"Restart required: {error.detail or error}. `vllm-sr config apply` "
                "saves the change for the next `vllm-sr serve`."
            ) from error
        raise
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
@_connection_options(timeout=MUTATION_TIMEOUT_SECONDS)
@_user_errors
def config_apply(
    config_path: Path,
    mode: str,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Plan, compare-and-swap, persist, and hot-reload a configuration.

    A change the running Router can't take without a restart is saved for the
    next `vllm-sr serve` of a local stack, as the Dashboard saves one.
    """

    client = _client(endpoint, timeout, token_env)
    document, stack = _document_for(config_path, mode, endpoint)
    plan = None
    try:
        plan = client.plan_config(document, mode).payload
        if not isinstance(plan, dict):
            raise click.ClickException("Router returned an invalid config plan")
        if not plan.get("changed"):
            _json({"applied": False, "reason": "unchanged", "plan": plan})
            return
        current_etag = plan.get("current_etag")
        if not isinstance(current_etag, str) or not current_etag.strip():
            raise click.ClickException("Router config plan did not return current_etag")
        mutation = client.apply_config(document, mode, current_etag)
    except RouterManagementError as error:
        if error.code == RESTART_REQUIRED:
            _restart_required(error, stack, config_path, plan)
            return
        if error.timed_out:
            raise click.ClickException(_timeout_message(error, timeout)) from error
        raise
    _json({"applied": True, "plan": plan, "result": mutation.payload})


@click.command("versions")
@_connection_options
@_user_errors
def config_versions(endpoint: str | None, timeout: float, token_env: str) -> None:
    """List the configuration history, newest first."""

    _json(_client(endpoint, timeout, token_env).config_versions().payload)


@click.command("rollback")
@click.argument("version")
@_connection_options(timeout=MUTATION_TIMEOUT_SECONDS)
@_user_errors
def config_rollback(
    version: str,
    endpoint: str | None,
    timeout: float,
    token_env: str,
) -> None:
    """Compare-and-swap the active configuration to a recorded version.

    VERSION is a configuration version number from `vllm-sr config versions`,
    or the timestamp of a backup. The restored document activates as a new
    version.
    """

    client = _client(endpoint, timeout, token_env)
    etag = client.get_config().etag
    try:
        _json(client.rollback_config(version, etag).payload)
    except RouterManagementError as error:
        if error.timed_out:
            raise click.ClickException(_timeout_message(error, timeout)) from error
        raise


CONFIG_MANAGEMENT_COMMANDS = (
    config_get,
    config_plan,
    config_apply,
    config_versions,
    config_rollback,
)
