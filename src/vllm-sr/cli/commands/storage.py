"""Lifecycle commands for the local stack's storage credentials."""

from __future__ import annotations

import click

from cli.commands.common import exit_with_logged_error
from cli.commands.runtime_paths import resolve_state_root_dir
from cli.commands.runtime_support import apply_container_runtime_override
from cli.consts import SUPPORTED_CONTAINER_RUNTIMES
from cli.container_services import container_start_redis, container_status
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.storage_backends import rekey_managed_postgres
from cli.storage_secrets import (
    RECOVERY_HINT,
    StorageSecrets,
    redis_conf_path,
    rotate_storage_secrets,
)
from cli.terminal import success
from cli.utils import get_logger

log = get_logger(__name__)

CONFIG_HELP = (
    "Config file whose directory holds the runtime state (default: "
    "config.yaml, matching `vllm-sr serve`)."
)
RUNTIME_HELP = (
    "Container runtime for the local Docker target: "
    f"{', '.join(SUPPORTED_CONTAINER_RUNTIMES)}"
)


@click.group()
def storage() -> None:
    """Manage the local stack's Redis and Postgres credentials."""


@storage.command("rotate")
@click.option("--config", "config", default=None, help=CONFIG_HELP)
@click.option(
    "--runtime",
    type=click.Choice(SUPPORTED_CONTAINER_RUNTIMES, case_sensitive=False),
    default=None,
    help=RUNTIME_HELP,
)
@exit_with_logged_error(log)
def rotate(config: str | None, runtime: str | None) -> None:
    """Replace this stack's storage credentials with freshly generated ones.

    Rotation generates new values, applies them to Postgres with `ALTER ROLE`
    and to Redis by rebuilding it against the same named volume, then asks you
    to re-run `serve` so Router picks the new values up.

    Router has to be restarted, and this command deliberately does not do it
    for you. Router receives the credentials as environment values captured
    when its container was created, so `restart` would bring back the old ones
    and only a re-create picks up the new ones -- and re-creating it here would
    have to guess the images, profile, and Recipe bindings you originally
    served with. Re-run your own `serve` command instead.

    Until you do, the stack is degraded: `ALTER ROLE` takes effect at once, so
    connections Router already holds keep working while every new one fails.
    The order is forced -- restarting Router first would start it on a
    credential Postgres has not accepted yet -- so run the two steps back to
    back and plan the rotation for a moment when a brief restart is acceptable.

    The scope is one stack, resolved from VLLM_SR_STACK_NAME exactly like
    `serve`, `stop`, and `status`. Rotate other stacks one at a time. There is
    deliberately no cross-stack mode: a failure partway through would leave
    some stacks revoked and others not, with no value left to roll back to.

    Examples:
        vllm-sr storage rotate
        VLLM_SR_STACK_NAME=staging vllm-sr storage rotate
    """

    apply_container_runtime_override(runtime)
    stack_layout = resolve_runtime_stack()
    config_path = config or "config.yaml"
    state_root_dir = resolve_state_root_dir(config_path)

    postgres_container = stack_layout.postgres_container_name
    if container_status(postgres_container) != "running":
        raise click.ClickException(
            f"{postgres_container} is not running. Rotation re-keys the live "
            "Postgres role in place, which needs the container up. Start the "
            "stack with `vllm-sr serve` and rotate again."
        )

    def apply_secrets(rotated: StorageSecrets) -> None:
        rekey_managed_postgres(postgres_container, rotated.postgres)
        _rebuild_redis(stack_layout, state_root_dir, rotated)

    rotate_storage_secrets(
        state_root_dir,
        stack_layout=stack_layout,
        apply_secrets=apply_secrets,
    )
    success(f"Storage credentials rotated for stack {stack_layout.stack_name}")
    log.warning(
        "Router is still running on the previous credentials. Every new "
        "connection it opens to Postgres fails until it is recreated. Re-run "
        "the `vllm-sr serve` command you started this stack with, now."
    )


def _rebuild_redis(
    stack_layout: RuntimeStackLayout, state_root_dir: str, rotated: StorageSecrets
) -> None:
    """Restart Redis against the rewritten config file.

    Redis reads `requirepass` once at startup, so a running container keeps
    serving the previous value; there is no live re-key path on purpose. The
    rebuild goes through `docker stop`, whose SIGTERM lets the image reach its
    save point, and remounts the volume recorded in the credential state, so
    the data survives the restart.
    """

    if container_status(stack_layout.redis_container_name) == "not found":
        # Nothing is holding the previous value, so rewriting the config file
        # already completes the rotation for Redis. Creating a container here
        # would start a service this stack does not run, as a side effect of a
        # credential change.
        log.info(
            f"{stack_layout.redis_container_name} does not exist; its rotated "
            "config file is in place and the next `vllm-sr serve` will use it."
        )
        return

    conf_file = str(redis_conf_path(state_root_dir, stack_layout=stack_layout))
    return_code, _stdout, stderr = container_start_redis(
        stack_layout.network_name,
        stack_layout,
        recreate=True,
        redis_conf_file=conf_file,
        data_volume=rotated.redis.volume,
    )
    if return_code != 0:
        raise click.ClickException(
            f"Failed to restart {stack_layout.redis_container_name} with the "
            f"rotated credential: {stderr.strip() or f'exit code {return_code}'}. "
            f"{RECOVERY_HINT}"
        )
