"""Config-driven storage backend provisioning for vllm-sr serve."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

from cli.container_runtime import get_container_runtime
from cli.container_services import (
    container_mount_destinations,
    container_network_disconnect,
    container_start_milvus,
    container_start_postgres,
    container_start_redis,
    container_status,
    container_stop_container,
)
from cli.managed_storage_detection import detect_canonical_storage_backends
from cli.runtime_stack import RuntimeStackLayout
from cli.storage_secrets import (
    CONTAINER_POSTGRES_PASSWORD_PATH,
    CONTAINER_REDIS_CONF_PATH,
    RECOVERY_HINT,
    PostgresSecret,
    StorageSecrets,
    StorageVolumes,
    adopt_storage_volumes,
    default_storage_volume_names,
    ensure_storage_secrets,
    postgres_password_path,
    redis_conf_path,
    rekey_postgres_role,
    storage_state_path,
)
from cli.utils import get_logger

log = get_logger(__name__)

CREDENTIALED_BACKENDS = frozenset({"redis", "postgres"})

# How long a freshly created Postgres container may take to finish `initdb`
# and start accepting TCP connections before an in-place re-key gives up.
POSTGRES_READY_TIMEOUT_SECONDS = 90
POSTGRES_READY_POLL_SECONDS = 1.0
RUNTIME_COMMAND_TIMEOUT_SECONDS = 10

# Mount points only a container this CLI created with per-stack credentials
# carries. They are the evidence that its data volume has a recorded name.
MANAGED_STORAGE_CREDENTIAL_MOUNTS = frozenset(
    {CONTAINER_POSTGRES_PASSWORD_PATH, CONTAINER_REDIS_CONF_PATH}
)


def detect_required_backends(
    config: dict, stack_layout: RuntimeStackLayout | None = None
) -> set[str]:
    """Read store_backend values from the config and return backends that need provisioning.

    Reads from the canonical v0.3 path: global.services.<key>.store_backend and
    falls back to router-owned canonical defaults for local serve workflows.
    Returns only backends the CLI knows how to provision (redis, postgres).
    """
    return detect_canonical_storage_backends(config, stack_layout)


def start_storage_backends(
    required_backends: set[str],
    network_name: str,
    stack_layout: RuntimeStackLayout,
    *,
    state_root_dir: str,
) -> set[str]:
    """Start Docker containers for the required storage backends.

    Redis and Postgres are started with this stack's own credentials. When no
    usable credential state exists the stack is taken over in place: the data
    volumes of the existing containers are resolved first, fresh values are
    generated, the containers are rebuilt against those volumes, Postgres is
    re-keyed, and only then is the state written. That order is the contract --
    see :func:`cli.storage_secrets.ensure_storage_secrets`.

    Returns the set of backends that were actually started.
    """
    if not required_backends:
        return set()

    started: set[str] = set()
    log.info(
        f"Storage backends required by config: {', '.join(sorted(required_backends))}"
    )

    if required_backends & CREDENTIALED_BACKENDS:
        _provision_credentialed_backends(
            required_backends, network_name, stack_layout, state_root_dir, started
        )

    if "milvus" in required_backends:
        _start_backend(
            "Milvus",
            lambda: container_start_milvus(
                network_name,
                stack_layout,
                state_root_dir=state_root_dir,
            ),
        )
        started.add("milvus")

    return started


def provision_storage_backends(
    config: dict,
    network_name: str,
    stack_layout: RuntimeStackLayout,
    *,
    state_root_dir: str,
    additional_backends: set[str] | frozenset[str] = frozenset(),
) -> set[str]:
    """Detect config and control-plane backends and provision this stack."""
    required = detect_required_backends(config, stack_layout) | set(additional_backends)
    return start_storage_backends(
        required,
        network_name,
        stack_layout,
        state_root_dir=state_root_dir,
    )


def _provision_credentialed_backends(
    required_backends: set[str],
    network_name: str,
    stack_layout: RuntimeStackLayout,
    state_root_dir: str,
    started: set[str],
) -> None:
    """Resolve this stack's credentials and start the backends that use them."""

    volumes = None
    targets = required_backends & CREDENTIALED_BACKENDS
    if not storage_state_path(state_root_dir, stack_layout=stack_layout).exists():
        # The inspection has to precede every removal below, because removing a
        # container keeps its volume alive but erases the only record of the
        # pairing. `ensure_storage_secrets` is still the authority on whether a
        # state file is usable; this check only decides which message to log.
        volumes = adopt_storage_volumes(stack_layout)
        existing = _existing_managed_storage(stack_layout)
        _log_takeover_intent(stack_layout, volumes, existing)
        # Takeover is triggered by what exists, not by what the config asks
        # for. A container the config no longer needs still has its volume name
        # committed to the state file, so if it were skipped here the state
        # would record a password its data directory never received -- and the
        # reuse path would never re-key it.
        targets = targets | existing

    applied = False

    def apply_secrets(generated: StorageSecrets) -> None:
        nonlocal applied
        applied = True
        _start_credentialed_backends(
            targets,
            required_backends,
            network_name,
            stack_layout,
            state_root_dir,
            generated,
            started,
            credentials_are_new=True,
        )

    secrets = ensure_storage_secrets(
        state_root_dir,
        stack_layout=stack_layout,
        volumes=volumes,
        apply_secrets=apply_secrets,
    )
    if applied:
        return
    _start_credentialed_backends(
        required_backends & CREDENTIALED_BACKENDS,
        required_backends,
        network_name,
        stack_layout,
        state_root_dir,
        secrets,
        started,
        credentials_are_new=False,
    )


def _existing_managed_storage(stack_layout: RuntimeStackLayout) -> set[str]:
    """Return the credentialed backends that already have a container here."""

    present = set()
    if container_status(stack_layout.redis_container_name) != "not found":
        present.add("redis")
    if container_status(stack_layout.postgres_container_name) != "not found":
        present.add("postgres")
    return present


def _start_credentialed_backends(
    targets: set[str],
    required_backends: set[str],
    network_name: str,
    stack_layout: RuntimeStackLayout,
    state_root_dir: str,
    secrets: StorageSecrets,
    started: set[str],
    *,
    credentials_are_new: bool,
) -> None:
    """Start the credentialed backends, rebuilding them when the values changed.

    A running container keeps serving the credentials it started with -- Redis
    reads its config once, and Postgres holds the mount it was created with --
    so reusing one after generating new values is what produces the
    authentication failure this whole path exists to avoid. Reuse is correct
    only when the credentials came back unchanged from the state file, which is
    the "a restart keeps the same key" case.

    *targets* may exceed *required_backends* during a takeover. A backend the
    config does not need is still re-keyed, then stopped again: the point is to
    leave it consistent with the state file, not to run it.
    """

    if "redis" in targets:
        conf_file = str(redis_conf_path(state_root_dir, stack_layout=stack_layout))
        _start_backend(
            "Redis",
            lambda: container_start_redis(
                network_name,
                stack_layout,
                recreate=credentials_are_new,
                redis_conf_file=conf_file,
                data_volume=secrets.redis.volume,
            ),
        )
        _record_or_park(
            "redis", stack_layout.redis_container_name, required_backends, started
        )

    if "postgres" in targets:
        password_file = str(
            postgres_password_path(state_root_dir, stack_layout=stack_layout)
        )
        _start_backend(
            "Postgres",
            lambda: container_start_postgres(
                network_name,
                stack_layout,
                recreate=credentials_are_new,
                postgres_password_file=password_file,
                data_volume=secrets.postgres.volume,
            ),
        )
        if credentials_are_new:
            rekey_managed_postgres(
                stack_layout.postgres_container_name, secrets.postgres
            )
        _record_or_park(
            "postgres", stack_layout.postgres_container_name, required_backends, started
        )


def _record_or_park(
    backend: str, container_name: str, required_backends: set[str], started: set[str]
) -> None:
    """Keep a required backend running; stop one the config does not ask for."""

    if backend in required_backends:
        started.add(backend)
        return
    log.info(
        f"{container_name} was taken over so its credentials match this "
        "stack's state, and is stopped again because the config does not "
        "require it."
    )
    if not container_stop_container(container_name):
        # It keeps its published loopback port, so the next `serve` would trip
        # over a port conflict for a service it was told not to run.
        log.warning(
            f"{container_name} could not be stopped and is still holding its "
            "port; stop it by hand before the next `vllm-sr serve`."
        )


def rekey_managed_postgres(container_name: str, secret: PostgresSecret) -> None:
    """Apply *secret* to a managed Postgres role that is already initialised.

    ``POSTGRES_PASSWORD_FILE`` is read only by ``initdb``, so a container
    rebuilt against an existing data volume keeps whatever password that volume
    was created with -- ``ALTER ROLE`` is the only mechanism that can change it.
    The image trusts local socket connections, so this needs no knowledge of the
    previous value, which is what lets a takeover run unattended.

    A fresh volume has just been initialised with this very value, so the
    statement is a harmless no-op there and the two cases stay on one path.
    """

    if not _wait_for_postgres(container_name, secret):
        log.error(
            f"{container_name} did not become ready in "
            f"{POSTGRES_READY_TIMEOUT_SECONDS}s, so its credentials were left "
            f"unchanged and no credential state was written. {RECOVERY_HINT}"
        )
        raise SystemExit(1)

    return_code, _stdout, stderr = rekey_postgres_role(container_name, secret)
    if return_code != 0:
        log.error(
            f"Failed to apply the generated Postgres credential to "
            f"{container_name}: {stderr.strip() or f'exit code {return_code}'}. "
            f"No credential state was written; rerun `vllm-sr serve` to retry. "
            f"{RECOVERY_HINT}"
        )
        raise SystemExit(1)
    log.info(f"Applied this stack's Postgres credential to {container_name}")


def _wait_for_postgres(container_name: str, secret: PostgresSecret) -> bool:
    """Poll until Postgres accepts TCP connections, or the deadline passes.

    The TCP check is deliberate: during ``initdb`` the image runs a temporary
    server that listens on the local socket only, so a socket probe would
    report ready while the container is still initialising.
    """

    runtime = get_container_runtime()
    command = [
        runtime,
        "exec",
        container_name,
        "pg_isready",
        "-h",
        "127.0.0.1",
        "-p",
        "5432",
        "-U",
        secret.user,
        "-d",
        secret.database,
    ]
    deadline = time.monotonic() + POSTGRES_READY_TIMEOUT_SECONDS
    while True:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=RUNTIME_COMMAND_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug(f"Postgres readiness probe failed: {exc}")
        # A container that has stopped will never become ready, and the usual
        # reason it stops is the very thing this wait was meant to surface --
        # an unreadable password file, a data directory it cannot open. Waiting
        # out the deadline only delays that report.
        if container_status(container_name) != "running":
            log.error(f"{container_name} is no longer running")
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(POSTGRES_READY_POLL_SECONDS)


def _log_takeover_intent(
    stack_layout: RuntimeStackLayout,
    volumes: StorageVolumes | None,
    existing: set[str],
) -> None:
    """Say what generating fresh credentials means for the data already here."""

    if existing:
        log.warning(
            "Taking over the existing storage containers "
            f"({', '.join(sorted(existing))}) with credentials generated for this "
            "stack. Any value they shared before -- including the constant "
            "this repository used to ship -- is treated as compromised and is "
            "revoked as part of this takeover. Their data volumes are adopted, "
            "so no data is lost."
        )
        return
    defaults = default_storage_volume_names(stack_layout)
    if volumes is None or volumes == defaults:
        log.info(
            "No managed storage containers exist yet, so this stack starts on "
            "empty data volumes. If an older CLI removed them with "
            "`vllm-sr stop`, their data may survive as orphaned volumes. "
            f"{RECOVERY_HINT}"
        )


def _start_backend(name: str, starter: Callable[[], tuple[int, str, str]]) -> None:
    return_code, _stdout, stderr = starter()
    if return_code != 0:
        log.error(f"Failed to start {name}: {stderr}")
        raise SystemExit(1)
    log.info(f"{name} started successfully")


def detach_preserved_storage_container(network_name: str, container_name: str) -> None:
    """Best-effort detach so the stack network can still be removed."""

    return_code, _stdout, stderr = container_network_disconnect(
        network_name, container_name
    )
    if return_code != 0:
        log.warning(
            f"Could not disconnect the preserved {container_name} from "
            f"{network_name}: {stderr.strip() or f'exit code {return_code}'}"
        )


def storage_data_volume_is_recorded(container_name: str) -> bool:
    """Report whether this CLI created *container_name* and knows its volume.

    The evidence has to come from the container itself. ``stop_vllm_sr()``
    takes no arguments and has no config file, so it cannot resolve a state
    root without guessing at the working directory -- which would make "does
    stopping delete my database" depend on where the command was run. A
    credential mount is visible regardless of where the caller stands.

    A runtime that cannot answer counts as "not recorded": keeping a container
    costs a stopped container, removing one can cost the database.
    """

    destinations = container_mount_destinations(container_name)
    if destinations is None:
        return False
    return bool(destinations & MANAGED_STORAGE_CREDENTIAL_MOUNTS)


__all__ = [
    "MANAGED_STORAGE_CREDENTIAL_MOUNTS",
    "detach_preserved_storage_container",
    "detect_required_backends",
    "provision_storage_backends",
    "rekey_managed_postgres",
    "start_storage_backends",
    "storage_data_volume_is_recorded",
]
