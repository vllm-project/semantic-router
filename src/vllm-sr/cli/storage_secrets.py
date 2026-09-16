"""Per-stack credentials for the CLI-managed Redis and Postgres containers.

The CLI provisions both storage services, so it also owns their credentials:
each stack generates its own values instead of inheriting a constant that ships
in the repository. Generated values live in one owner-only state file below the
private runtime-state directory, reach the containers through bind-mounted
credential files, and reach Router through an inherited environment name. They
never enter a ``docker`` argv list, a generated config file, or a log record.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_urlsafe
from typing import NoReturn

from cli.commands.runtime_paths import (
    CONTAINER_READABLE_STATE_FILE_MODE,
    private_runtime_state_subdirectory,
    read_private_state_bytes,
    write_private_state_bytes,
)
from cli.consts import DEFAULT_STACK_NAME
from cli.container_mounts import (
    ContainerMountsUnavailableError,
    inspect_container_mounts,
)
from cli.container_runtime import get_container_runtime
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.utils import get_logger

log = get_logger(__name__)

# Reserved host environment names Router expands inside the runtime config.
POSTGRES_PASSWORD_ENV = "VLLM_SR_STACK_POSTGRES_PASSWORD"
REDIS_PASSWORD_ENV = "VLLM_SR_STACK_REDIS_PASSWORD"
STORAGE_SECRET_ENV_NAMES: tuple[str, str] = (POSTGRES_PASSWORD_ENV, REDIS_PASSWORD_ENV)

# What a generated runtime config carries in place of the value itself.
POSTGRES_PASSWORD_PLACEHOLDER = f"${{{POSTGRES_PASSWORD_ENV}}}"
REDIS_PASSWORD_PLACEHOLDER = f"${{{REDIS_PASSWORD_ENV}}}"

STORAGE_SECRETS_SCHEMA = "vllm-sr/storage-secrets/v1"
STORAGE_SECRETS_DIRECTORY = "storage-secrets"

# Container-side paths are fixed: one container always serves one stack, so the
# stack suffix belongs on the host file name only.
CONTAINER_POSTGRES_PASSWORD_PATH = "/run/secrets/postgres-password"
CONTAINER_REDIS_CONF_PATH = "/usr/local/etc/redis/redis.conf"

# Data directories the managed images declare as volumes.
POSTGRES_DATA_MOUNT_PATH = "/var/lib/postgresql/data"
REDIS_DATA_MOUNT_PATH = "/data"

MANAGED_POSTGRES_USER = "router"
MANAGED_POSTGRES_DATABASE = "vsr"

# 256 bits of CSPRNG output, URL-safe so neither a Redis ``requirepass`` line
# nor a Postgres connection string needs escaping.
SECRET_TOKEN_BYTES = 32

RUNTIME_COMMAND_TIMEOUT_SECONDS = 10

RECOVERY_HINT = (
    "Delete the state file and rerun `vllm-sr serve` to reprovision this "
    "stack; see the storage credential recovery section of the "
    "security-hardening documentation."
)


class StorageSecretError(ValueError):
    """One unusable storage credential state, reported instead of regenerated."""


@dataclass(frozen=True)
class PostgresSecret:
    """Managed Postgres role, database, and the durable volume behind them."""

    user: str
    database: str
    password: str = field(repr=False)
    volume: str


@dataclass(frozen=True)
class RedisSecret:
    """Managed Redis ``requirepass`` value and the durable volume behind it."""

    password: str = field(repr=False)
    volume: str


@dataclass(frozen=True)
class StorageSecrets:
    """One stack's complete storage credential state."""

    stack: str
    postgres: PostgresSecret
    redis: RedisSecret


@dataclass(frozen=True)
class StorageVolumes:
    """Volume names the two storage containers must mount for this stack."""

    postgres: str
    redis: str


def _resolve_layout(stack_layout: RuntimeStackLayout | None) -> RuntimeStackLayout:
    return stack_layout or resolve_runtime_stack()


def _fail(message: str, path: Path) -> NoReturn:
    raise StorageSecretError(f"{message}: {path}. {RECOVERY_HINT}")


def _redact(text: str | None, *values: str) -> str:
    """Remove generated values from runtime output before it can be logged."""

    redacted = text or ""
    for value in values:
        if value:
            redacted = redacted.replace(value, "***")
    return redacted


def storage_secrets_directory(state_root_dir: str | Path) -> Path:
    """Return the owner-only directory that holds every credential artifact."""

    try:
        return private_runtime_state_subdirectory(
            state_root_dir, STORAGE_SECRETS_DIRECTORY
        )
    except StorageSecretError:
        raise
    except ValueError as error:
        raise StorageSecretError(
            f"storage credential directory is unusable: {error}. {RECOVERY_HINT}"
        ) from error


def _stack_filename(base: str, extension: str, stack_name: str) -> str:
    """Name one file after the stack, following the runtime-config convention.

    ``.vllm-sr`` is a state-root directory shared by every stack, so each file
    below it must carry the stack suffix. A file that forgets the suffix is
    silently overwritten by the next stack that writes it.
    """

    if stack_name == DEFAULT_STACK_NAME:
        return f"{base}{extension}"
    return f"{base}.{stack_name}{extension}"


def storage_state_path(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Return this stack's credential state file path."""

    layout = _resolve_layout(stack_layout)
    directory = storage_secrets_directory(state_root_dir)
    return directory / _stack_filename("secrets", ".json", layout.stack_name)


def redis_conf_path(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Return the host path of this stack's Redis config file."""

    layout = _resolve_layout(stack_layout)
    directory = storage_secrets_directory(state_root_dir)
    return directory / _stack_filename("redis", ".conf", layout.stack_name)


def postgres_password_path(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> Path:
    """Return the host path of this stack's Postgres password file."""

    layout = _resolve_layout(stack_layout)
    directory = storage_secrets_directory(state_root_dir)
    return directory / _stack_filename("postgres-password", "", layout.stack_name)


def default_storage_volume_names(
    stack_layout: RuntimeStackLayout | None = None,
) -> StorageVolumes:
    """Return the deterministic named volumes a fresh stack creates."""

    layout = _resolve_layout(stack_layout)
    return StorageVolumes(
        postgres=f"{layout.postgres_container_name}-data",
        redis=f"{layout.redis_container_name}-data",
    )


def adopted_volume_name(container_name: str, destination: str) -> str | None:
    """Return the volume an existing container already mounts at *destination*.

    Adoption exists because the managed images declare ``VOLUME``, so every
    ``docker run`` so far created an anonymous volume whose name lives only in
    the owning container's ``.Mounts``. Reading it before the container is
    removed is the one chance to keep that data; neither Docker nor Podman can
    rename a volume afterwards, which is why the resolved name is recorded in
    the credential state rather than recomputed.

    ``None`` means "nothing to adopt, create a fresh named volume": either the
    container is gone, or its image declares no ``VOLUME`` (newer ``redis``
    tags dropped the declaration, which is why an empty ``.Mounts`` is a normal
    branch rather than an error).

    A runtime that cannot answer is not the same thing and raises instead.
    Treating an unreachable daemon as "nothing to adopt" would mint a fresh
    volume beside data that is still there, so only an explicit "no such
    object" counts as absence.
    """

    try:
        mounts = inspect_container_mounts(container_name)
    except ContainerMountsUnavailableError as exc:
        if exc.container_absent:
            log.debug(f"No container to adopt a volume from: {container_name}")
            return None
        raise StorageSecretError(
            f"Cannot inspect {container_name} to find its data volume: {exc}"
        ) from exc

    for mount in mounts:
        if mount.get("Type") != "volume":
            continue
        if mount.get("Destination") != destination:
            continue
        name = str(mount.get("Name") or "").strip()
        if name:
            return name
    return None


def adopt_storage_volumes(
    stack_layout: RuntimeStackLayout | None = None,
) -> StorageVolumes:
    """Resolve both data volumes, adopting existing ones where they still exist.

    This must run before the existing containers are removed: removal keeps the
    volume alive but destroys the only record of which volume belonged to which
    container.
    """

    layout = _resolve_layout(stack_layout)
    defaults = default_storage_volume_names(layout)
    postgres = adopted_volume_name(
        layout.postgres_container_name, POSTGRES_DATA_MOUNT_PATH
    )
    redis = adopted_volume_name(layout.redis_container_name, REDIS_DATA_MOUNT_PATH)
    if postgres:
        log.info(f"Adopting existing Postgres data volume: {postgres}")
    if redis:
        log.info(f"Adopting existing Redis data volume: {redis}")
    return StorageVolumes(
        postgres=postgres or defaults.postgres,
        redis=redis or defaults.redis,
    )


def storage_secret_env(secrets: StorageSecrets) -> dict[str, str]:
    """Return the environment overlay that hands the values to one child process.

    Callers merge this into a copy of ``os.environ`` for a single
    ``subprocess.run(env=...)`` call, paired with an inheriting ``-e NAME``
    flag. It must never be assigned into ``os.environ`` itself, which would
    expose the values to every other process the CLI spawns.
    """

    return {
        POSTGRES_PASSWORD_ENV: secrets.postgres.password,
        REDIS_PASSWORD_ENV: secrets.redis.password,
    }


def write_postgres_password_file(
    state_root_dir: str | Path,
    secrets: StorageSecrets,
    *,
    stack_layout: RuntimeStackLayout | None = None,
) -> Path:
    """Write the file ``POSTGRES_PASSWORD_FILE`` points at.

    The Postgres entrypoint reads the file as root before dropping privileges,
    so this one stays owner-only. It carries no trailing newline because the
    entrypoint uses the whole file content as the password.
    """

    path = postgres_password_path(state_root_dir, stack_layout=stack_layout)
    return write_private_state_bytes(path, secrets.postgres.password.encode("utf-8"))


def write_redis_conf_file(
    state_root_dir: str | Path,
    secrets: StorageSecrets,
    *,
    stack_layout: RuntimeStackLayout | None = None,
) -> Path:
    """Write the Redis config that carries ``requirepass`` for this stack.

    The value is mounted rather than passed as ``redis-server --requirepass``:
    a container process is a host process, and ``/proc/<pid>/cmdline`` is world
    readable, so the argv form would publish the password to every host user.

    This file is 0644 on purpose -- do not "harden" it to 0600. The Redis image
    entrypoint drops to uid 999 before it reads the config, so an owner-only
    file is unreadable inside the container and Redis fails to start. Privacy
    comes from the enclosing 0700 directory instead: bind-mount path resolution
    starts at the container's mount point and never traverses the host parent
    directory, so the container reads the file while other host users cannot
    even enter the directory holding it.
    """

    path = redis_conf_path(state_root_dir, stack_layout=stack_layout)
    content = f"requirepass {secrets.redis.password}\n".encode()
    return write_private_state_bytes(
        path, content, mode=CONTAINER_READABLE_STATE_FILE_MODE
    )


def materialize_storage_secret_files(
    state_root_dir: str | Path,
    secrets: StorageSecrets,
    *,
    stack_layout: RuntimeStackLayout | None = None,
) -> tuple[Path, Path]:
    """Rewrite both container-facing credential files from *secrets*.

    Returns the Postgres password path and the Redis config path, in that
    order. Rewriting is idempotent, so every ``serve`` restores a file a user
    or a backup tool removed.
    """

    layout = _resolve_layout(stack_layout)
    return (
        write_postgres_password_file(state_root_dir, secrets, stack_layout=layout),
        write_redis_conf_file(state_root_dir, secrets, stack_layout=layout),
    )


def load_storage_secrets(
    state_root_dir: str | Path, *, stack_layout: RuntimeStackLayout | None = None
) -> StorageSecrets:
    """Load this stack's credentials, failing closed instead of regenerating.

    Silent regeneration is the failure mode this guards against: the CLI would
    believe it holds valid credentials while every consumer is rejected, which
    is the hardest class of failure to diagnose.
    """

    layout = _resolve_layout(stack_layout)
    path = storage_state_path(state_root_dir, stack_layout=layout)
    secrets = _read_state(path, layout)
    if secrets is None:
        raise StorageSecretError(
            f"no storage credential state exists for stack {layout.stack_name}; "
            f"run `vllm-sr serve` to provision it: {path}"
        )
    return secrets


def ensure_storage_secrets(
    state_root_dir: str | Path,
    *,
    stack_layout: RuntimeStackLayout | None = None,
    volumes: StorageVolumes | None = None,
    apply_secrets: Callable[[StorageSecrets], None] | None = None,
) -> StorageSecrets:
    """Return this stack's credentials, generating and committing them if absent.

    A usable state is reused verbatim so a restart keeps the same key; only the
    derived container-facing files are rewritten. Otherwise a fresh set is
    generated and *apply_secrets* is invoked to re-key and rebuild the
    containers.

    The step order is the contract, not an implementation detail: existing
    mounts are resolved first, the values are generated in memory, the
    containers are re-keyed, and the state file is written last as the commit
    point. Writing state any earlier would let a failed re-key look like a
    healthy stack on the next ``serve`` -- Router would then authenticate with
    a key the database never received. Because re-keying never needs the old
    value, every step before the commit is idempotent and a crash simply
    replays the whole takeover.

    *apply_secrets* runs only when a new set is generated, so its invocation is
    also the signal a caller uses to report that previously shared values have
    been revoked.
    """

    layout = _resolve_layout(stack_layout)
    path = storage_state_path(state_root_dir, stack_layout=layout)

    existing = _read_state(path, layout)
    if existing is not None:
        materialize_storage_secret_files(state_root_dir, existing, stack_layout=layout)
        return existing

    resolved_volumes = volumes if volumes is not None else adopt_storage_volumes(layout)
    generated = _generate_secrets(layout, resolved_volumes)
    materialize_storage_secret_files(state_root_dir, generated, stack_layout=layout)
    if apply_secrets is not None:
        apply_secrets(generated)
    _write_state(path, generated)
    log.info(f"Generated storage credentials for stack {layout.stack_name}")
    return generated


def rotate_storage_secrets(
    state_root_dir: str | Path,
    *,
    stack_layout: RuntimeStackLayout | None = None,
    apply_secrets: Callable[[StorageSecrets], None] | None = None,
) -> StorageSecrets:
    """Replace this stack's credentials, keeping its recorded data volumes.

    The previous values are neither retained nor needed: both backends store a
    single credential, so the old one stops working the moment *apply_secrets*
    re-keys them, and the recovery path never consults it. State is written
    last for the same reason as in :func:`ensure_storage_secrets`; a rotation
    that crashes before the commit leaves the old state in place and is retried
    by deleting it and rerunning ``serve``.
    """

    layout = _resolve_layout(stack_layout)
    path = storage_state_path(state_root_dir, stack_layout=layout)
    current = load_storage_secrets(state_root_dir, stack_layout=layout)

    rotated = StorageSecrets(
        stack=layout.stack_name,
        postgres=PostgresSecret(
            user=current.postgres.user,
            database=current.postgres.database,
            password=token_urlsafe(SECRET_TOKEN_BYTES),
            volume=current.postgres.volume,
        ),
        redis=RedisSecret(
            password=token_urlsafe(SECRET_TOKEN_BYTES),
            volume=current.redis.volume,
        ),
    )
    materialize_storage_secret_files(state_root_dir, rotated, stack_layout=layout)
    if apply_secrets is not None:
        try:
            apply_secrets(rotated)
        except BaseException as error:
            # Unlike a takeover, a half-finished rotation is not self-healing.
            # `ALTER ROLE` may already have landed while the committed state
            # still names the previous value, and the next `serve` would
            # faithfully restore that stale value and hand Router a credential
            # Postgres no longer accepts. Say so here, where the cause is still
            # visible, instead of letting it surface as an unexplained
            # authentication failure later.
            materialize_storage_secret_files(
                state_root_dir, current, stack_layout=layout
            )
            raise StorageSecretError(
                f"rotation of stack {layout.stack_name} failed partway through, "
                "so its backends may no longer agree on a credential. The "
                f"recorded state was left untouched. {RECOVERY_HINT}"
            ) from error
    _write_state(path, rotated)
    log.info(f"Rotated storage credentials for stack {layout.stack_name}")
    return rotated


def postgres_rekey_statement(secret: PostgresSecret) -> str:
    """Return the ``ALTER ROLE`` statement that re-keys the managed role."""

    return (
        f"ALTER ROLE {_quote_identifier(secret.user)} "
        f"WITH PASSWORD {_quote_literal(secret.password)};\n"
    )


def rekey_postgres_role(
    container_name: str,
    secret: PostgresSecret,
    *,
    timeout: int = RUNTIME_COMMAND_TIMEOUT_SECONDS,
) -> tuple[int, str, str]:
    """Change the managed Postgres password in place, over stdin.

    ``POSTGRES_PASSWORD*`` only applies during ``initdb``, so once a named
    volume keeps the data directory alive, recreating the container can no
    longer change the password -- ``ALTER ROLE`` is the only mechanism left.

    The statement travels on stdin because a container process is a host
    process whose argv every host user can read. The image's default pg_hba
    trusts the local socket, so this needs no knowledge of the previous value,
    which is what lets an unattended takeover succeed without data loss.

    Returns ``(returncode, stdout, stderr)`` with the new value stripped from
    both streams, so a caller may log the result verbatim.
    """

    runtime = get_container_runtime()
    command = [
        runtime,
        "exec",
        "-i",
        container_name,
        "psql",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        secret.user,
        "-d",
        secret.database,
        "-f",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            input=postgres_rekey_statement(secret),
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        return (
            exc.returncode,
            _redact(exc.stdout, secret.password),
            _redact(exc.stderr, secret.password),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (1, "", _redact(str(exc), secret.password))
    return (
        0,
        _redact(result.stdout, secret.password),
        _redact(result.stderr, secret.password),
    )


def _quote_identifier(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def _quote_literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _generate_secrets(
    stack_layout: RuntimeStackLayout, volumes: StorageVolumes
) -> StorageSecrets:
    return StorageSecrets(
        stack=stack_layout.stack_name,
        postgres=PostgresSecret(
            user=MANAGED_POSTGRES_USER,
            database=MANAGED_POSTGRES_DATABASE,
            password=token_urlsafe(SECRET_TOKEN_BYTES),
            volume=volumes.postgres,
        ),
        redis=RedisSecret(
            password=token_urlsafe(SECRET_TOKEN_BYTES),
            volume=volumes.redis,
        ),
    )


def _state_document(secrets: StorageSecrets) -> dict[str, object]:
    return {
        "schema": STORAGE_SECRETS_SCHEMA,
        "stack": secrets.stack,
        "postgres": {
            "user": secrets.postgres.user,
            "database": secrets.postgres.database,
            "password": secrets.postgres.password,
            "volume": secrets.postgres.volume,
        },
        "redis": {
            "password": secrets.redis.password,
            "volume": secrets.redis.volume,
        },
    }


def _write_state(path: Path, secrets: StorageSecrets) -> Path:
    encoded = (
        json.dumps(_state_document(secrets), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return write_private_state_bytes(path, encoded)


def _read_state(path: Path, layout: RuntimeStackLayout) -> StorageSecrets | None:
    try:
        data = read_private_state_bytes(path)
    except StorageSecretError:
        raise
    except ValueError as error:
        raise StorageSecretError(
            f"storage credential state is unusable: {error}. {RECOVERY_HINT}"
        ) from error
    if data is None:
        return None
    return _decode_state(data, stack_name=layout.stack_name, path=path)


def _decode_state(data: bytes, *, stack_name: str, path: Path) -> StorageSecrets:
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"storage credential state is not valid JSON ({error})", path)
    if not isinstance(document, dict):
        _fail("storage credential state is not an object", path)
    # Schema first: a file a newer CLI wrote will also fail the field-set check
    # below, and "I do not understand this version" is the actionable half of
    # that answer -- it says upgrade, where a shape error says reprovision.
    if document.get("schema") != STORAGE_SECRETS_SCHEMA:
        _fail(
            f"storage credential state does not use schema {STORAGE_SECRETS_SCHEMA}",
            path,
        )
    if set(document) != {"schema", "stack", "postgres", "redis"}:
        _fail("storage credential state has an invalid field set", path)

    recorded_stack = _required_text(document, "stack", "state", path)
    if recorded_stack != stack_name:
        _fail(
            f"storage credential state belongs to stack {recorded_stack}, "
            f"not {stack_name}",
            path,
        )

    postgres = _required_section(document, "postgres", path)
    if set(postgres) != {"user", "database", "password", "volume"}:
        _fail("storage credential state postgres section has unexpected fields", path)
    redis = _required_section(document, "redis", path)
    if set(redis) != {"password", "volume"}:
        _fail("storage credential state redis section has unexpected fields", path)

    return StorageSecrets(
        stack=recorded_stack,
        postgres=PostgresSecret(
            user=_required_text(postgres, "user", "postgres", path),
            database=_required_text(postgres, "database", "postgres", path),
            password=_required_text(postgres, "password", "postgres", path),
            volume=_required_text(postgres, "volume", "postgres", path),
        ),
        redis=RedisSecret(
            password=_required_text(redis, "password", "redis", path),
            volume=_required_text(redis, "volume", "redis", path),
        ),
    )


def _required_section(
    document: dict[str, object], name: str, path: Path
) -> dict[str, object]:
    section = document.get(name)
    if not isinstance(section, dict):
        _fail(f"storage credential state section {name} is not an object", path)
    return section


def _required_text(
    section: dict[str, object], name: str, where: str, path: Path
) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        _fail(f"storage credential state field {where}.{name} is empty", path)
    return value
