"""Container service and observability helpers for vLLM Semantic Router."""

import json
import os
import socket
import subprocess

from cli import container_log_io as log_io
from cli.container_mounts import (
    ContainerMountsUnavailableError,
    inspect_container_mounts,
)
from cli.container_observability import _ensure_hidden_config_dir, _run_service_start
from cli.container_observability import (
    render_observability_template as _render_observability_template,
)
from cli.container_runtime import get_container_runtime
from cli.recipe_topology_storage import validate_storage_port_isolation
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack
from cli.storage_secrets import (
    CONTAINER_POSTGRES_PASSWORD_PATH,
    CONTAINER_REDIS_CONF_PATH,
    MANAGED_POSTGRES_DATABASE,
    MANAGED_POSTGRES_USER,
    POSTGRES_DATA_MOUNT_PATH,
    REDIS_DATA_MOUNT_PATH,
    StorageSecretError,
    adopted_volume_name,
)
from cli.utils import get_logger

log = get_logger(__name__)
render_observability_template = _render_observability_template


def container_status(container_name):
    """
    Get the status of a container.

    Returns:
        'running', 'created', 'exited', 'paused', or 'not found'
    """
    runtime = get_container_runtime()
    try:
        result = subprocess.run(
            [
                runtime,
                "inspect",
                "--format",
                "{{.State.Status}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return "not found"
        status = result.stdout.strip().lower()
        if status in {"running", "created", "exited", "paused"}:
            return status
        return status or "unknown"
    except Exception as exc:
        log.error(f"Failed to get container status: {exc}")
        return "error"


def container_status_strict(container_name: str) -> str:
    """Return one exact state or fail when absence cannot be proven.

    Activation recovery is destructive transaction work, so it must not use
    the compatibility helper above that historically treats every inspect
    failure as an absent container.
    """

    runtime = get_container_runtime()
    try:
        result = subprocess.run(
            [
                runtime,
                "inspect",
                "--format",
                "{{.State.Status}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("managed container status inspection failed") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).lower()
        if any(
            marker in detail
            for marker in (
                "no such container",
                "no such object",
                "no container with name or id",
            )
        ):
            return "not found"
        raise RuntimeError("managed container status inspection failed")
    status = result.stdout.strip().lower()
    if status not in {
        "created",
        "restarting",
        "running",
        "removing",
        "paused",
        "exited",
        "dead",
    }:
        raise RuntimeError("managed container status inspection is invalid")
    return status


def container_stop_container(container_name):
    """Stop a container."""
    runtime = get_container_runtime()
    try:
        log.info(f"Stopping container: {container_name}")
        subprocess.run(
            [runtime, "stop", container_name], check=True, capture_output=True
        )
        log.info(f"Container stopped: {container_name}")
        return True
    except subprocess.CalledProcessError as exc:
        log.error(f"Failed to stop container: {exc}")
        return False


def container_remove_container(container_name):
    """Remove a container."""
    runtime = get_container_runtime()
    try:
        log.info(f"Removing container: {container_name}")
        subprocess.run([runtime, "rm", container_name], check=True, capture_output=True)
        log.info(f"Container removed: {container_name}")
        return True
    except subprocess.CalledProcessError as exc:
        log.error(f"Failed to remove container: {exc}")
        return False


def container_logs(container_name, follow=False, tail=None, *, merge_output=False):
    """Stream logs from a container and report whether the command succeeded."""
    return log_io.stream_container_logs(
        get_container_runtime(),
        container_name,
        follow=follow,
        tail=tail,
        merge_output=merge_output,
        run=subprocess.run,
        logger=log,
    )


def container_logs_output(container_name, tail=None):
    """Capture a bounded container log snapshot with stderr merged in order."""
    return log_io.capture_container_logs(
        get_container_runtime(), container_name, tail=tail, run=subprocess.run
    )


def container_logs_since(container_name, since_timestamp):
    """Get logs from a container since a specific timestamp."""
    return log_io.capture_container_logs_since(
        get_container_runtime(),
        container_name,
        since_timestamp,
        run=subprocess.run,
    )


def container_exec(container_name, command):
    """Execute a command in a running container."""
    runtime = get_container_runtime()
    cmd = [runtime, "exec", container_name, *command]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def container_create_network(network_name):
    """Create a Docker network if it doesn't exist."""
    runtime = get_container_runtime()
    cmd = [
        runtime,
        "network",
        "ls",
        "--filter",
        f"name={network_name}",
        "--format",
        "{{.Name}}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        existing_networks = {
            line.strip() for line in result.stdout.splitlines() if line.strip()
        }
        if network_name in existing_networks:
            log.debug(f"Network {network_name} already exists")
            return (0, "", "")
    except subprocess.CalledProcessError:
        pass

    cmd = [runtime, "network", "create", network_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.info(f"Created network: {network_name}")
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def container_remove_network(network_name):
    """Remove a Docker network."""
    runtime = get_container_runtime()
    cmd = [runtime, "network", "rm", network_name]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


def container_start_redis(
    network_name=None,
    stack_layout: RuntimeStackLayout | None = None,
    *,
    reuse_existing: bool = True,
    recreate: bool = False,
    redis_conf_file: str,
    data_volume: str | None = None,
):
    """Start a Redis container for durable storage backends.

    Reuses an already-running container to preserve data across router restarts.
    Pass *recreate* to rebuild a running container anyway, which is what a
    credential change requires: the password lives in the mounted config file,
    and a running Redis keeps serving the config it started with.

    *redis_conf_file* is a host path holding this stack's ``requirepass`` line.
    It is bind mounted instead of being passed as ``redis-server
    --requirepass``: a container process is a host process, and
    ``/proc/<pid>/cmdline`` is world readable, so the argv form would publish
    the password to every user on the host. It has no default on purpose:
    starting this stack's Redis without authentication is not a mode.
    """
    runtime = get_container_runtime()
    stack_layout = stack_layout or resolve_runtime_stack()
    container_name = stack_layout.redis_container_name
    # Storage defaults to the data network, never the application network:
    # observability sidecars and user-selected OpenClaw workloads join the
    # latter, and reaching a storage port from one of them is the east-west half
    # of the exposure that loopback-only publication closes from the host side.
    network_name = network_name or stack_layout.data_network_name

    adopted_volume = None
    if reuse_existing and not recreate:
        reuse_result = _reuse_running_storage_container(
            container_name, "Redis", network_name, stack_layout.network_name
        )
        if reuse_result is not None:
            return reuse_result
    if reuse_existing or recreate:
        adopted_volume = _replace_existing_container(
            container_name, adopt_volume_destination=REDIS_DATA_MOUNT_PATH
        )

    if _is_port_in_use(stack_layout.redis_port):
        return _storage_port_conflict_result(
            "Redis", stack_layout.redis_port, container_name
        )

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--hostname",
        "redis",
        "--network",
        network_name,
        "-p",
        f"127.0.0.1:{stack_layout.redis_port}:6379",
    ]
    cmd += [
        "-v",
        f"{os.path.abspath(redis_conf_file)}:{CONTAINER_REDIS_CONF_PATH}:ro,z",
    ]
    volume = data_volume or adopted_volume
    if volume:
        cmd += ["-v", f"{volume}:{REDIS_DATA_MOUNT_PATH}"]
    # Pinned deliberately: `redis:7-alpine` still declares VOLUME /data, while
    # newer tags (`redis:8`, `redis:alpine`) dropped the declaration. Bumping
    # this tag would make the anonymous volume of an unadopted stack vanish and
    # take its data with it, so a bump has to come with a migration that has
    # already recorded a named volume for every managed stack.
    cmd.append("docker.io/library/redis:7-alpine")
    cmd += ["redis-server", CONTAINER_REDIS_CONF_PATH]
    return _run_service_start(cmd, "Redis")


def container_start_postgres(
    network_name=None,
    stack_layout: RuntimeStackLayout | None = None,
    *,
    reuse_existing: bool = True,
    recreate: bool = False,
    postgres_password_file: str,
    data_volume: str | None = None,
):
    """Start a Postgres container for durable storage backends.

    Reuses an already-running container to preserve data across router restarts.

    *postgres_password_file* is a host path holding this stack's password. It
    is handed over as ``POSTGRES_PASSWORD_FILE``, whose value is the
    *container* path the official image reads through its ``file_env``
    convention, so the value itself never reaches the argv list nor
    ``docker inspect .Config.Env`` -- which matters because Dashboard holds the
    container runtime socket and can read the latter. It has no default on
    purpose: there is no credential-less way to start this stack's Postgres.
    """
    runtime = get_container_runtime()
    stack_layout = stack_layout or resolve_runtime_stack()
    container_name = stack_layout.postgres_container_name
    # The data network, for the reason spelled out in `container_start_redis`.
    network_name = network_name or stack_layout.data_network_name

    adopted_volume = None
    if reuse_existing and not recreate:
        reuse_result = _reuse_running_storage_container(
            container_name, "Postgres", network_name, stack_layout.network_name
        )
        if reuse_result is not None:
            return reuse_result
    if reuse_existing or recreate:
        adopted_volume = _replace_existing_container(
            container_name, adopt_volume_destination=POSTGRES_DATA_MOUNT_PATH
        )

    if _is_port_in_use(stack_layout.postgres_port):
        return _storage_port_conflict_result(
            "Postgres", stack_layout.postgres_port, container_name
        )

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--hostname",
        "postgres",
        "--network",
        network_name,
        "-e",
        f"POSTGRES_DB={MANAGED_POSTGRES_DATABASE}",
        "-e",
        f"POSTGRES_USER={MANAGED_POSTGRES_USER}",
    ]
    cmd += [
        "-e",
        f"POSTGRES_PASSWORD_FILE={CONTAINER_POSTGRES_PASSWORD_PATH}",
        "-v",
        (
            f"{os.path.abspath(postgres_password_file)}:"
            f"{CONTAINER_POSTGRES_PASSWORD_PATH}:ro,z"
        ),
    ]
    volume = data_volume or adopted_volume
    if volume:
        cmd += ["-v", f"{volume}:{POSTGRES_DATA_MOUNT_PATH}"]
    cmd += ["-p", f"127.0.0.1:{stack_layout.postgres_port}:5432"]
    # Pinned deliberately, for the same reason as the Redis tag above:
    # `postgres:16-alpine` declares VOLUME /var/lib/postgresql/data, and a stack
    # that has not been adopted yet keeps its only copy of the database in the
    # anonymous volume that declaration creates. A major bump also changes the
    # on-disk data directory format, which a recreated container cannot read.
    cmd.append("docker.io/library/postgres:16-alpine")
    return _run_service_start(cmd, "Postgres")


def container_start_milvus(
    network_name=None,
    stack_layout: RuntimeStackLayout | None = None,
    *,
    state_root_dir: str | None = None,
    host_hidden_state_dir: str | None = None,
    reuse_existing: bool = True,
):
    """Start a Milvus container for the semantic cache backend.

    Reuses an already-running container to preserve data across router restarts.
    """
    runtime = get_container_runtime()
    stack_layout = stack_layout or resolve_runtime_stack()
    container_name = stack_layout.milvus_container_name
    # The data network, for the reason spelled out in `container_start_redis`.
    # Milvus joins it even though it has no credentials of its own yet; those
    # are separate work, and network reachability is what this closes.
    network_name = network_name or stack_layout.data_network_name

    if reuse_existing:
        reuse_result = _reuse_running_storage_container(
            container_name, "Milvus", network_name, stack_layout.network_name
        )
        if reuse_result is not None:
            return reuse_result
        reuse_result = _reuse_running_storage_network_alias(
            runtime, container_name, "Milvus", network_name
        )
        if reuse_result is not None:
            return reuse_result
        _replace_existing_container(container_name)

    if _is_port_in_use(stack_layout.milvus_port):
        return _storage_port_conflict_result(
            "Milvus", stack_layout.milvus_port, container_name
        )

    config_dir = _ensure_hidden_config_dir(state_root_dir)
    milvus_data_dir = os.path.join(config_dir, "milvus-data")
    os.makedirs(milvus_data_dir, exist_ok=True)
    mount_data_dir = milvus_data_dir
    if host_hidden_state_dir is not None:
        if not os.path.isabs(host_hidden_state_dir):
            raise ValueError("Milvus host state directory must be absolute")
        mount_data_dir = os.path.join(host_hidden_state_dir, "milvus-data")

    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        container_name,
        "--hostname",
        "milvus",
        "--network",
        network_name,
        "--security-opt",
        "seccomp:unconfined",
        "-e",
        "ETCD_USE_EMBED=true",
        "-e",
        "ETCD_DATA_DIR=/var/lib/milvus/etcd",
        "-e",
        "ETCD_CONFIG_PATH=/milvus/configs/advanced/etcd.yaml",
        "-e",
        "COMMON_STORAGETYPE=local",
        "-e",
        "CLUSTER_ENABLED=false",
        "-p",
        f"127.0.0.1:{stack_layout.milvus_port}:19530",
        "-v",
        f"{os.path.abspath(mount_data_dir)}:/var/lib/milvus:z",
        "docker.io/milvusdb/milvus:v2.3.3",
        "milvus",
        "run",
        "standalone",
    ]
    return _run_service_start(cmd, "Milvus")


def container_network_disconnect(network_name, container_name):
    """Disconnect a container from a Docker network."""
    runtime = get_container_runtime()
    cmd = [runtime, "network", "disconnect", network_name, container_name]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=10
        )
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        return (exc.returncode, exc.stdout, exc.stderr)


# Substrings that mean "already detached", across runtimes and versions.
# Short enough to survive wording drift -- Docker says "is not connected to
# network X" on one code path and "is not connected to the network X" on
# another, and Podman words the missing-network case differently again -- but
# each one still names what it matches. A bare "not found" would also swallow
# unrelated daemon errors, and this call is the one that has to fail loudly:
# reporting a disconnect that never happened reports an isolation that is not
# there. Verified wordings: Docker "... is not connected to network X" and
# "network X not found"; Podman "unable to find network".
DETACHED_NETWORK_MARKERS = (
    "not connected",
    "no such network",
    "network not found",
    "unable to find network",
)


def container_network_disconnect_if_attached(network_name, container_name):
    """Disconnect a container from a network it may already have left.

    A container that never joined *network_name*, or a network that no longer
    exists, is already the intended end state, so both are reported as success
    and quietly: they are the steady state once a stack has been migrated.
    Every other failure is passed through unchanged, because this call is how a
    stack provisioned before the storage network split gives up its reach into
    the application network, and swallowing a real error there would report an
    isolation that was never applied.
    """
    return_code, stdout, stderr = container_network_disconnect(
        network_name, container_name
    )
    if return_code == 0:
        return (0, stdout, stderr)
    detail = (stderr or "").lower()
    if any(marker in detail for marker in DETACHED_NETWORK_MARKERS):
        log.debug(f"{container_name} is already detached from {network_name}")
        return (0, stdout, stderr)
    return (return_code, stdout, stderr)


def container_network_connect(network_name, container_name):
    """Connect a container to a Docker network (idempotent)."""
    runtime = get_container_runtime()
    cmd = [runtime, "network", "connect", network_name, container_name]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=10
        )
        return (0, result.stdout, result.stderr)
    except subprocess.CalledProcessError as exc:
        if "already" in (exc.stderr or "").lower():
            return (0, exc.stdout, exc.stderr)
        return (exc.returncode, exc.stdout, exc.stderr)


def container_start_container(container_name):
    """Start a stopped container."""
    runtime = get_container_runtime()
    try:
        log.info(f"Starting container: {container_name}")
        subprocess.run(
            [runtime, "start", container_name], check=True, capture_output=True
        )
        log.info(f"✓ Container started: {container_name}")
        return True
    except subprocess.CalledProcessError as exc:
        log.error(f"Failed to start container: {exc}")
        return False


def load_openclaw_registry(data_dir):
    """Load OpenClaw container entries from containers.json."""
    registry_path = os.path.join(data_dir, "containers.json")
    if not os.path.exists(registry_path):
        return []
    try:
        with open(registry_path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(f"Failed to load OpenClaw registry: {exc}")
        return []


def _reuse_running_storage_container(
    container_name: str,
    label: str,
    data_network_name: str | None,
    app_network_name: str | None,
) -> tuple[int, str, str] | None:
    """Return a success/failure tuple when a running storage container is reused.

    Reuse is also the migration step. A stack provisioned before storage moved
    off the application network still has its containers attached there, and
    reuse is the only path they take on a later `serve` -- they are never
    rebuilt, so nothing else would ever move them. Connecting the data network
    without leaving the application network would keep every such stack
    reachable from its own sidecars for good. Connect comes first so a failure
    leaves the container on the network it already had rather than on none.
    """
    status = container_status(container_name)
    if status == "running":
        safe, detail = _storage_ports_are_loopback_only(container_name)
        if not safe:
            return (
                1,
                "",
                (
                    f"{label} container {container_name} has unsafe published storage "
                    f"ports ({detail}). Preserve its data mounts and recreate it with "
                    "127.0.0.1 bindings before serving; automatic reuse is disabled."
                ),
            )
        log.info(f"{label} container already running, reusing to preserve data")
        if data_network_name:
            return_code, stdout, stderr = container_network_connect(
                data_network_name, container_name
            )
            if return_code != 0:
                return return_code, stdout, stderr
        if app_network_name and app_network_name != data_network_name:
            return_code, stdout, stderr = container_network_disconnect_if_attached(
                app_network_name, container_name
            )
            if return_code != 0:
                return return_code, stdout, stderr
        return 0, "", ""
    return None


def _storage_ports_are_loopback_only(container_name: str) -> tuple[bool, str]:
    runtime = get_container_runtime()
    try:
        result = subprocess.run(
            [
                runtime,
                "inspect",
                "--format",
                (
                    '{"network_mode":{{json .HostConfig.NetworkMode}},'
                    '"publish_all_ports":{{json .HostConfig.PublishAllPorts}},'
                    '"configured":{{json .HostConfig.PortBindings}},'
                    '"actual":{{json .NetworkSettings.Ports}}}'
                ),
                container_name,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        inspection = json.loads(result.stdout or "")
        validate_storage_port_isolation(inspection)
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        return False, f"inspection failed: {exc}"
    return True, ""


def _reuse_running_storage_network_alias(
    runtime: str, alias: str, label: str, network_name: str | None
) -> tuple[int, str, str] | None:
    """Return success when a running storage service already owns the alias."""
    container_name = _running_container_for_network_alias(runtime, network_name, alias)
    if not container_name:
        return None

    safe, detail = _storage_ports_are_loopback_only(container_name)
    if not safe:
        return (
            1,
            "",
            (
                f"{label} container {container_name} has unsafe published storage "
                f"ports ({detail}). Preserve its data mounts and recreate it with "
                "127.0.0.1 bindings before serving; automatic reuse is disabled."
            ),
        )

    log.info(
        f"{label} service already attached to {network_name} as {alias} "
        f"via container {container_name}, reusing"
    )
    return 0, "", ""


def _running_container_for_network_alias(
    runtime: str, network_name: str | None, alias: str
) -> str | None:
    if not network_name:
        return None

    try:
        result = subprocess.run(
            [
                runtime,
                "network",
                "inspect",
                network_name,
                "--format",
                "{{json .Containers}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    try:
        containers = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None

    if not isinstance(containers, dict):
        return None

    for metadata in containers.values():
        if not isinstance(metadata, dict):
            continue
        name = str(metadata.get("Name") or "").lstrip("/")
        aliases = metadata.get("Aliases") or []
        if name == alias or alias in aliases:
            return name or alias

    return None


def _storage_port_conflict_result(
    label: str, port: int, container_name: str
) -> tuple[int, str, str]:
    message = (
        f"{label} port {port} is already in use, but {container_name} is not a "
        f"running reusable container. Stop the process using port {port}, set a "
        "different VLLM_SR_PORT_OFFSET, or point the config at an explicitly "
        f"reachable external {label} service."
    )
    log.error(message)
    return 1, "", message


def _is_port_in_use(port: int) -> bool:
    """Return True if a TCP port is already bound on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _replace_existing_container(container_name, *, adopt_volume_destination=None):
    """Remove an existing container, first recording the volume it mounts.

    ``docker rm`` keeps the volume alive but destroys the only record of which
    volume belonged to which container, so the inspection has to happen while
    the container still exists -- afterwards an anonymous volume is just an
    unlabelled hex id nobody can attribute. Returns the volume mounted at
    *adopt_volume_destination*, or ``None`` when there is nothing to adopt.

    Removal goes through ``docker stop`` rather than ``rm -f`` so the image's
    own shutdown handling runs; for Redis that is the SIGTERM save point that
    flushes the RDB file.

    An empty ``.Mounts`` is a normal answer, not a failure: an image that
    declares no ``VOLUME`` keeps its data in the container layer, and the
    caller then simply creates a fresh named volume.
    """
    status = container_status(container_name)
    if status == "not found":
        return None
    log.info(f"{container_name} already exists (status: {status}), cleaning up...")
    adopted = None
    if adopt_volume_destination:
        try:
            adopted = adopted_volume_name(container_name, adopt_volume_destination)
        except StorageSecretError as exc:
            # Report and keep going: this helper answers with a tuple, and the
            # caller's own credential resolution already carries the strict
            # inspection that fails closed when the runtime cannot answer.
            log.warning(f"Cannot read the data volume of {container_name}: {exc}")
    container_stop_container(container_name)
    container_remove_container(container_name)
    return adopted


def container_mount_destinations(container_name):
    """Return the container-side paths *container_name* mounts.

    ``None`` means the runtime could not answer, which callers must not read as
    "mounts nothing": the difference decides whether a container is safe to
    remove.
    """
    try:
        mounts = inspect_container_mounts(container_name)
    except ContainerMountsUnavailableError as exc:
        log.warning(f"Cannot inspect the mounts of {container_name}: {exc}")
        return None
    return {
        str(mount.get("Destination")) for mount in mounts if mount.get("Destination")
    }
