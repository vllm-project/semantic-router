"""Standalone ``vllm-sr drun`` lifecycle orchestration."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import MappingProxyType

from cli.consts import (
    HEALTH_CHECK_TIMEOUT,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    SUPPORTED_CONTAINER_RUNTIMES,
)
from cli.decision_runtime.catalog import (
    RESOLVED_DECISION_BACKENDS,
    SUPPORTED_DECISION_BACKENDS,
    DecisionCatalogResolver,
    DecisionRuntimeMount,
    DecisionRuntimeRequest,
    ResolvedDecisionRuntime,
)
from cli.decision_runtime.container import (
    DecisionContainerDriver,
    DecisionContainerIdentity,
    DecisionContainerLaunch,
    DecisionContainerObservation,
    DecisionContainerOwnershipError,
    LowLevelDecisionContainerDriver,
    select_container_runtime,
    validate_decision_environment,
    validate_decision_mount,
)
from cli.decision_runtime.image_reference import (
    ImmutableImageReferenceError,
    validate_immutable_image_reference,
)
from cli.decision_runtime.registry import (
    DecisionInstanceRecord,
    DecisionInstanceRegistry,
    DecisionRegistryError,
    DecisionRegistryStaleRecordError,
)

_INSTANCE_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9_.-]{0,62}[a-z0-9])?$")
_ARTIFACT_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
_ASCII_CONTROL_THRESHOLD = 32
_VALID_PULL_POLICIES = frozenset(
    {
        IMAGE_PULL_POLICY_ALWAYS,
        IMAGE_PULL_POLICY_IF_NOT_PRESENT,
        IMAGE_PULL_POLICY_NEVER,
    }
)


class DecisionLifecycleError(RuntimeError):
    """The requested standalone Decision runtime could not be launched."""


@dataclass(frozen=True)
class DrunOptions:
    """Validated public options for one ``drun`` launch."""

    model: str
    revision: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    backend: str = "auto"
    dtype: str | None = None
    max_batch: int | None = None
    max_concurrency: int | None = None
    max_queue: int | None = None
    instance_name: str | None = None
    image: str | None = None
    image_pull_policy: str = IMAGE_PULL_POLICY_IF_NOT_PRESENT
    runtime: str | None = None
    startup_timeout: int = HEALTH_CHECK_TIMEOUT
    detach: bool = False


@dataclass(frozen=True)
class DecisionLaunchReceipt:
    """Public, secret-free launch result."""

    instance_name: str
    model: str
    revision: str
    endpoint: str
    backend: str
    dtype: str
    artifact_digest: str
    detached: bool


def run_decision_runtime(
    options: DrunOptions,
    *,
    resolver: DecisionCatalogResolver,
    registry: DecisionInstanceRegistry | None = None,
    driver: DecisionContainerDriver | None = None,
    runtime_selector: Callable[[str | None], str] | None = None,
    on_ready: Callable[[DecisionLaunchReceipt], None] | None = None,
) -> DecisionLaunchReceipt:
    """Resolve, reserve, launch, and supervise one standalone model runtime."""

    _validate_options(options)
    spec = _resolve_runtime(options, resolver)
    runtime = (runtime_selector or select_container_runtime)(options.runtime)
    if runtime not in SUPPORTED_CONTAINER_RUNTIMES:
        raise DecisionLifecycleError(
            "Container runtime selection returned an unsupported runtime."
        )
    host = _normalize_host(options.host)
    instance_name = options.instance_name or _default_instance_name(
        spec.canonical_model, spec.revision, options.port
    )
    _validate_instance_name(instance_name)
    endpoint = _host_url(host, options.port, spec.api_path)
    instance_id = str(uuid.uuid4())
    identity_digest = _identity_digest(
        instance_id=instance_id,
        instance_name=instance_name,
        endpoint=endpoint,
        runtime=runtime,
        spec=spec,
    )
    container_name = f"vllm-sr-drun-{instance_name}"
    launch = DecisionContainerLaunch(
        runtime=runtime,
        container_name=container_name,
        instance_name=instance_name,
        identity_digest=identity_digest,
        host=host,
        port=options.port,
        pull_policy=options.image_pull_policy,
        runtime_spec=spec,
    )
    record = DecisionInstanceRecord(
        instance_id=instance_id,
        instance_name=instance_name,
        container_name=container_name,
        model=spec.canonical_model,
        revision=spec.revision,
        endpoint=endpoint,
        backend=spec.backend,
        dtype=spec.dtype,
        artifact_digest=spec.artifact_digest,
        identity_digest=identity_digest,
        image=spec.image,
        container_id=None,
        runtime=runtime,
        max_batch=spec.max_batch,
        max_concurrency=spec.max_concurrency,
        max_queue=spec.max_queue,
        state="starting",
        generation=0,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    instance_registry = registry or DecisionInstanceRegistry()
    container_driver = driver or LowLevelDecisionContainerDriver()
    reservation = instance_registry.reserve(record)
    if reservation.record is None:  # pragma: no cover - reserve invariant.
        raise DecisionRegistryError("Decision runtime registry is invalid.")
    registry_record = reservation.record
    _require_durable_registry_commit(reservation.durable, "reservation")
    ownership: DecisionContainerIdentity | None = None
    ownership_uncertain = False
    try:
        container_driver.ensure_image(launch)
        try:
            observation = container_driver.start(launch)
            ownership = _validated_launch_ownership(launch, observation)
        except BaseException:
            # Once creation was dispatched, failure does not prove that no
            # container exists. Retain the reservation for label-backed
            # reconciliation; cleanup must never guess by reusable name.
            ownership_uncertain = True
            raise
        identity_commit = instance_registry.transition(
            instance_name,
            instance_id,
            expected_generation=registry_record.generation,
            expected_state=registry_record.state,
            state="starting",
            container_id=ownership.container_id,
        )
        if identity_commit.record is None:  # pragma: no cover - transition invariant.
            raise DecisionRegistryError("Decision runtime registry is invalid.")
        registry_record = identity_commit.record
        _require_durable_registry_commit(identity_commit.durable, "container identity")
        container_driver.wait_ready(
            launch,
            ownership,
            startup_timeout=options.startup_timeout,
        )
        running_commit = instance_registry.transition(
            instance_name,
            instance_id,
            expected_generation=registry_record.generation,
            expected_state=registry_record.state,
            state="running",
            container_id=ownership.container_id,
        )
        if running_commit.record is None:  # pragma: no cover - transition invariant.
            raise DecisionRegistryError("Decision runtime registry is invalid.")
        registry_record = running_commit.record
        _require_durable_registry_commit(running_commit.durable, "running state")
        receipt = DecisionLaunchReceipt(
            instance_name=instance_name,
            model=spec.canonical_model,
            revision=spec.revision,
            endpoint=endpoint,
            backend=spec.backend,
            dtype=spec.dtype,
            artifact_digest=spec.artifact_digest,
            detached=options.detach,
        )
        if on_ready is not None:
            on_ready(receipt)
        if options.detach:
            return receipt
        try:
            container_driver.follow_logs(ownership)
        finally:
            container_driver.stop_and_remove(ownership)
            ownership = None
            instance_registry.remove(
                instance_name,
                instance_id,
                expected_generation=registry_record.generation,
                expected_state=registry_record.state,
            )
        return receipt
    except BaseException:
        _rollback_launch(
            registry=instance_registry,
            driver=container_driver,
            launch=launch,
            record=registry_record,
            ownership=ownership,
            ownership_uncertain=ownership_uncertain,
        )
        raise


def _resolve_runtime(
    options: DrunOptions,
    resolver: DecisionCatalogResolver,
) -> ResolvedDecisionRuntime:
    """Resolve and validate a container-compatible runtime description."""

    request = DecisionRuntimeRequest(
        model=options.model,
        revision=options.revision,
        backend=options.backend,
        dtype=options.dtype,
        image=options.image,
        max_batch=options.max_batch,
        max_concurrency=options.max_concurrency,
        max_queue=options.max_queue,
    )
    spec = resolver.resolve(request)
    if not isinstance(spec, ResolvedDecisionRuntime):
        raise DecisionLifecycleError(
            "Decision runtime resolver returned an invalid launch description."
        )
    spec = _freeze_resolved_runtime(spec)
    _validate_resolved_runtime(request, spec)
    if spec.backend == "mlx":
        raise DecisionLifecycleError(
            "The MLX backend requires the native Decision runtime driver, which is "
            "not included in this container lifecycle foundation yet."
        )
    return spec


def _rollback_launch(
    *,
    registry: DecisionInstanceRegistry,
    driver: DecisionContainerDriver,
    launch: DecisionContainerLaunch,
    record: DecisionInstanceRecord,
    ownership: DecisionContainerIdentity | None,
    ownership_uncertain: bool,
) -> None:
    cleanup_failed = False
    if ownership is not None:
        try:
            driver.stop_and_remove(ownership)
        except Exception:
            cleanup_failed = True
    try:
        if cleanup_failed or ownership_uncertain:
            registry.transition(
                launch.instance_name,
                record.instance_id,
                expected_generation=record.generation,
                expected_state=record.state,
                state="cleanup-required",
                container_id=(
                    ownership.container_id if ownership is not None else None
                ),
            )
        else:
            registry.remove(
                launch.instance_name,
                record.instance_id,
                expected_generation=record.generation,
                expected_state=record.state,
            )
    except DecisionRegistryStaleRecordError:
        pass


def _require_durable_registry_commit(durable: bool, description: str) -> None:
    if not durable:
        raise DecisionLifecycleError(
            f"Decision runtime {description} was committed, but crash durability "
            "could not be confirmed. No detached runtime will be reported ready."
        )


def _validated_launch_ownership(
    launch: DecisionContainerLaunch,
    observation: object,
) -> DecisionContainerIdentity:
    """Reject an untrusted driver receipt unless it exactly proves this launch."""

    identity = getattr(observation, "identity", None)
    state = getattr(observation, "state", None)
    expected = launch.identity()
    if (
        not isinstance(observation, DecisionContainerObservation)
        or not isinstance(identity, DecisionContainerIdentity)
        or identity.runtime != expected.runtime
        or identity.container_name != expected.container_name
        or identity.instance_name != expected.instance_name
        or identity.identity_digest != expected.identity_digest
        or identity.image != expected.image
        or not isinstance(identity.container_id, str)
        or not _CONTAINER_ID.fullmatch(identity.container_id)
        or state
        not in {
            "created",
            "restarting",
            "running",
            "removing",
            "paused",
            "exited",
            "dead",
        }
    ):
        raise DecisionContainerOwnershipError(
            "Decision runtime driver did not return a full, matching ownership receipt."
        )
    return identity


def _validate_options(options: DrunOptions) -> None:
    if not isinstance(options.model, str) or not options.model.strip():
        raise DecisionLifecycleError("MODEL must be an exact non-empty model ID.")
    if options.model != options.model.strip():
        raise DecisionLifecycleError("MODEL must not contain surrounding whitespace.")
    if options.backend not in SUPPORTED_DECISION_BACKENDS:
        raise DecisionLifecycleError(
            f"Unsupported backend {options.backend!r}; choose "
            f"{', '.join(SUPPORTED_DECISION_BACKENDS)}."
        )
    _validate_host(options.host)
    _positive_integer("port", options.port, maximum=65535)
    _positive_integer("startup timeout", options.startup_timeout)
    for label, value in (
        ("max batch", options.max_batch),
        ("max concurrency", options.max_concurrency),
    ):
        if value is not None:
            _positive_integer(label, value)
    if options.max_queue is not None:
        _nonnegative_integer("max queue", options.max_queue)
    if options.image_pull_policy not in _VALID_PULL_POLICIES:
        raise DecisionLifecycleError(
            f"Unsupported image pull policy {options.image_pull_policy!r}."
        )
    if options.instance_name is not None:
        _validate_instance_name(options.instance_name)
    for label, value in (
        ("revision", options.revision),
        ("dtype", options.dtype),
        ("image", options.image),
    ):
        if value is not None and (
            not isinstance(value, str) or not value or value != value.strip()
        ):
            raise DecisionLifecycleError(
                f"{label} must be a non-empty value without surrounding whitespace."
            )
    if options.image is not None:
        try:
            validate_immutable_image_reference(options.image)
        except ImmutableImageReferenceError as error:
            raise DecisionLifecycleError(
                f"Decision runtime image override is invalid: {error}."
            ) from error


def _validate_resolved_runtime(
    request: DecisionRuntimeRequest, spec: ResolvedDecisionRuntime
) -> None:
    string_fields = {
        "canonical model": spec.canonical_model,
        "revision": spec.revision,
        "dtype": spec.dtype,
        "image": spec.image,
    }
    for label, value in string_fields.items():
        if not isinstance(value, str) or not value or value != value.strip():
            raise DecisionLifecycleError(
                f"Resolved Decision runtime {label} is invalid."
            )
    if spec.backend not in RESOLVED_DECISION_BACKENDS:
        raise DecisionLifecycleError(
            "Resolved Decision runtime backend must be rocm, cuda, or mlx."
        )
    try:
        validate_immutable_image_reference(spec.image)
    except ImmutableImageReferenceError as error:
        raise DecisionLifecycleError(
            f"Resolved Decision runtime image is invalid: {error}."
        ) from error
    if request.backend not in ("auto", spec.backend):
        raise DecisionLifecycleError(
            "Decision runtime resolver changed the explicitly requested backend."
        )
    if request.revision is not None and spec.revision != request.revision:
        raise DecisionLifecycleError(
            "Decision runtime resolver changed the explicitly requested revision."
        )
    if request.dtype is not None and spec.dtype != request.dtype:
        raise DecisionLifecycleError(
            "Decision runtime resolver changed the explicitly requested dtype."
        )
    if request.image is not None and spec.image != request.image:
        raise DecisionLifecycleError(
            "Decision runtime resolver changed the explicitly requested image."
        )
    for label, requested, resolved in (
        ("max batch", request.max_batch, spec.max_batch),
        ("max concurrency", request.max_concurrency, spec.max_concurrency),
    ):
        _positive_integer(f"resolved {label}", resolved)
        if requested is not None and resolved != requested:
            raise DecisionLifecycleError(
                f"Decision runtime resolver changed the explicitly requested {label}."
            )
    _nonnegative_integer("resolved max queue", spec.max_queue)
    if request.max_queue is not None and spec.max_queue != request.max_queue:
        raise DecisionLifecycleError(
            "Decision runtime resolver changed the explicitly requested max queue."
        )
    _positive_integer("resolved container port", spec.container_port, maximum=65535)
    if not _ARTIFACT_DIGEST.fullmatch(spec.artifact_digest):
        raise DecisionLifecycleError(
            "Resolved Decision runtime artifact digest must be "
            "sha256:<64 lowercase hex>."
        )
    for label, path in (("health", spec.health_path), ("API", spec.api_path)):
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path.startswith("//")
            or "?" in path
            or "#" in path
            or any(
                character.isspace() or ord(character) < _ASCII_CONTROL_THRESHOLD
                for character in path
            )
        ):
            raise DecisionLifecycleError(
                f"Resolved Decision runtime {label} path is invalid."
            )
    if not isinstance(spec.command, tuple) or any(
        not isinstance(part, str) or not part or "\0" in part for part in spec.command
    ):
        raise DecisionLifecycleError("Resolved Decision runtime command is invalid.")
    validate_decision_environment(spec.environment)
    mount_targets: set[str] = set()
    for mount in spec.mounts:
        _source, target = validate_decision_mount(mount)
        if target in mount_targets:
            raise DecisionLifecycleError(
                "Resolved Decision runtime mount targets must be unique."
            )
        mount_targets.add(target)


def _freeze_resolved_runtime(spec: ResolvedDecisionRuntime) -> ResolvedDecisionRuntime:
    """Snapshot mutable resolver-owned launch data before hashing or execution."""

    try:
        environment = dict(spec.environment.items())
    except (AttributeError, TypeError, ValueError) as error:
        raise DecisionLifecycleError(
            "Resolved Decision runtime environment is invalid."
        ) from error
    try:
        mounts = tuple(spec.mounts)
    except TypeError as error:
        raise DecisionLifecycleError(
            "Resolved Decision runtime mounts are invalid."
        ) from error
    if any(not isinstance(mount, DecisionRuntimeMount) for mount in mounts):
        raise DecisionLifecycleError("Resolved Decision runtime mounts are invalid.")
    return replace(
        spec,
        environment=MappingProxyType(environment),
        mounts=mounts,
    )


def _validate_host(host: str) -> None:
    _normalize_host(host)


def _normalize_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise DecisionLifecycleError(
            "host must be an IPv4 or IPv6 address; use 127.0.0.1 for loopback."
        ) from error
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)
    return address.compressed


def _validate_instance_name(value: str) -> None:
    if not isinstance(value, str) or not _INSTANCE_NAME.fullmatch(value):
        raise DecisionLifecycleError(
            "instance name must be 1-64 lowercase letters, digits, '.', '_' or '-', "
            "and must start and end with a letter or digit."
        )


def _positive_integer(label: str, value: object, *, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DecisionLifecycleError(f"{label} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise DecisionLifecycleError(f"{label} must not exceed {maximum}.")


def _nonnegative_integer(label: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DecisionLifecycleError(f"{label} must be a non-negative integer.")


def _default_instance_name(model: str, revision: str, port: int) -> str:
    model_slug = re.sub(r"[^a-z0-9]+", "-", model.rsplit("/", 1)[-1].lower()).strip("-")
    model_slug = (model_slug or "decision")[:40].rstrip("-")
    suffix = hashlib.sha256(f"{model}\0{revision}\0{port}".encode()).hexdigest()[:8]
    return f"{model_slug}-{port}-{suffix}"


def _identity_digest(
    *,
    instance_id: str,
    instance_name: str,
    endpoint: str,
    runtime: str,
    spec: ResolvedDecisionRuntime,
) -> str:
    payload = {
        "artifact_digest": spec.artifact_digest,
        "backend": spec.backend,
        "canonical_model": spec.canonical_model,
        "command": spec.command,
        "container_port": spec.container_port,
        "dtype": spec.dtype,
        "endpoint": endpoint,
        "environment": dict(sorted(spec.environment.items())),
        "health_path": spec.health_path,
        "image": spec.image,
        "instance_id": instance_id,
        "instance_name": instance_name,
        "max_batch": spec.max_batch,
        "max_concurrency": spec.max_concurrency,
        "max_queue": spec.max_queue,
        "mounts": [
            {"source": mount.source, "target": mount.target} for mount in spec.mounts
        ],
        "revision": spec.revision,
        "runtime": runtime,
        "systemone_path": spec.api_path,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _host_url(host: str, port: int, path: str) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    return f"http://{rendered_host}:{port}{path}"
