"""Safe reconciliation and public lifecycle operations for ``drun`` instances."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from cli.consts import SUPPORTED_CONTAINER_RUNTIMES
from cli.decision_runtime.container import (
    DecisionContainerCandidate,
    DecisionContainerDriver,
    DecisionContainerError,
    DecisionContainerIdentity,
    DecisionContainerObservation,
    DecisionContainerOwnershipError,
    LowLevelDecisionContainerDriver,
)
from cli.decision_runtime.registry import (
    DecisionInstanceRecord,
    DecisionInstanceRegistry,
    DecisionRegistryError,
    DecisionRegistryStaleRecordError,
)


class DecisionManagementError(RuntimeError):
    """A requested managed-instance lifecycle operation could not complete."""


@dataclass(frozen=True)
class DecisionManagedStatus:
    """Secret-free reconciled state for one registry entry."""

    record: DecisionInstanceRecord
    runtime_state: str
    readiness_state: str
    ownership_verified: bool
    registry_durable: bool = True


@dataclass(frozen=True)
class DecisionOrphanStatus:
    """Visible but deliberately unadopted managed-label container candidate."""

    candidate: DecisionContainerCandidate


@dataclass(frozen=True)
class DecisionDiscoveryStatus:
    """Non-fatal indication that managed-label discovery was unavailable."""

    runtime: str | None


@dataclass(frozen=True)
class DecisionStopReceipt:
    """Result of a safe stop or stale-record cleanup."""

    instance_name: str
    action: str
    unknown_container_preserved: bool = False
    registry_durable: bool = True


def list_decision_instances(
    *,
    registry: DecisionInstanceRegistry | None = None,
    driver: DecisionContainerDriver | None = None,
) -> tuple[DecisionManagedStatus | DecisionOrphanStatus | DecisionDiscoveryStatus, ...]:
    """Reconcile and list every recoverable managed instance."""

    instance_registry = registry or DecisionInstanceRegistry()
    container_driver = driver or LowLevelDecisionContainerDriver()
    records = instance_registry.records()
    observed: list[
        DecisionManagedStatus | DecisionOrphanStatus | DecisionDiscoveryStatus
    ] = []
    for record in records:
        try:
            status = _reconcile_record(
                record,
                registry=instance_registry,
                driver=container_driver,
            )
        except DecisionRegistryStaleRecordError:
            # Another lifecycle command changed ownership while this snapshot
            # was being reconciled. Its committed result is authoritative.
            continue
        observed.append(status)
    discover = getattr(container_driver, "list_managed", None)
    if callable(discover):
        # Docker and Podman are independent namespaces. Always inspect both so
        # the selected/default runtime cannot hide an unregistered candidate in
        # the other supported runtime. Discovery remains read-only and failures
        # are surfaced independently below.
        for runtime in SUPPORTED_CONTAINER_RUNTIMES:
            try:
                candidates = discover(runtime)
            except DecisionContainerError:
                observed.append(DecisionDiscoveryStatus(runtime=runtime))
                continue
            for candidate in candidates:
                if not _candidate_is_registered(candidate, records):
                    observed.append(DecisionOrphanStatus(candidate=candidate))
    return tuple(sorted(observed, key=_status_sort_key))


def status_decision_instance(
    instance_name: str,
    *,
    registry: DecisionInstanceRegistry | None = None,
    driver: DecisionContainerDriver | None = None,
) -> DecisionManagedStatus:
    """Return one ownership-aware status, reconciling stale registry state."""

    instance_registry = registry or DecisionInstanceRegistry()
    container_driver = driver or LowLevelDecisionContainerDriver()
    record = instance_registry.get(instance_name)
    status = _reconcile_record(
        record,
        registry=instance_registry,
        driver=container_driver,
        probe_readiness=True,
    )
    return status


def stop_decision_instance(
    instance_name: str,
    *,
    registry: DecisionInstanceRegistry | None = None,
    driver: DecisionContainerDriver | None = None,
) -> DecisionStopReceipt:
    """Stop one owned container, or remove a proven container-missing record."""

    instance_registry = registry or DecisionInstanceRegistry()
    container_driver = driver or LowLevelDecisionContainerDriver()
    record = instance_registry.get(instance_name)
    identity = _record_identity(record)
    try:
        observation = container_driver.inspect(identity)
    except DecisionContainerOwnershipError as error:
        if record.state != "cleanup-required":
            instance_registry.transition(
                record.instance_name,
                record.instance_id,
                expected_generation=record.generation,
                expected_state=record.state,
                state="cleanup-required",
            )
        raise DecisionManagementError(
            f"Refusing to stop Decision runtime instance {instance_name!r}: "
            "container ownership does not match the registry. The unknown "
            "container and registry evidence were preserved; inspect them before "
            "using 'drun forget'."
        ) from error
    except DecisionContainerError:
        raise

    if observation is None:
        if record.state == "starting":
            raise DecisionManagementError(
                f"Refusing to clear starting Decision runtime instance "
                f"{instance_name!r}: its launch process may still own the "
                "reservation. Retry after launch completes, or use 'drun forget' "
                "only after confirming no launch is active."
            )
        removal = instance_registry.remove(
            record.instance_name,
            record.instance_id,
            expected_generation=record.generation,
            expected_state=record.state,
        )
        return DecisionStopReceipt(
            instance_name=record.instance_name,
            action="stale-record-removed",
            registry_durable=removal.durable,
        )

    try:
        container_driver.stop_and_remove(observation.identity)
    except DecisionContainerError:
        instance_registry.transition(
            record.instance_name,
            record.instance_id,
            expected_generation=record.generation,
            expected_state=record.state,
            state="cleanup-required",
            container_id=observation.identity.container_id,
        )
        raise
    removal = instance_registry.remove(
        record.instance_name,
        record.instance_id,
        expected_generation=record.generation,
        expected_state=record.state,
    )
    return DecisionStopReceipt(
        instance_name=record.instance_name,
        action="stopped",
        registry_durable=removal.durable,
    )


def forget_decision_instance(
    instance_name: str,
    *,
    force: bool = False,
    registry: DecisionInstanceRegistry | None = None,
) -> DecisionStopReceipt:
    """Remove registry evidence without inspecting or changing any container."""

    instance_registry = registry or DecisionInstanceRegistry()
    record = instance_registry.get(instance_name)
    if record.state == "starting" and not force:
        raise DecisionManagementError(
            f"Refusing to forget starting Decision runtime instance "
            f"{instance_name!r}: its launch process may still create a container. "
            "Retry after launch completes, or pass --force only after confirming "
            "the launch process is no longer active."
        )
    removal = instance_registry.remove(
        record.instance_name,
        record.instance_id,
        expected_generation=record.generation,
        expected_state=record.state,
    )
    return DecisionStopReceipt(
        instance_name=record.instance_name,
        action="registry-record-forgotten",
        unknown_container_preserved=True,
        registry_durable=removal.durable,
    )


def _reconcile_record(
    record: DecisionInstanceRecord,
    *,
    registry: DecisionInstanceRegistry,
    driver: DecisionContainerDriver,
    probe_readiness: bool = False,
) -> DecisionManagedStatus:
    try:
        observation = driver.inspect(_record_identity(record))
    except DecisionContainerOwnershipError as error:
        updated = record
        if record.state != "cleanup-required":
            transition = registry.transition(
                record.instance_name,
                record.instance_id,
                expected_generation=record.generation,
                expected_state=record.state,
                state="cleanup-required",
            )
            if transition.record is None:  # pragma: no cover - transition invariant.
                raise DecisionRegistryError(
                    "Decision runtime registry is invalid."
                ) from error
            updated = transition.record
            durable = transition.durable
        else:
            durable = True
        return DecisionManagedStatus(
            record=updated,
            runtime_state="ownership-mismatch",
            readiness_state=_unavailable_readiness(updated, probe_readiness),
            ownership_verified=False,
            registry_durable=durable,
        )
    except DecisionContainerError:
        return DecisionManagedStatus(
            record=record,
            runtime_state="runtime-unavailable",
            readiness_state=_unavailable_readiness(record, probe_readiness),
            ownership_verified=False,
        )

    if observation is None:
        # Absence is not proof that no launch process still owns this
        # reservation: image preparation and container creation are not leased.
        # Keep the evidence and let an explicit stop/forget release it.
        return DecisionManagedStatus(
            record=record,
            runtime_state="container-missing",
            readiness_state=(
                "not-ready"
                if probe_readiness and record.state == "running"
                else _unavailable_readiness(record, probe_readiness)
            ),
            ownership_verified=False,
        )

    target_state = record.state
    if record.state == "cleanup-required" or observation.state in {
        "removing",
        "exited",
        "dead",
    }:
        target_state = "cleanup-required"
    # Only the launch path may promote `starting` to `running`, and only after
    # its health probe succeeds. Label ownership plus container state alone do
    # not prove service readiness.
    updated = record
    # A live launch owns the `starting` generation until it commits the exact
    # container identity returned by start(). Observers must not advance that
    # CAS merely because a matching name/label set has become visible.
    observed_identity_changed = (
        record.container_id != observation.identity.container_id
        and record.state != "starting"
    )
    if record.state != target_state or observed_identity_changed:
        updated = registry.transition(
            record.instance_name,
            record.instance_id,
            expected_generation=record.generation,
            expected_state=record.state,
            state=target_state,
            container_id=observation.identity.container_id,
        )
        if updated.record is None:  # pragma: no cover - transition invariant.
            raise DecisionRegistryError("Decision runtime registry is invalid.")
        return DecisionManagedStatus(
            record=updated.record,
            runtime_state=observation.state,
            readiness_state=_readiness_state(
                updated.record,
                observation,
                driver=driver,
                probe_readiness=probe_readiness,
            ),
            ownership_verified=True,
            registry_durable=updated.durable,
        )
    return DecisionManagedStatus(
        record=updated,
        runtime_state=observation.state,
        readiness_state=_readiness_state(
            updated,
            observation,
            driver=driver,
            probe_readiness=probe_readiness,
        ),
        ownership_verified=True,
    )


def _readiness_state(
    record: DecisionInstanceRecord,
    observation: DecisionContainerObservation,
    *,
    driver: DecisionContainerDriver,
    probe_readiness: bool,
) -> str:
    if record.state != "running":
        return "not-applicable"
    if not probe_readiness:
        return "not-probed"
    if observation.state != "running":
        return "not-ready"
    probe = getattr(driver, "probe_ready", None)
    if not callable(probe):
        return "unavailable"
    parsed = urlsplit(record.endpoint)
    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:  # guarded by registry validation.
        raise DecisionRegistryError("Decision runtime registry is invalid.")
    try:
        ready = probe(observation.identity, host=host, port=port)
    except DecisionContainerError:
        return "unavailable"
    return "ready" if ready else "not-ready"


def _unavailable_readiness(
    record: DecisionInstanceRecord,
    probe_readiness: bool,
) -> str:
    if record.state != "running":
        return "not-applicable"
    return "unavailable" if probe_readiness else "not-probed"


def _record_identity(record: DecisionInstanceRecord) -> DecisionContainerIdentity:
    return DecisionContainerIdentity(
        runtime=record.runtime,
        container_name=record.container_name,
        instance_name=record.instance_name,
        identity_digest=record.identity_digest,
        image=record.image,
        container_id=record.container_id,
    )


def _candidate_is_registered(
    candidate: DecisionContainerCandidate,
    records: tuple[DecisionInstanceRecord, ...],
) -> bool:
    if not candidate.label_contract_valid:
        return False
    return any(
        record.runtime == candidate.runtime
        and record.container_name == candidate.container_name
        and record.instance_name == candidate.instance_name
        and record.identity_digest == candidate.identity_digest
        and record.image == candidate.image
        and (
            record.container_id is None or record.container_id == candidate.container_id
        )
        for record in records
    )


def _status_sort_key(
    status: DecisionManagedStatus | DecisionOrphanStatus | DecisionDiscoveryStatus,
) -> str:
    if isinstance(status, DecisionManagedStatus):
        return f"0:{status.record.instance_name}"
    if isinstance(status, DecisionOrphanStatus):
        candidate = status.candidate
        return f"1:{candidate.instance_name or candidate.container_name}"
    return f"2:{status.runtime or ''}"
