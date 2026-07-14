"""Deploy and teardown orchestration for the Kubernetes CLI backend."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cli.k8s_secret_quarantine import (
    validate_quarantine_deadline,
)
from cli.utils import get_logger

if TYPE_CHECKING:
    from cli.k8s_backend import K8sBackend

log = get_logger(__name__)


@dataclass(frozen=True)
class TeardownSecretCleanup:
    """Credential generations quarantined by one completed uninstall phase."""

    names: tuple[str, ...]
    deadline: datetime


def deploy_release(
    backend: K8sBackend,
    config_file: str,
    env_vars: dict[str, str] | None,
    *,
    image: str | None,
    pull_policy: str | None,
    enable_observability: bool,
    strip_dashboard_auth: Callable[[Mapping[str, str] | None], dict[str, str]],
    print_logo: Callable[[], None],
    load_profile: Callable[..., dict[str, Any] | None],
    translate_values: Callable[..., dict[str, Any]],
    write_values_file: Callable[..., str],
) -> None:
    """Deploy one atomic Helm revision and clean obsolete credentials."""
    backend._require_tool("helm")
    backend._require_tool("kubectl")
    backend._ensure_namespace()
    with backend._release_operation_lock():
        _deploy_release_locked(
            backend,
            config_file,
            env_vars,
            image=image,
            pull_policy=pull_policy,
            enable_observability=enable_observability,
            strip_dashboard_auth=strip_dashboard_auth,
            print_logo=print_logo,
            load_profile=load_profile,
            translate_values=translate_values,
            write_values_file=write_values_file,
        )


def _deploy_release_locked(
    backend: K8sBackend,
    config_file: str,
    env_vars: dict[str, str] | None,
    *,
    image: str | None,
    pull_policy: str | None,
    enable_observability: bool,
    strip_dashboard_auth: Callable[[Mapping[str, str] | None], dict[str, str]],
    print_logo: Callable[[], None],
    load_profile: Callable[..., dict[str, Any] | None],
    translate_values: Callable[..., dict[str, Any]],
    write_values_file: Callable[..., str],
) -> None:
    """Run one deploy while the release-scoped Lease is held."""
    runtime_env = strip_dashboard_auth(env_vars)
    backend._assert_release_operation_lock()

    print_logo()
    log.info("Deploying vLLM Semantic Router to Kubernetes")
    log.info(f"  Release:   {backend.release_name}")
    log.info(f"  Namespace: {backend.namespace}")
    log.info(f"  Chart:     {backend.chart_dir}")
    if backend.context:
        log.info(f"  Context:   {backend.context}")

    secret_plan = backend._plan_env_secret(runtime_env)
    profile_values = load_profile(backend.profile, backend.chart_dir)
    values = translate_values(
        config_file,
        image=image,
        pull_policy=pull_policy,
        enable_observability=enable_observability,
        profile_values=profile_values,
        env_vars=runtime_env,
        env_secret_name=secret_plan.active_name,
    )
    values["runtimeCredentials"] = {
        "generation": secret_plan.active_name or "",
        "recreateForLooperRotation": secret_plan.recreate_for_looper_rotation,
    }
    values_directory = tempfile.mkdtemp(prefix="vllm-sr-helm-")
    try:
        os.chmod(values_directory, 0o700)
        values_path = write_values_file(values, values_directory)
        command = [
            *backend._helm_base_cmd(),
            "upgrade",
            "--install",
            backend.release_name,
            backend.chart_dir,
            "--namespace",
            backend.namespace,
            "--create-namespace",
            "-f",
            values_path,
            "--atomic",
            "--wait",
            "--history-max",
            str(backend._helm_history_max),
            "--timeout",
            "10m",
        ]

        created_name: str | None = None
        try:
            backend._assert_release_operation_lock()
            created_name = backend._create_planned_secret(secret_plan)
            backend._assert_release_operation_lock()
            log.info("Running helm upgrade --install ...")
            backend._run(command)
        except Exception:
            if created_name is not None:
                backend._delete_managed_secret_if_unreferenced(created_name)
            raise

        log.info("Helm release deployed successfully")
        backend._assert_release_operation_lock()
        backend._garbage_collect_managed_secrets()
        backend._wait_for_pods()
        backend._log_k8s_summary()
    finally:
        shutil.rmtree(values_directory)


def teardown_release(backend: K8sBackend) -> None:
    """Uninstall a release before deleting only its unreferenced Secrets."""
    backend._require_tool("helm")
    backend._require_tool("kubectl")
    if not backend._namespace_exists():
        log.info("Kubernetes namespace is absent; release is already stopped")
        return
    with backend._release_operation_lock():
        cleanup = _teardown_release_locked(backend)
    if cleanup is None:
        _log_teardown_cleanup_complete(backend)
        return

    # Do not hold the release Lease through the rollback quarantine. A later
    # deploy must be able to restore service immediately; it can safely adopt a
    # quarantined exact generation because the final cleanup re-acquires the
    # Lease and rechecks current references before every deletion.
    cleanup_now = backend._wait_for_secret_quarantine(cleanup.deadline)
    with backend._release_operation_lock():
        _cleanup_teardown_secrets_locked(backend, cleanup, now=cleanup_now)


def _teardown_release_locked(
    backend: K8sBackend,
) -> TeardownSecretCleanup | None:
    """Uninstall and quarantine credentials while the release Lease is held."""

    backend._assert_release_operation_lock()
    release_exists = backend._helm_release_exists()
    # Validate current Kubernetes state before starting a destructive uninstall.
    backend._current_router_secret_refs()
    backend._log_legacy_secret_retention()
    managed_secrets = backend._list_release_managed_secret_candidates()
    now = backend._managed_secret_gc_now()
    quarantine_deadline: datetime | None = None

    if release_exists:
        # Stamp every old generation before uninstall. An already-running atomic
        # or manual rollback may have loaded an old manifest and adopt its Secret
        # only after the live-reference scan and uninstall complete.
        backend._assert_release_operation_lock()
        quarantine_deadline = backend._quarantine_managed_secrets(
            managed_secrets, now=now
        )
        backend._assert_release_operation_lock()
        log.info(f"Uninstalling Helm release: {backend.release_name}")
        command = [
            *backend._helm_base_cmd(),
            "uninstall",
            backend.release_name,
            "--namespace",
            backend.namespace,
            "--wait",
        ]
        result = backend._run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "Helm uninstall failed; runtime credential Secrets were retained"
            )
    else:
        log.info("Helm release is already absent; retrying runtime credential cleanup")

    if quarantine_deadline is None and any(
        candidate.delete_not_before is None for candidate in managed_secrets.values()
    ):
        # Migrate pre-quarantine state on the first cleanup attempt. Deleting an
        # old generation immediately is unsafe because an earlier Helm operation
        # may already have loaded its manifest.
        backend._assert_release_operation_lock()
        quarantine_deadline = backend._quarantine_managed_secrets(
            managed_secrets, now=now
        )

    if not managed_secrets:
        return None
    if quarantine_deadline is None:
        try:
            quarantine_deadline = validate_quarantine_deadline(
                now,
                max(
                    candidate.protected_until()
                    for candidate in managed_secrets.values()
                ),
            )
        except ValueError as exc:
            raise RuntimeError("managed Secret has unsafe lifecycle metadata") from exc
    return TeardownSecretCleanup(
        names=tuple(sorted(managed_secrets)),
        deadline=quarantine_deadline,
    )


def _cleanup_teardown_secrets_locked(
    backend: K8sBackend,
    cleanup: TeardownSecretCleanup,
    *,
    now: datetime,
) -> None:
    """Delete only still-unreferenced quarantined generations under a new Lease."""
    backend._assert_release_operation_lock()
    if backend._helm_release_exists():
        # A deployment completed while teardown was waiting without the Lease.
        # Even when its live Deployment has moved to a newer Secret, retained
        # Helm revisions may still reference any generation in this cleanup
        # set. Defer the entire batch so the normal deploy GC can evaluate both
        # current and historical references; a live-only check is insufficient.
        log.info(
            "A later deployment exists; quarantined runtime credential Secret "
            "cleanup was safely deferred"
        )
        return
    latest = backend._list_release_managed_secret_candidates()
    cleanup_results = [
        backend._delete_secret_after_teardown(
            name,
            latest[name],
            now=now,
        )
        for name in cleanup.names
        if name in latest
    ]
    if not all(cleanup_results):
        raise RuntimeError("Runtime credential Secret cleanup was incomplete")

    _log_teardown_cleanup_complete(backend)


def _log_teardown_cleanup_complete(backend: K8sBackend) -> None:
    log.info(
        "Release-labelled runtime credential Secret cleanup completed for "
        "the absent Helm release"
    )
