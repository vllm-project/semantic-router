"""Release-scoped Kubernetes Lease for CLI deploy/teardown serialization."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Any

from cli.k8s_secret_quarantine import (
    format_rfc3339_timestamp,
    parse_rfc3339_timestamp,
)
from cli.utils import get_logger

log = get_logger(__name__)

RELEASE_OPERATION_LEASE_SECONDS = 30 * 60
RELEASE_OPERATION_RENEW_SECONDS = 60
_KUBECTL_LOCK_REQUEST_TIMEOUT = "10s"
_RENEWAL_THREAD_JOIN_SECONDS = 15
_LEASE_MAX_CLOCK_SKEW = timedelta(minutes=5)
_LOCK_MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
_LOCK_MANAGED_BY_VALUE = "vllm-sr-cli"
_LOCK_RELEASE_LABEL = "app.kubernetes.io/instance"


class K8sReleaseLockMixin:
    """Serialize CLI mutations for one Helm release through a Kubernetes Lease."""

    @contextmanager
    def _release_operation_lock(self) -> Iterator[None]:
        holder = f"vllm-sr-cli-{secrets.token_hex(12)}"
        self._acquire_release_operation_lock(holder)
        self._active_release_lock_holder = holder
        stop_renewal, renewal_failed, renewal_thread = self._start_release_lock_renewal(
            holder
        )
        self._active_release_lock_renewal_failed = renewal_failed
        try:
            yield
        except BaseException:
            self._finish_failed_release_operation_lock(
                holder, stop_renewal, renewal_thread
            )
            raise
        else:
            self._finish_successful_release_operation_lock(
                holder,
                stop_renewal,
                renewal_failed,
                renewal_thread,
            )
        finally:
            self._clear_active_release_operation_lock(holder, renewal_failed)

    def _start_release_lock_renewal(
        self, holder: str
    ) -> tuple[threading.Event, threading.Event, threading.Thread]:
        """Start the bounded background Lease renewer for one holder."""
        stop_renewal = threading.Event()
        renewal_failed = threading.Event()

        def renew() -> None:
            while not stop_renewal.wait(RELEASE_OPERATION_RENEW_SECONDS):
                try:
                    self._renew_release_operation_lock(holder)
                except (OSError, RuntimeError):
                    renewal_failed.set()
                    log.warning(
                        "Unable to renew the Kubernetes operation Lease; the "
                        "current command will fail closed at its next mutation boundary"
                    )
                    return

        renewal_thread = threading.Thread(
            target=renew,
            name="vllm-sr-k8s-release-lock",
            daemon=True,
        )
        try:
            renewal_thread.start()
        except BaseException:
            # A thread-start failure must not leave a locally active holder or
            # an avoidable 30-minute recovery delay.
            with suppress(OSError, RuntimeError):
                self._release_release_operation_lock(holder)
            if getattr(self, "_active_release_lock_holder", None) == holder:
                self._active_release_lock_holder = None
            raise
        return stop_renewal, renewal_failed, renewal_thread

    def _finish_failed_release_operation_lock(
        self,
        holder: str,
        stop_renewal: threading.Event,
        renewal_thread: threading.Thread,
    ) -> None:
        renewal_stopped = self._stop_release_lock_renewal(stop_renewal, renewal_thread)
        if not renewal_stopped:
            log.warning(
                "The Kubernetes operation Lease renewer did not stop; the "
                "Lease was retained until its bounded expiry"
            )
            return
        try:
            self._release_release_operation_lock(holder)
        except (OSError, RuntimeError):
            log.warning(
                "Unable to release the Kubernetes operation Lease after a "
                "failed command; its bounded expiry will permit recovery"
            )

    def _finish_successful_release_operation_lock(
        self,
        holder: str,
        stop_renewal: threading.Event,
        renewal_failed: threading.Event,
        renewal_thread: threading.Thread,
    ) -> None:
        if not self._stop_release_lock_renewal(stop_renewal, renewal_thread):
            raise RuntimeError(
                "Kubernetes release operation Lease renewer did not stop"
            )
        if renewal_failed.is_set():
            with suppress(OSError, RuntimeError):
                self._release_release_operation_lock(holder)
            raise RuntimeError("Kubernetes release operation Lease renewal failed")
        self._release_release_operation_lock(holder)

    def _clear_active_release_operation_lock(
        self,
        holder: str,
        renewal_failed: threading.Event,
    ) -> None:
        if getattr(self, "_active_release_lock_holder", None) == holder:
            self._active_release_lock_holder = None
        if getattr(self, "_active_release_lock_renewal_failed", None) is renewal_failed:
            self._active_release_lock_renewal_failed = None

    @staticmethod
    def _stop_release_lock_renewal(
        stop_renewal: threading.Event,
        renewal_thread: threading.Thread,
    ) -> bool:
        """Stop and join the renewer before clearing the Lease holder.

        The kubectl calls made by the renewer have a ten-second request timeout,
        so the join window is deliberately longer.  If the thread still cannot
        stop, retaining the holder until Lease expiry is safer than racing a late
        renew against a successful release.
        """
        stop_renewal.set()
        renewal_thread.join(timeout=_RENEWAL_THREAD_JOIN_SECONDS)
        return not renewal_thread.is_alive()

    @staticmethod
    def _release_lock_now() -> datetime:
        return datetime.now(timezone.utc)

    def _release_lock_name(self) -> str:
        normalized = re.sub(r"[^a-z0-9-]", "-", self.release_name.lower()).strip("-")
        digest = hashlib.sha256(self.release_name.encode()).hexdigest()[:8]
        return f"{(normalized or 'release')[:35].rstrip('-')}-vsr-op-{digest}"

    def _acquire_release_operation_lock(self, holder: str) -> None:
        now = self._release_lock_now()
        payload = self._get_release_operation_lease()
        if payload is None:
            result = self._run(
                [
                    *self._kubectl_base_cmd(),
                    "create",
                    "-f",
                    "-",
                    f"--request-timeout={_KUBECTL_LOCK_REQUEST_TIMEOUT}",
                ],
                check=False,
                input_text=self._release_lease_manifest(holder=holder, now=now),
            )
            if result.returncode == 0:
                return
            payload = self._get_release_operation_lease()
            if payload is None:
                raise RuntimeError("Kubernetes release operation lock failed")

        resource_version, active_holder = self._release_lease_state(payload, now)
        if active_holder is not None:
            raise RuntimeError(
                "another Kubernetes operation for this release is already in progress"
            )
        result = self._run(
            [
                *self._kubectl_base_cmd(),
                "replace",
                "-f",
                "-",
                f"--request-timeout={_KUBECTL_LOCK_REQUEST_TIMEOUT}",
            ],
            check=False,
            input_text=self._release_lease_manifest(
                holder=holder,
                now=now,
                resource_version=resource_version,
            ),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "another Kubernetes operation for this release won the lock race"
            )

    def _release_release_operation_lock(self, holder: str) -> None:
        payload = self._get_release_operation_lease()
        if payload is None:
            raise RuntimeError("Kubernetes release operation Lease disappeared")
        resource_version, active_holder = self._release_lease_state(
            payload, self._release_lock_now()
        )
        if active_holder != holder:
            raise RuntimeError("Kubernetes release operation Lease ownership changed")
        result = self._run(
            [
                *self._kubectl_base_cmd(),
                "replace",
                "-f",
                "-",
                f"--request-timeout={_KUBECTL_LOCK_REQUEST_TIMEOUT}",
            ],
            check=False,
            input_text=self._release_lease_manifest(
                holder=None,
                now=self._release_lock_now(),
                resource_version=resource_version,
            ),
        )
        if result.returncode != 0:
            raise RuntimeError("Kubernetes release operation Lease release failed")

    def _renew_release_operation_lock(self, holder: str) -> None:
        now = self._release_lock_now()
        payload = self._get_release_operation_lease()
        if payload is None:
            raise RuntimeError("Kubernetes release operation Lease disappeared")
        resource_version, active_holder = self._release_lease_state(payload, now)
        if active_holder != holder:
            raise RuntimeError("Kubernetes release operation Lease ownership changed")
        result = self._run(
            [
                *self._kubectl_base_cmd(),
                "replace",
                "-f",
                "-",
                f"--request-timeout={_KUBECTL_LOCK_REQUEST_TIMEOUT}",
            ],
            check=False,
            input_text=self._release_lease_manifest(
                holder=holder,
                now=now,
                resource_version=resource_version,
            ),
        )
        if result.returncode != 0:
            raise RuntimeError("Kubernetes release operation Lease renewal failed")

    def _assert_release_operation_lock(self) -> None:
        renewal_failed = getattr(self, "_active_release_lock_renewal_failed", None)
        if isinstance(renewal_failed, threading.Event) and renewal_failed.is_set():
            raise RuntimeError("Kubernetes release operation Lease renewal failed")
        holder = getattr(self, "_active_release_lock_holder", None)
        if not isinstance(holder, str) or not holder:
            raise RuntimeError("Kubernetes release operation Lease is not held")
        payload = self._get_release_operation_lease()
        if payload is None:
            raise RuntimeError("Kubernetes release operation Lease disappeared")
        _, active_holder = self._release_lease_state(payload, self._release_lock_now())
        if active_holder != holder:
            raise RuntimeError("Kubernetes release operation Lease ownership changed")

    def _get_release_operation_lease(self) -> dict[str, Any] | None:
        return self._get_json(
            [
                *self._kubectl_base_cmd(),
                "get",
                "lease",
                self._release_lock_name(),
                "--namespace",
                self.namespace,
                "-o",
                "json",
                f"--request-timeout={_KUBECTL_LOCK_REQUEST_TIMEOUT}",
            ],
            allow_not_found=True,
        )

    def _release_lease_state(
        self, payload: dict[str, Any], now: datetime
    ) -> tuple[str, str | None]:
        metadata = payload.get("metadata")
        spec = payload.get("spec")
        if not isinstance(metadata, dict) or not isinstance(spec, dict):
            raise RuntimeError("Kubernetes release operation Lease is malformed")
        if (
            metadata.get("name") != self._release_lock_name()
            or metadata.get("namespace") != self.namespace
        ):
            raise RuntimeError("Kubernetes release operation Lease identity changed")
        resource_version = metadata.get("resourceVersion")
        if not isinstance(resource_version, str) or not resource_version:
            raise RuntimeError("Kubernetes release operation Lease has no version")

        holder = spec.get("holderIdentity")
        if holder in (None, ""):
            return resource_version, None
        duration = spec.get("leaseDurationSeconds")
        renew_time = spec.get("renewTime")
        if (
            not isinstance(holder, str)
            or isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration < 1
            or duration > RELEASE_OPERATION_LEASE_SECONDS
        ):
            raise RuntimeError("Kubernetes release operation Lease is malformed")
        try:
            renewed_at = parse_rfc3339_timestamp(renew_time)
        except ValueError as exc:
            raise RuntimeError(
                "Kubernetes release operation Lease is malformed"
            ) from exc
        if renewed_at > now + _LEASE_MAX_CLOCK_SKEW:
            raise RuntimeError("Kubernetes release operation Lease is malformed")
        if now < renewed_at + timedelta(seconds=duration):
            return resource_version, holder
        return resource_version, None

    def _release_lease_manifest(
        self,
        *,
        holder: str | None,
        now: datetime,
        resource_version: str | None = None,
    ) -> str:
        metadata: dict[str, Any] = {
            "name": self._release_lock_name(),
            "namespace": self.namespace,
            "labels": {
                _LOCK_MANAGED_BY_LABEL: _LOCK_MANAGED_BY_VALUE,
                _LOCK_RELEASE_LABEL: self.release_name,
            },
        }
        if resource_version is not None:
            metadata["resourceVersion"] = resource_version
        spec: dict[str, Any] = {
            "leaseDurationSeconds": RELEASE_OPERATION_LEASE_SECONDS,
            "renewTime": format_rfc3339_timestamp(now),
        }
        if holder is not None:
            spec["holderIdentity"] = holder
            spec["acquireTime"] = format_rfc3339_timestamp(now)
        return json.dumps(
            {
                "apiVersion": "coordination.k8s.io/v1",
                "kind": "Lease",
                "metadata": metadata,
                "spec": spec,
            },
            separators=(",", ":"),
        )
