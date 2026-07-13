"""Release-scoped immutable Secret lifecycle for the K8s CLI backend."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from cli.k8s_secret_history import read_helm_retained_secret_refs
from cli.k8s_secret_plan import (
    EnvSecretPlan,
    SecretSnapshot,
    desired_encoded_secret_data,
    encode_secret_data,
)
from cli.k8s_secret_quarantine import (
    MANAGED_SECRET_QUARANTINE_ANNOTATION,
    ManagedSecretCandidate,
    candidate_from_metadata,
    format_rfc3339_timestamp,
    quarantine_deadline,
    validate_quarantine_deadline,
)
from cli.utils import get_logger

log = get_logger(__name__)

ENV_SECRET_NAME = "vllm-sr-env-secrets"
HELM_HISTORY_MAX = 10
_MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
_MANAGED_BY_VALUE = "vllm-sr-cli"
_RELEASE_LABEL = "app.kubernetes.io/instance"
_CLI_SECRET_LABEL = "semantic-router.vllm.ai/runtime-env-secret"


class K8sSecretLifecycleMixin:
    """Own immutable runtime credential generations for one Helm release."""

    def _plan_env_secret(self, env_vars: dict[str, str] | None) -> EnvSecretPlan:
        """Plan an immutable credential generation without mutating the cluster."""
        previous = self._current_cli_secret_snapshot()
        encoded_desired = desired_encoded_secret_data(env_vars, previous)

        # The release operation Lease prevents a concurrent CLI teardown from
        # deleting this generation between snapshot and Helm commit. Reusing an
        # exact immutable generation avoids a no-op Deployment rollout while
        # every changed credential set still gets a new Secret.
        if (
            previous is not None
            and previous.cli_managed
            and previous.immutable
            and previous.data_known
            and previous.data == encoded_desired
        ):
            return EnvSecretPlan(
                active_name=previous.name,
                new_manifest=None,
                recreate_for_looper_rotation=True,
            )

        new_name = self._new_secret_name()
        return EnvSecretPlan(
            active_name=new_name,
            new_manifest=self._secret_manifest(new_name, encoded_desired),
            recreate_for_looper_rotation=True,
        )

    @staticmethod
    def _encode_secret_data(secret_data: dict[str, str]) -> dict[str, str]:
        return encode_secret_data(secret_data)

    def _secret_manifest(self, name: str, encoded_data: dict[str, str]) -> str:
        return json.dumps(
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {
                    "name": name,
                    "namespace": self.namespace,
                    "labels": {
                        _MANAGED_BY_LABEL: _MANAGED_BY_VALUE,
                        _RELEASE_LABEL: self.release_name,
                        _CLI_SECRET_LABEL: "true",
                    },
                },
                "immutable": True,
                "type": "Opaque",
                "data": encoded_data,
            },
            separators=(",", ":"),
        )

    def _create_planned_secret(self, plan: EnvSecretPlan) -> str | None:
        if not plan.creates_secret or plan.active_name is None:
            return None

        self._ensure_namespace()
        log.info("Creating a release-scoped immutable runtime credential Secret")
        self._run(
            [*self._kubectl_base_cmd(), "create", "-f", "-"],
            input_text=plan.new_manifest,
        )
        return plan.active_name

    def _garbage_collect_managed_secrets(self) -> bool:
        protected = self._protected_secret_refs()
        if protected is None:
            log.warning("Unable to verify rollback Secret references; skipping cleanup")
            return False

        try:
            managed_secrets = self._list_release_managed_secret_candidates()
        except (OSError, RuntimeError):
            log.warning("Unable to list managed runtime Secrets; skipping cleanup")
            return False
        candidates = set(managed_secrets) - protected
        if not candidates:
            return True

        now = self._managed_secret_gc_now()
        cleanup_ok = True
        for name in sorted(candidates):
            candidate = managed_secrets[name]
            if candidate.is_protected(now):
                log.info(
                    "Retaining a quarantined runtime Secret while a concurrent "
                    "Helm operation may still reference it"
                )
                continue
            cleanup_ok = (
                self._delete_managed_secret_if_unreferenced(name, candidate)
                and cleanup_ok
            )
        return cleanup_ok

    @staticmethod
    def _managed_secret_gc_now() -> datetime:
        """Return an injectable, timezone-aware reference time for GC."""
        return datetime.now(timezone.utc)

    def _delete_managed_secret_if_unreferenced(
        self,
        name: str,
        candidate: ManagedSecretCandidate | None = None,
    ) -> bool:
        if candidate is None:
            try:
                candidate = self._managed_secret_candidate_for_delete(name)
            except (OSError, RuntimeError):
                log.warning("Unable to verify managed Secret identity; retaining it")
                return False
            if candidate is None:
                return True
        protected = self._protected_secret_refs()
        if protected is None:
            log.warning("Unable to verify rollback Secret references; retaining it")
            return False
        if name in protected:
            log.info("Retaining a runtime Secret referenced by the release history")
            return True
        return self._delete_secret_if_exists(name, candidate)

    def _delete_secret_if_not_current(
        self,
        name: str,
        candidate: ManagedSecretCandidate,
    ) -> bool:
        try:
            current_refs = self._current_router_secret_refs()
        except (OSError, RuntimeError):
            log.warning("Unable to verify current Secret references; retaining it")
            return False
        if name in current_refs:
            log.info("Retaining a runtime Secret still referenced by the release")
            return False
        return self._delete_secret_if_exists(name, candidate)

    def _delete_secret_after_teardown(
        self,
        name: str,
        candidate: ManagedSecretCandidate,
        *,
        now: datetime,
    ) -> bool:
        """Delete expired quarantine state after a final live-reference check."""
        if candidate.is_protected(now):
            log.info(
                "Retaining a quarantined runtime Secret while a concurrent "
                "Helm operation may still reference it"
            )
            return False
        return self._delete_secret_if_not_current(name, candidate)

    def _quarantine_managed_secrets(
        self,
        candidates: Mapping[str, ManagedSecretCandidate],
        *,
        now: datetime,
    ) -> datetime:
        """Persist a teardown-relative no-delete boundary before uninstall."""
        deadline = quarantine_deadline(now)
        if candidates:
            deadline = max(
                deadline,
                *(candidate.protected_until() for candidate in candidates.values()),
            )
        try:
            deadline = validate_quarantine_deadline(now, deadline)
        except ValueError as exc:
            raise RuntimeError("managed Secret has unsafe lifecycle metadata") from exc
        encoded_deadline = format_rfc3339_timestamp(deadline)
        for name in sorted(candidates):
            if name == self._legacy_env_secret_name:
                continue
            self._run(
                [
                    *self._kubectl_base_cmd(),
                    "annotate",
                    "secret",
                    name,
                    f"{MANAGED_SECRET_QUARANTINE_ANNOTATION}={encoded_deadline}",
                    "--namespace",
                    self.namespace,
                    "--overwrite",
                ]
            )
        return deadline

    def _wait_for_secret_quarantine(self, deadline: datetime) -> datetime:
        """Wait interruptibly until every pre-uninstall Helm operation expires."""
        log.info(
            "Waiting for the bounded Helm rollback window before deleting "
            "runtime credential Secrets"
        )
        started_at = self._managed_secret_gc_now()
        max_wait_seconds = max(0.0, (deadline - started_at).total_seconds())
        monotonic_deadline = time.monotonic() + max_wait_seconds
        while True:
            remaining = monotonic_deadline - time.monotonic()
            if remaining <= 0:
                return max(self._managed_secret_gc_now(), deadline)
            time.sleep(min(remaining, 5.0))

    def _protected_secret_refs(self) -> set[str] | None:
        try:
            current_refs = set(self._current_router_secret_refs())
        except (OSError, RuntimeError):
            return None
        retained_refs = self._helm_retained_secret_refs()
        if retained_refs is None:
            return None
        return current_refs | retained_refs

    def _helm_retained_secret_refs(self) -> set[str] | None:
        return read_helm_retained_secret_refs(
            self._helm_base_cmd(),
            self.release_name,
            self.namespace,
            self._helm_history_max,
        )

    def _current_cli_secret_snapshot(self) -> SecretSnapshot | None:
        for name in reversed(self._current_router_secret_refs()):
            payload = self._get_secret_json(name)
            if self._is_release_managed_secret(payload):
                return self._snapshot_from_payload(name, payload)
        return None

    def _release_secret_prefix(self) -> str:
        release = re.sub(r"[^a-z0-9-]", "-", self.release_name.lower()).strip("-")
        return f"{(release or 'release')[:30].rstrip('-')}-vsr-env"

    def _snapshot_from_payload(
        self,
        name: str,
        payload: dict[str, Any] | None,
    ) -> SecretSnapshot:
        if payload is None:
            return SecretSnapshot(name=name, data={})
        raw_data = payload.get("data")
        data = (
            {
                key: value
                for key, value in raw_data.items()
                if isinstance(key, str) and isinstance(value, str)
            }
            if isinstance(raw_data, dict)
            else {}
        )
        return SecretSnapshot(
            name=name,
            data=data,
            immutable=payload.get("immutable") is True,
            cli_managed=True,
            data_known=True,
        )

    def _get_secret_json(self, name: str) -> dict[str, Any] | None:
        return self._get_json(
            [
                *self._kubectl_base_cmd(),
                "get",
                "secret",
                name,
                "--namespace",
                self.namespace,
                "-o",
                "json",
            ],
            allow_not_found=True,
        )

    def _is_release_managed_secret(self, payload: dict[str, Any] | None) -> bool:
        if payload is None:
            return False
        metadata = payload.get("metadata")
        labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
        if not isinstance(labels, dict):
            return False
        return (
            labels.get(_MANAGED_BY_LABEL) == _MANAGED_BY_VALUE
            and labels.get(_RELEASE_LABEL) == self.release_name
            and labels.get(_CLI_SECRET_LABEL) == "true"
        )

    def _current_router_secret_refs(self) -> list[str]:
        selector = (
            f"{_RELEASE_LABEL}={self.release_name},app.kubernetes.io/component=router"
        )
        payload = self._get_json(
            [
                *self._kubectl_base_cmd(),
                "get",
                "deployment",
                "--namespace",
                self.namespace,
                "-l",
                selector,
                "-o",
                "json",
            ],
            allow_not_found=True,
        )
        refs: list[str] = []
        for deployment in self._json_items(payload):
            for name in self._deployment_secret_refs(deployment):
                if name not in refs:
                    refs.append(name)
        return refs

    def _list_release_managed_secret_payloads(self) -> list[dict[str, Any]]:
        selector = (
            f"{_MANAGED_BY_LABEL}={_MANAGED_BY_VALUE},"
            f"{_RELEASE_LABEL}={self.release_name},"
            f"{_CLI_SECRET_LABEL}=true"
        )
        payload = self._get_json(
            [
                *self._kubectl_base_cmd(),
                "get",
                "secret",
                "--namespace",
                self.namespace,
                "-l",
                selector,
                "-o",
                "json",
            ],
            allow_not_found=True,
        )
        return [
            item
            for item in self._json_items(payload)
            if self._is_release_managed_secret(item)
        ]

    def _list_release_managed_secret_candidates(
        self,
    ) -> dict[str, ManagedSecretCandidate]:
        """Return GC candidates with trusted creation times, or fail closed."""
        candidates: dict[str, ManagedSecretCandidate] = {}
        for item in self._list_release_managed_secret_payloads():
            try:
                name, candidate = candidate_from_metadata(item.get("metadata"))
            except ValueError as exc:
                raise RuntimeError("managed Secret has malformed metadata") from exc
            if name in candidates:
                raise RuntimeError("managed Secret has malformed metadata")
            candidates[name] = candidate
        return dict(sorted(candidates.items()))

    def _list_release_managed_secret_names(self) -> list[str]:
        names = []
        for item in self._list_release_managed_secret_payloads():
            metadata = item.get("metadata", {})
            name = metadata.get("name") if isinstance(metadata, dict) else None
            if isinstance(name, str):
                names.append(name)
        return sorted(set(names))

    def _ensure_namespace(self) -> None:
        """Create an absent namespace without updating an existing one.

        Namespace-scoped operators commonly have all permissions needed for
        Secrets, Leases, and Helm workloads but cannot patch the cluster-scoped
        Namespace object.  A read-first, create-only path preserves that least-
        privilege deployment.  A failed create is accepted only when a second
        structured read proves that another actor won the creation race.
        """
        if self._namespace_exists():
            return
        log.debug(f"Creating namespace {self.namespace}")
        result = self._run(
            [*self._kubectl_base_cmd(), "create", "namespace", self.namespace],
            check=False,
        )
        if result.returncode != 0 and not self._namespace_exists():
            raise RuntimeError("Kubernetes namespace creation failed")

    def _managed_secret_candidate_for_delete(
        self,
        name: str,
    ) -> ManagedSecretCandidate | None:
        payload = self._get_secret_json(name)
        if payload is None:
            return None
        if not self._is_release_managed_secret(payload):
            raise RuntimeError("Secret identity is not release-managed")
        try:
            payload_name, candidate = candidate_from_metadata(payload.get("metadata"))
        except ValueError as exc:
            raise RuntimeError("managed Secret has malformed metadata") from exc
        if payload_name != name:
            raise RuntimeError("managed Secret identity changed during lookup")
        return candidate

    def _delete_secret_if_exists(
        self,
        name: str,
        candidate: ManagedSecretCandidate | None = None,
    ) -> bool:
        if name == self._legacy_env_secret_name:
            self._log_legacy_secret_retention()
            return True
        if candidate is None:
            try:
                candidate = self._managed_secret_candidate_for_delete(name)
            except (OSError, RuntimeError):
                log.warning(
                    "Failed to verify a managed runtime Secret identity; "
                    "a later cleanup will retry"
                )
                return False
            if candidate is None:
                return True
        if not candidate.uid or not candidate.resource_version:
            log.warning(
                "Failed to verify a managed runtime Secret identity; "
                "a later cleanup will retry"
            )
            return False

        # A normal `kubectl delete` deliberately omits resource-version checks.
        # Send DeleteOptions to the core/v1 endpoint so Kubernetes atomically
        # refuses deletion if this name now identifies a replacement object or
        # if the inspected generation changed after the final reference scan.
        delete_options = json.dumps(
            {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {
                    "uid": candidate.uid,
                    "resourceVersion": candidate.resource_version,
                },
                "propagationPolicy": "Background",
            },
            separators=(",", ":"),
        )
        resource_path = (
            f"/api/v1/namespaces/{quote(self.namespace, safe='')}/secrets/"
            f"{quote(name, safe='')}"
        )
        command = [
            *self._kubectl_base_cmd(),
            "delete",
            "--raw",
            resource_path,
            "-f",
            "-",
        ]
        try:
            result = self._run(command, check=False, input_text=delete_options)
        except OSError:
            log.warning(
                "Failed to remove a managed runtime Secret; a later cleanup will retry"
            )
            return False
        if result.returncode != 0:
            log.warning(
                "Failed to remove a managed runtime Secret; a later cleanup will retry"
            )
            return False
        return True

    def _log_legacy_secret_retention(self) -> None:
        log.warning(
            "Retaining namespace-global legacy runtime Secret "
            f"{self._legacy_env_secret_name}; "
            "the CLI never deletes this unowned compatibility Secret automatically. "
            "An operator may delete it only after auditing all live workload "
            "references and retained Helm revisions for every release in namespace "
            f"{self.namespace}."
        )
