"""Kubernetes deployment backend — wraps Helm and kubectl operations."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
from typing import Any

from cli.config_translator import (
    load_profile_values,
    translate_config_to_helm_values,
    write_helm_values_file,
)
from cli.dashboard_auth_runtime import without_dashboard_auth_env
from cli.k8s_secret_history import (
    deployment_secret_refs,
    read_helm_retained_secret_refs,
)
from cli.k8s_secret_plan import (
    EnvSecretPlan,
    SecretSnapshot,
    desired_encoded_secret_data,
    encode_secret_data,
)
from cli.logo import print_vllm_logo
from cli.utils import get_logger

log = get_logger(__name__)

HELM_RELEASE_NAME = "semantic-router"
DEFAULT_NAMESPACE = "vllm-semantic-router-system"
CHART_REL_PATH = os.path.join("deploy", "helm", "semantic-router")
ENV_SECRET_NAME = "vllm-sr-env-secrets"
HELM_HISTORY_MAX = 10
_MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
_MANAGED_BY_VALUE = "vllm-sr-cli"
_RELEASE_LABEL = "app.kubernetes.io/instance"
_CLI_SECRET_LABEL = "semantic-router.vllm.ai/runtime-env-secret"

K8S_SERVICE_TO_LABEL: dict[str, str] = {
    "router": "app.kubernetes.io/name=semantic-router",
    "dashboard": "app.kubernetes.io/component=dashboard",
    "envoy": "app.kubernetes.io/component=envoy",
}


class K8sBackend:
    """DeploymentBackend implementation for Kubernetes via Helm."""

    def __init__(
        self,
        *,
        namespace: str | None = None,
        context: str | None = None,
        release_name: str | None = None,
        profile: str | None = None,
        chart_dir: str | None = None,
    ) -> None:
        self.namespace = namespace or DEFAULT_NAMESPACE
        self.context = context
        self.release_name = release_name or HELM_RELEASE_NAME
        self.profile = profile
        self.chart_dir = chart_dir or self._find_chart_dir()

    # -- DeploymentBackend interface ------------------------------------------

    def deploy(
        self,
        config_file: str,
        env_vars: dict[str, str] | None = None,
        *,
        image: str | None = None,
        pull_policy: str | None = None,
        enable_observability: bool = True,
        **kwargs: Any,
    ) -> None:
        env_vars = without_dashboard_auth_env(env_vars)
        self._require_tool("helm")
        self._require_tool("kubectl")

        print_vllm_logo()
        log.info("Deploying vLLM Semantic Router to Kubernetes")
        log.info(f"  Release:   {self.release_name}")
        log.info(f"  Namespace: {self.namespace}")
        log.info(f"  Chart:     {self.chart_dir}")
        if self.context:
            log.info(f"  Context:   {self.context}")

        secret_plan = self._plan_env_secret(env_vars)

        profile_values = load_profile_values(self.profile, self.chart_dir)
        values = translate_config_to_helm_values(
            config_file,
            image=image,
            pull_policy=pull_policy,
            enable_observability=enable_observability,
            profile_values=profile_values,
            env_vars=env_vars,
            env_secret_name=secret_plan.active_name,
        )
        values["runtimeCredentials"] = {
            "generation": secret_plan.active_name or "",
            "recreateForLooperRotation": secret_plan.recreate_for_looper_rotation,
        }
        values_path = write_helm_values_file(values)

        cmd = [
            *self._helm_base_cmd(),
            "upgrade",
            "--install",
            self.release_name,
            self.chart_dir,
            "--namespace",
            self.namespace,
            "--create-namespace",
            "-f",
            values_path,
            "--atomic",
            "--wait",
            "--history-max",
            str(HELM_HISTORY_MAX),
            "--timeout",
            "10m",
        ]

        created_name: str | None = None
        try:
            created_name = self._create_planned_secret(secret_plan)
            log.info("Running helm upgrade --install ...")
            self._run(cmd)
        except Exception:
            if created_name is not None:
                self._delete_managed_secret_if_unreferenced(created_name)
            raise

        log.info("Helm release deployed successfully")
        self._garbage_collect_managed_secrets()

        self._wait_for_pods()
        self._log_k8s_summary()

    def teardown(self) -> None:
        self._require_tool("helm")
        self._require_tool("kubectl")

        release_exists = self._helm_release_exists()
        # Validate current Kubernetes state before starting a destructive uninstall.
        self._current_router_secret_refs()
        self._log_legacy_secret_retention()
        managed_secret_names = self._list_release_managed_secret_names()

        if release_exists:
            log.info(f"Uninstalling Helm release: {self.release_name}")
            cmd = [
                *self._helm_base_cmd(),
                "uninstall",
                self.release_name,
                "--namespace",
                self.namespace,
                "--wait",
            ]
            result = self._run(cmd, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    "Helm uninstall failed; runtime credential Secrets were retained"
                )
        else:
            log.info(
                "Helm release is already absent; retrying runtime credential cleanup"
            )

        cleanup_results = [
            self._delete_secret_if_not_current(name) for name in managed_secret_names
        ]
        if not all(cleanup_results):
            raise RuntimeError("Runtime credential Secret cleanup was incomplete")
        if release_exists:
            log.info(
                "Helm release uninstalled and release-labelled runtime credential "
                "Secrets cleaned up"
            )
        else:
            log.info(
                "Release-labelled runtime credential Secret cleanup completed for "
                "the absent Helm release"
            )

    def logs(self, service: str, follow: bool = False) -> None:
        self._require_tool("kubectl")

        label = self._label_for_service(service)
        cmd = [
            *self._kubectl_base_cmd(),
            "logs",
            "-l",
            label,
            "--namespace",
            self.namespace,
            "--all-containers",
            "--tail=200",
        ]
        if follow:
            cmd.append("--follow")

        log.info(f"Streaming {service} logs from Kubernetes ...")
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            log.info("\nLog streaming stopped")

    def status(self, service: str = "all") -> None:
        self._require_tool("kubectl")

        log.info("=" * 60)
        log.info(f"Kubernetes deployment status  (namespace: {self.namespace})")
        log.info("")

        self._run_display(
            [
                *self._kubectl_base_cmd(),
                "get",
                "pods",
                "--namespace",
                self.namespace,
                "-o",
                "wide",
            ]
        )
        log.info("")

        helm_cmd = [
            *self._helm_base_cmd(),
            "status",
            self.release_name,
            "--namespace",
            self.namespace,
            "--show-desc",
        ]
        self._run_display(helm_cmd)
        log.info("=" * 60)

    def get_dashboard_url(self) -> str | None:
        cmd = [
            *self._kubectl_base_cmd(),
            "get",
            "svc",
            "--namespace",
            self.namespace,
            "-l",
            "app.kubernetes.io/component=dashboard",
            "-o",
            "jsonpath={.items[0].spec.clusterIP}:{.items[0].spec.ports[0].port}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return f"http://{result.stdout.strip()}"
        return None

    def is_running(self) -> bool:
        cmd = [
            *self._helm_base_cmd(),
            "status",
            self.release_name,
            "--namespace",
            self.namespace,
            "-o",
            "json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode == 0

    # -- helpers --------------------------------------------------------------

    def _plan_env_secret(self, env_vars: dict[str, str] | None) -> EnvSecretPlan:
        """Plan an immutable credential generation without mutating the cluster."""
        previous = self._current_cli_secret_snapshot()
        encoded_desired = desired_encoded_secret_data(env_vars, previous)

        if (
            previous is not None
            and previous.name != ENV_SECRET_NAME
            and previous.cli_managed
            and previous.immutable
            and previous.data_known
            and previous.data == encoded_desired
        ):
            return EnvSecretPlan(
                active_name=previous.name,
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

    def _new_secret_name(self) -> str:
        return f"{self._release_secret_prefix()}-{secrets.token_hex(6)}"

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
            managed_names = self._list_release_managed_secret_names()
        except (OSError, RuntimeError):
            log.warning("Unable to list managed runtime Secrets; skipping cleanup")
            return False
        candidates = set(managed_names) - protected
        cleanup_ok = True
        for name in sorted(candidates):
            cleanup_ok = (
                self._delete_managed_secret_if_unreferenced(name) and cleanup_ok
            )
        return cleanup_ok

    def _delete_managed_secret_if_unreferenced(self, name: str) -> bool:
        protected = self._protected_secret_refs()
        if protected is None:
            log.warning("Unable to verify rollback Secret references; retaining it")
            return False
        if name in protected:
            log.info("Retaining a runtime Secret referenced by the release history")
            return True
        return self._delete_secret_if_exists(name)

    def _delete_secret_if_not_current(self, name: str) -> bool:
        try:
            current_refs = self._current_router_secret_refs()
        except (OSError, RuntimeError):
            log.warning("Unable to verify current Secret references; retaining it")
            return False
        if name in current_refs:
            log.info("Retaining a runtime Secret still referenced by the release")
            return False
        return self._delete_secret_if_exists(name)

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
            HELM_HISTORY_MAX,
        )

    def _helm_release_exists(self) -> bool:
        """Return an exact Helm release state, failing closed on query ambiguity."""
        release_pattern = "^" + re.escape(self.release_name).replace(r"\-", "-") + "$"
        try:
            result = subprocess.run(
                [
                    *self._helm_base_cmd(),
                    "list",
                    "--all",
                    "--namespace",
                    self.namespace,
                    "--filter",
                    release_pattern,
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise RuntimeError("Helm release state query failed") from exc
        if result.returncode != 0:
            raise RuntimeError("Helm release state query failed")
        if not result.stdout.strip():
            raise RuntimeError("Helm returned an empty release state")
        try:
            releases = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Helm returned malformed release state JSON") from exc
        if not isinstance(releases, list) or len(releases) > 1:
            raise RuntimeError("Helm returned an unexpected release state")
        if not releases:
            return False
        release = releases[0]
        if not isinstance(release, dict):
            raise RuntimeError("Helm returned a malformed release state")
        name = release.get("name")
        status = release.get("status")
        if name != self.release_name or not isinstance(status, str) or not status:
            raise RuntimeError("Helm returned a mismatched release state")
        return True

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

    def _list_release_managed_secret_names(self) -> list[str]:
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
        names = []
        for item in self._json_items(payload):
            if not self._is_release_managed_secret(item):
                continue
            metadata = item.get("metadata", {})
            name = metadata.get("name") if isinstance(metadata, dict) else None
            if isinstance(name, str):
                names.append(name)
        return sorted(set(names))

    def _get_json(
        self,
        cmd: list[str],
        *,
        allow_not_found: bool = False,
    ) -> dict[str, Any] | None:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr if isinstance(result.stderr, str) else ""
            not_found = "notfound" in stderr.replace(" ", "").lower()
            if allow_not_found and not_found:
                return None
            raise RuntimeError("kubectl query failed while inspecting runtime Secrets")
        if not result.stdout.strip():
            raise RuntimeError(
                "kubectl returned an empty response while inspecting Secrets"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("kubectl returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("kubectl returned an unexpected JSON document")
        return payload

    @staticmethod
    def _json_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        if payload is None:
            return []
        items = payload.get("items")
        if not isinstance(items, list):
            raise RuntimeError("kubectl returned a malformed resource list")
        if any(not isinstance(item, dict) for item in items):
            raise RuntimeError("kubectl returned malformed resource list items")
        return [item for item in items if isinstance(item, dict)]

    @staticmethod
    def _deployment_secret_refs(deployment: dict[str, Any]) -> list[str]:
        try:
            return sorted(deployment_secret_refs(deployment))
        except ValueError as exc:
            raise RuntimeError("kubectl returned a malformed Deployment") from exc

    def _ensure_namespace(self) -> None:
        cmd = [
            *self._kubectl_base_cmd(),
            "create",
            "namespace",
            self.namespace,
            "--dry-run=client",
            "-o",
            "yaml",
        ]
        pipe_cmd = [*self._kubectl_base_cmd(), "apply", "-f", "-"]
        log.debug(f"Ensuring namespace {self.namespace}")
        ns_yaml = subprocess.run(cmd, capture_output=True, text=True, check=True)
        subprocess.run(pipe_cmd, input=ns_yaml.stdout, text=True, check=True)

    def _delete_secret_if_exists(self, name: str) -> bool:
        if name == ENV_SECRET_NAME:
            self._log_legacy_secret_retention()
            return True
        cmd = [
            *self._kubectl_base_cmd(),
            "delete",
            "secret",
            name,
            "--namespace",
            self.namespace,
            "--ignore-not-found",
        ]
        try:
            result = self._run(cmd, check=False)
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
            f"Retaining namespace-global legacy runtime Secret {ENV_SECRET_NAME}; "
            "the CLI never deletes this unowned compatibility Secret automatically. "
            "An operator may delete it only after auditing all live workload "
            "references and retained Helm revisions for every release in namespace "
            f"{self.namespace}."
        )

    def _helm_base_cmd(self) -> list[str]:
        cmd = ["helm"]
        if self.context:
            cmd += ["--kube-context", self.context]
        return cmd

    def _kubectl_base_cmd(self) -> list[str]:
        cmd = ["kubectl"]
        if self.context:
            cmd += ["--context", self.context]
        return cmd

    def _label_for_service(self, service: str) -> str:
        if service == "all":
            return f"app.kubernetes.io/instance={self.release_name}"
        label = K8S_SERVICE_TO_LABEL.get(service)
        if label is None:
            log.warning(
                f"Unknown service '{service}', falling back to release selector"
            )
            return f"app.kubernetes.io/instance={self.release_name}"
        return label

    def _wait_for_pods(self) -> None:
        log.info("Waiting for pods to become ready ...")
        cmd = [
            *self._kubectl_base_cmd(),
            "wait",
            "--for=condition=ready",
            "pod",
            "-l",
            f"app.kubernetes.io/instance={self.release_name}",
            "--namespace",
            self.namespace,
            "--timeout=600s",
        ]
        self._run(cmd, check=False)

    def _log_k8s_summary(self) -> None:
        log.info("=" * 60)
        log.info("vLLM Semantic Router deployed on Kubernetes!")
        log.info("")
        log.info("Commands:")
        log.info("  - vllm-sr status --target k8s")
        log.info("  - vllm-sr logs router --target k8s [-f]")
        log.info("  - vllm-sr stop --target k8s")
        log.info("")
        log.info("Port forwarding (access locally):")
        log.info(
            f"  kubectl port-forward -n {self.namespace} "
            f"svc/{self.release_name} 8080:8080"
        )
        log.info("=" * 60)

    @staticmethod
    def _require_tool(name: str) -> None:
        if shutil.which(name) is None:
            raise SystemExit(
                f"'{name}' is required for Kubernetes deployment but was not "
                "found on PATH."
            )

    @staticmethod
    def _find_chart_dir() -> str:
        candidates = [
            CHART_REL_PATH,
            os.path.join(os.getcwd(), CHART_REL_PATH),
        ]
        for path in candidates:
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "Chart.yaml")):
                return os.path.abspath(path)
        raise SystemExit(
            f"Helm chart directory not found. Looked in: {candidates}. "
            "Set --chart-dir or run from the repository root."
        )

    @staticmethod
    def _run(
        cmd: list[str],
        check: bool = True,
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess:
        log.debug(f"Running: {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            check=check,
            input=input_text,
            text=input_text is not None,
        )

    @staticmethod
    def _run_display(cmd: list[str]) -> None:
        subprocess.run(cmd, check=False)
