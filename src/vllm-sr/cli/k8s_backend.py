"""Kubernetes deployment backend — wraps Helm and kubectl operations."""

from __future__ import annotations

import os
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
from cli.k8s_release_lifecycle import (
    deploy_release,
    teardown_release,
)
from cli.k8s_release_lock import K8sReleaseLockMixin
from cli.k8s_resource_inspection import K8sResourceInspectionMixin
from cli.k8s_secret_lifecycle import (
    ENV_SECRET_NAME as _ENV_SECRET_NAME,
)
from cli.k8s_secret_lifecycle import (
    HELM_HISTORY_MAX as _HELM_HISTORY_MAX,
)
from cli.k8s_secret_lifecycle import (
    K8sSecretLifecycleMixin,
)
from cli.logo import print_vllm_logo
from cli.utils import get_logger

log = get_logger(__name__)

ENV_SECRET_NAME = _ENV_SECRET_NAME
HELM_HISTORY_MAX = _HELM_HISTORY_MAX
HELM_RELEASE_NAME = "semantic-router"
DEFAULT_NAMESPACE = "vllm-semantic-router-system"
CHART_REL_PATH = os.path.join("deploy", "helm", "semantic-router")
K8S_SERVICE_TO_LABEL: dict[str, str] = {
    "router": "app.kubernetes.io/name=semantic-router",
    "dashboard": "app.kubernetes.io/component=dashboard",
    "envoy": "app.kubernetes.io/component=envoy",
}


class K8sBackend(
    K8sReleaseLockMixin,
    K8sSecretLifecycleMixin,
    K8sResourceInspectionMixin,
):
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
        deploy_release(
            self,
            config_file,
            env_vars,
            image=image,
            pull_policy=pull_policy,
            enable_observability=enable_observability,
            strip_dashboard_auth=without_dashboard_auth_env,
            print_logo=print_vllm_logo,
            load_profile=load_profile_values,
            translate_values=translate_config_to_helm_values,
            write_values_file=write_helm_values_file,
        )

    def teardown(self) -> None:
        teardown_release(self)

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

    @property
    def _helm_history_max(self) -> int:
        """Preserve the module-level history-limit override seam."""
        return HELM_HISTORY_MAX

    @property
    def _legacy_env_secret_name(self) -> str:
        """Preserve the module-level legacy Secret compatibility seam."""
        return ENV_SECRET_NAME

    def _new_secret_name(self) -> str:
        return f"{self._release_secret_prefix()}-{secrets.token_hex(6)}"

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
