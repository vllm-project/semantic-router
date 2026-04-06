"""Kubernetes deployment backend — wraps Helm and kubectl operations."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from cli.config_translator import (
    load_profile_values,
    translate_config_to_helm_values,
    write_helm_values_file,
)
from cli.docker_images import get_runtime_images
from cli.kind_cluster import KindClusterManager
from cli.logo import print_vllm_logo
from cli.utils import get_logger

log = get_logger(__name__)

HELM_RELEASE_NAME = "semantic-router"
DEFAULT_NAMESPACE = "vllm-semantic-router-system"
CHART_REL_PATH = os.path.join("deploy", "helm", "semantic-router")
CHART_NAME = "semantic-router"
ENV_SECRET_NAME = "vllm-sr-env-secrets"
CHART_CONFIG_PATH = "/app/config.yaml"
HELM_WAIT_TIMEOUT = "20m"

K8S_SERVICE_TO_LABEL: dict[str, str] = {
    "router": "app.kubernetes.io/component=router",
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
        cluster_name: str | None = None,
    ) -> None:
        self.kind_cluster = KindClusterManager(cluster_name=cluster_name)
        self.namespace = namespace or DEFAULT_NAMESPACE
        self.managed_kind = context in (None, self.kind_cluster.context_name)
        self.context = context or self.kind_cluster.context_name
        self.release_name = release_name or HELM_RELEASE_NAME
        self.resource_name = self.release_name
        self.primary_listener_port = 8888
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
        self._require_tool("helm")
        self._require_tool("kubectl")
        profile_values = load_profile_values(self.profile, self.chart_dir)
        dashboard_enabled = self._dashboard_enabled(profile_values)
        runtime_images = self._resolve_chart_images(
            image=image,
            router_image=kwargs.get("router_image"),
            envoy_image=kwargs.get("envoy_image"),
            dashboard_image=kwargs.get("dashboard_image"),
            pull_policy=pull_policy,
            platform=kwargs.get("platform"),
            dashboard_enabled=dashboard_enabled,
        )
        if self.managed_kind:
            self._require_tool("kind")
            self.kind_cluster.ensure()
            for image_name in dict.fromkeys(
                image_name for image_name in runtime_images.values() if image_name
            ):
                self.kind_cluster.load_image(image_name)

        print_vllm_logo()
        log.info("Deploying vLLM Semantic Router to Kubernetes")
        log.info(f"  Release:   {self.release_name}")
        log.info(f"  Namespace: {self.namespace}")
        log.info(f"  Chart:     {self.chart_dir}")
        if self.context:
            log.info(f"  Context:   {self.context}")
        if self.managed_kind:
            log.info(f"  Kind:      {self.kind_cluster.cluster_name}")

        chart_env_vars = self._normalize_chart_env_vars(env_vars)
        secret_name = self._sync_env_secret(chart_env_vars)

        self.resource_name = self._release_fullname(profile_values)
        router_service_host = f"{self.resource_name}.{self.namespace}.svc.cluster.local"
        values = translate_config_to_helm_values(
            config_file,
            image=runtime_images["router"],
            envoy_image=runtime_images["envoy"],
            dashboard_image=runtime_images["dashboard"],
            pull_policy=pull_policy,
            enable_observability=enable_observability,
            profile_values=profile_values,
            env_vars=chart_env_vars,
            env_secret_name=secret_name,
            router_service_host=router_service_host,
        )
        self.primary_listener_port = self._primary_envoy_listener_port(values)
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
            "--wait",
            "--timeout",
            HELM_WAIT_TIMEOUT,
        ]

        log.info("Running helm upgrade --install ...")
        self._run(cmd)
        log.info("Helm release deployed successfully")

        self._wait_for_pods()
        self._log_k8s_summary()

    def teardown(self) -> None:
        self._require_tool("helm")

        log.info(f"Uninstalling Helm release: {self.release_name}")
        cmd = [
            *self._helm_base_cmd(),
            "uninstall",
            self.release_name,
            "--namespace",
            self.namespace,
        ]
        self._run(cmd, check=False)
        self._delete_secret_if_exists(ENV_SECRET_NAME)
        log.info("Helm release uninstalled")
        if self.managed_kind:
            self.kind_cluster.delete()

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

    def _resolve_chart_images(
        self,
        *,
        image: str | None,
        router_image: str | None,
        envoy_image: str | None,
        dashboard_image: str | None,
        pull_policy: str | None,
        platform: str | None,
        dashboard_enabled: bool,
    ) -> dict[str, str]:
        if not self.managed_kind:
            return {
                "router": image or "",
                "envoy": envoy_image or "",
                "dashboard": dashboard_image or "" if dashboard_enabled else "",
            }

        runtime_images = get_runtime_images(
            image=image,
            router_image=router_image,
            envoy_image=envoy_image,
            dashboard_image=dashboard_image,
            pull_policy=pull_policy,
            platform=platform,
            include_dashboard=dashboard_enabled,
        )
        return {
            "router": runtime_images["router"],
            "envoy": runtime_images["envoy"],
            "dashboard": runtime_images["dashboard"],
        }

    # -- helpers --------------------------------------------------------------

    def _sync_env_secret(self, env_vars: dict[str, str] | None) -> str | None:
        """Create or update a K8s Secret with sensitive env vars; return the secret name or None."""
        if not env_vars:
            return None

        from cli.commands.runtime_support import PASSTHROUGH_ENV_RULES  # noqa: PLC0415

        sensitive_names = {name for name, masked in PASSTHROUGH_ENV_RULES if masked}
        secret_data = {k: v for k, v in env_vars.items() if k in sensitive_names and v}

        if not secret_data:
            return None

        self._ensure_namespace()
        self._delete_secret_if_exists(ENV_SECRET_NAME)

        cmd = [
            *self._kubectl_base_cmd(),
            "create",
            "secret",
            "generic",
            ENV_SECRET_NAME,
            "--namespace",
            self.namespace,
        ]
        for key, value in sorted(secret_data.items()):
            cmd.append(f"--from-literal={key}={value}")

        log.info(
            f"Creating K8s secret '{ENV_SECRET_NAME}' with {len(secret_data)} key(s)"
        )
        self._run(cmd)
        return ENV_SECRET_NAME

    @staticmethod
    def _normalize_chart_env_vars(
        env_vars: dict[str, str] | None,
    ) -> dict[str, str] | None:
        """Rewrite docker-local config env paths onto the Helm chart config mount."""
        if not env_vars:
            return env_vars

        from cli.commands.runtime_support import (  # noqa: PLC0415
            RUNTIME_CONFIG_PATH_ENV,
            SOURCE_CONFIG_PATH_ENV,
        )

        normalized = dict(env_vars)
        normalized[SOURCE_CONFIG_PATH_ENV] = CHART_CONFIG_PATH
        if RUNTIME_CONFIG_PATH_ENV in normalized:
            normalized[RUNTIME_CONFIG_PATH_ENV] = CHART_CONFIG_PATH
        return normalized

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

    def _delete_secret_if_exists(self, name: str) -> None:
        cmd = [
            *self._kubectl_base_cmd(),
            "delete",
            "secret",
            name,
            "--namespace",
            self.namespace,
            "--ignore-not-found",
        ]
        self._run(cmd, check=False)

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
        if self.managed_kind:
            log.info(
                f"Managed Kind cluster: {self.kind_cluster.cluster_name} ({self.context})"
            )
            log.info("")
        log.info("Port forwarding (access locally):")
        log.info(
            f"  kubectl port-forward -n {self.namespace} "
            f"svc/{self.resource_name}-envoy "
            f"{self.primary_listener_port}:{self.primary_listener_port}"
        )
        log.info("=" * 60)

    def _release_fullname(self, profile_values: dict | None) -> str:
        values = profile_values or {}
        fullname_override = str(values.get("fullnameOverride") or "").strip()
        if fullname_override:
            return fullname_override[:63].rstrip("-")
        name = str(values.get("nameOverride") or CHART_NAME).strip() or CHART_NAME
        fullname = (
            self.release_name
            if name in self.release_name
            else f"{self.release_name}-{name}"
        )
        return fullname[:63].rstrip("-")

    @staticmethod
    def _primary_envoy_listener_port(values: dict) -> int:
        listeners = values.get("envoy", {}).get("listeners") or []
        if listeners:
            return int(listeners[0].get("port") or 8888)
        return 8888

    @staticmethod
    def _dashboard_enabled(profile_values: dict | None) -> bool:
        return bool((profile_values or {}).get("dashboard", {}).get("enabled"))

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
    def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        log.debug(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd, check=check)

    @staticmethod
    def _run_display(cmd: list[str]) -> None:
        subprocess.run(cmd, check=False)
