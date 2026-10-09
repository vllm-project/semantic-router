"""Kubernetes deployment backend — wraps Helm and kubectl operations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any

import yaml

from cli import __version__
from cli.config_translator import (
    load_profile_values,
    temporary_helm_values_file,
    translate_config_to_helm_values,
)
from cli.consts import (
    DEFAULT_API_PORT,
    DEFAULT_LISTENER_PORT,
    image_tag_for_cli_version,
)
from cli.decision_model import configured_decision_model
from cli.gateway_mode import DEFAULT_GATEWAY, GATEWAY_STANDALONE
from cli.k8s_env_secret import (
    ENV_SECRET_MANAGER_LABEL,
    ENV_SECRET_MANAGER_VALUE,
    ENV_SECRET_OWNER_LABEL,
    ENV_SECRET_REVISION_ANNOTATION,
    LEGACY_ENV_SECRET_NAME,
    EnvSecretPlan,
    build_env_secret_plan,
    env_secret_owner,
    is_managed_env_secret_name,
    managed_env_secret_names,
    referenced_secret_names,
)
from cli.logo import print_vllm_logo
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
from cli.terminal import echo, fields, heading, success
from cli.utils import get_logger

log = get_logger(__name__)

HELM_RELEASE_NAME = "semantic-router"
DEFAULT_NAMESPACE = "vllm-semantic-router-system"
CHART_REL_PATH = os.path.join("deploy", "helm", "semantic-router")
PUBLISHED_CHART = "oci://ghcr.io/vllm-project/charts/semantic-router"
# helm-publish.yml publishes main as this version, a release as its own.
MAIN_CHART_VERSION = "0.0.0-latest"


def published_chart_version(cli_version: str = __version__) -> str:
    """The published chart that runs the images this CLI runs."""
    tag = image_tag_for_cli_version(cli_version)
    return MAIN_CHART_VERSION if tag == "latest" else tag.removeprefix("v")


ENV_SECRET_REFS_JSONPATH = (
    "jsonpath="
    "{range .items[*].spec.template.spec.containers[*].envFrom[*]}"
    '{.secretRef.name}{"\\n"}{end}'
    "{range .items[*].spec.template.spec.containers[*].env[*]}"
    '{.valueFrom.secretKeyRef.name}{"\\n"}{end}'
    "{range .items[*].spec.template.spec.initContainers[*].envFrom[*]}"
    '{.secretRef.name}{"\\n"}{end}'
    "{range .items[*].spec.template.spec.initContainers[*].env[*]}"
    '{.valueFrom.secretKeyRef.name}{"\\n"}{end}'
)
K8S_SERVICE_TO_LABEL: dict[str, str] = {
    "router": "app.kubernetes.io/component=router",
    "dashboard": "app.kubernetes.io/component=dashboard",
    "envoy": "app.kubernetes.io/component=envoy",
}


def _external_dashboard_management_entry(
    entry: dict[str, object], owner: str
) -> dict[str, object] | None:
    """Validate one operator-owned Dashboard management Secret reference."""

    if "value" in entry:
        raise ValueError(
            "Helm dashboard management credential must use a Secret "
            "secretKeyRef, not a plaintext value"
        )
    value_from = entry.get("valueFrom")
    secret_ref = (
        value_from.get("secretKeyRef") if isinstance(value_from, dict) else None
    )
    referenced_name = secret_ref.get("name") if isinstance(secret_ref, dict) else None
    if referenced_name == LEGACY_ENV_SECRET_NAME or (
        isinstance(referenced_name, str)
        and is_managed_env_secret_name(referenced_name, owner)
    ):
        return None
    referenced_key = secret_ref.get("key") if isinstance(secret_ref, dict) else None
    if not (
        isinstance(referenced_name, str)
        and referenced_name
        and isinstance(referenced_key, str)
        and referenced_key
    ):
        raise ValueError(
            "Helm dashboard management credential must use a Secret secretKeyRef"
        )
    return entry


def _dashboard_management_extra_env(
    configured_entries: list[dict[str, object]],
    *,
    managed_secret_name: str | None,
    owner: str,
) -> list[dict[str, object]]:
    """Resolve Dashboard env without retaining stale managed Secret revisions."""

    extra_env: list[dict[str, object]] = []
    for configured_entry in configured_entries:
        entry = dict(configured_entry)
        if entry.get("name") != MANAGEMENT_CREDENTIAL_ENV:
            extra_env.append(entry)
        elif managed_secret_name is None:
            external_entry = _external_dashboard_management_entry(entry, owner)
            if external_entry is not None:
                extra_env.append(external_entry)
    if managed_secret_name is not None:
        extra_env.append(
            {
                "name": MANAGEMENT_CREDENTIAL_ENV,
                "valueFrom": {
                    "secretKeyRef": {
                        "name": managed_secret_name,
                        "key": MANAGEMENT_CREDENTIAL_ENV,
                    }
                },
            }
        )
    return extra_env


def client_service_port(values: dict[str, Any]) -> int:
    """The Service port the chart's NOTES send API clients to.

    Standalone serves the API on the first listener without TLS, else the
    first listener; extproc leaves it to the gateway in front and exposes the
    Router API port.
    """

    mode = (values.get("gateway") or {}).get("mode") or DEFAULT_GATEWAY
    if mode != GATEWAY_STANDALONE:
        api = (values.get("service") or {}).get("api") or {}
        return int(api.get("port", DEFAULT_API_PORT))
    listeners = (values.get("configOverride") or {}).get("listeners") or []
    primary = next(
        (listener for listener in listeners if not listener.get("tls")),
        listeners[0] if listeners else {},
    )
    return int(primary.get("port", DEFAULT_LISTENER_PORT))


class K8sBackend:
    """Kubernetes deployment backend implemented through Helm."""

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
        # Only deploy installs the chart; status, logs, stop and dashboard
        # read the release, so none of them needs one.
        self.chart_dir = chart_dir

    # -- Deployment operations ------------------------------------------------

    def deploy(
        self,
        config_file: str,
        env_vars: dict[str, str] | None = None,
        *,
        config_document: dict[str, object] | None = None,
        source_config_file: str | None = None,
        image: str | None = None,
        pull_policy: str | None = None,
        enable_observability: bool = True,
        minimal: bool = False,
        readonly: bool = False,
        gateway: str | None = None,
        platform: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._require_tool("helm")
        self._require_tool("kubectl")

        print_vllm_logo()
        log.info("Deploying vLLM Semantic Router to Kubernetes")
        log.info(f"  Release:   {self.release_name}")
        log.info(f"  Namespace: {self.namespace}")
        log.info(f"  Chart:     {self._resolve_chart_dir()}")
        if self.context:
            log.info(f"  Context:   {self.context}")

        # env_vars was discovered against source_config_file (runtime.py forwards it
        # before algorithm/platform overrides rewrite the config), so sensitivity must
        # be re-checked against that same file, not the rewritten effective one, or a
        # name dropped by the rewrite silently loses its secret classification.
        sensitivity_config_file = source_config_file or config_file
        secret_plan = self._plan_env_secret(env_vars, sensitivity_config_file)
        secret_name = secret_plan.name if secret_plan else None

        profile_values = load_profile_values(self.profile, self.chart_dir)
        values = translate_config_to_helm_values(
            config_file,
            config_document=config_document,
            source_config_file=sensitivity_config_file,
            image=image,
            pull_policy=pull_policy,
            enable_observability=enable_observability,
            profile_values=profile_values,
            env_vars=env_vars,
            env_secret_name=secret_name,
            namespace=self.namespace,
            minimal=minimal,
            readonly=readonly,
            gateway=gateway,
            platform=platform,
        )

        self._bind_env_secret_revision(values, secret_name)
        self._bind_dashboard_management_credential(values, secret_plan)

        with temporary_helm_values_file(values) as values_path:
            self._deploy_helm_values(
                values_path,
                secret_plan=secret_plan,
                secret_name=secret_name,
                client_port=client_service_port(values),
            )

    def _deploy_helm_values(
        self,
        values_path: str,
        *,
        secret_plan: EnvSecretPlan | None,
        secret_name: str | None,
        client_port: int,
    ) -> None:
        """Commit one prepared values file and its credential revision."""

        cmd = [
            *self._helm_base_cmd(),
            "upgrade",
            "--install",
            self.release_name,
            self._resolve_chart_dir(),
            "--namespace",
            self.namespace,
            "--create-namespace",
            "-f",
            values_path,
            "--atomic",
            "--cleanup-on-fail",
            "--wait",
            "--timeout",
            "10m",
        ]

        log.info("Running helm upgrade --install ...")
        self._ensure_namespace()
        previous_secret_refs = self._current_release_env_secret_refs()
        previous_managed_secrets = self._list_managed_env_secrets()
        staged_secret_name: str | None = None
        try:
            if secret_plan is not None:
                staged_secret_name = secret_plan.name
                self._create_env_secret(secret_plan)
            self._run(cmd)
        except BaseException:
            if staged_secret_name is not None and not self._discard_staged_env_secret(
                staged_secret_name
            ):
                log.warning(
                    "Failed to remove staged Kubernetes credential Secret after "
                    "deployment failure"
                )
            raise
        self._verify_committed_env_secret(secret_name)
        self._cleanup_obsolete_env_secrets(
            active_secret_name=secret_name,
            previous_secret_refs=previous_secret_refs,
            previous_managed_secrets=previous_managed_secrets,
        )
        log.info("Helm release deployed successfully")

        self._wait_for_pods()
        self._log_k8s_summary(client_port)

    def teardown(self) -> None:
        self._require_tool("helm")
        self._require_tool("kubectl")

        log.info(f"Uninstalling Helm release: {self.release_name}")
        previous_secret_refs = self._current_release_env_secret_refs()
        previous_managed_secrets = self._list_managed_env_secrets()
        cmd = [
            *self._helm_base_cmd(),
            "uninstall",
            self.release_name,
            "--namespace",
            self.namespace,
            "--wait",
            "--timeout",
            "10m",
        ]
        result = self._run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Helm release uninstall failed with exit code {result.returncode}"
            )
        self._cleanup_obsolete_env_secrets(
            active_secret_name=None,
            previous_secret_refs=previous_secret_refs,
            previous_managed_secrets=previous_managed_secrets,
        )
        success("Helm release uninstalled")

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
            result = subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            log.info("\nLog streaming stopped")
            return
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    def status(self, service: str = "all") -> None:
        self._require_tool("kubectl")

        heading(f"Kubernetes deployment status ({self.namespace})")
        echo()

        pod_result = self._run_display(
            [
                *self._kubectl_base_cmd(),
                "get",
                "pods",
                "--namespace",
                self.namespace,
                "-l",
                self._label_for_service(service),
                "-o",
                "wide",
            ]
        )
        echo()

        helm_cmd = [
            *self._helm_base_cmd(),
            "status",
            self.release_name,
            "--namespace",
            self.namespace,
            "--show-desc",
        ]
        helm_result = self._run_display(helm_cmd)
        decision_model = self._live_decision_model()
        if decision_model:
            echo()
            fields((("Decision model", decision_model),))
        failed = next(
            (
                result.returncode
                for result in (pod_result, helm_result)
                if result.returncode
            ),
            0,
        )
        if failed:
            raise SystemExit(failed)

    def _live_decision_model(self) -> str | None:
        """The decision model of the release's live Router config, or None."""

        cmd = [
            *self._kubectl_base_cmd(),
            "get",
            "configmap",
            "--namespace",
            self.namespace,
            "-l",
            f"app.kubernetes.io/instance={self.release_name}",
            "-o",
            "json",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError:
            return None
        if result.returncode != 0:
            return None
        try:
            for item in json.loads(result.stdout).get("items", []):
                data = item.get("data") or {}
                if "config.yaml" in data and "tools_db.json" in data:
                    document = yaml.safe_load(data["config.yaml"]) or {}
                    return configured_decision_model(document)
        except (ValueError, yaml.YAMLError):
            return None
        return None

    def _dashboard_service_query(self, jsonpath: str) -> str | None:
        cmd = [
            *self._kubectl_base_cmd(),
            "get",
            "svc",
            "--namespace",
            self.namespace,
            "-l",
            f"app.kubernetes.io/instance={self.release_name},"
            "app.kubernetes.io/component=dashboard",
            "-o",
            jsonpath,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return result.stdout.strip()

    def get_dashboard_url(self) -> str | None:
        """Return the in-cluster Dashboard address, not reachable from outside."""
        address = self._dashboard_service_query(
            "jsonpath={.items[0].spec.clusterIP}:{.items[0].spec.ports[0].port}"
        )
        return f"http://{address}" if address else None

    def get_dashboard_port_forward(self) -> str | None:
        """Return the command that makes the Dashboard reachable locally."""
        service = self._dashboard_service_query(
            "jsonpath={.items[0].metadata.name}:{.items[0].spec.ports[0].port}"
        )
        if service is None:
            return None
        name, _, port = service.partition(":")
        if not name or not port:
            return None
        return (
            f"{' '.join(self._kubectl_base_cmd())} port-forward "
            f"--namespace {self.namespace} svc/{name} {port}:{port}"
        )

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

    def _plan_env_secret(
        self, env_vars: dict[str, str] | None, config_file: str | None = None
    ) -> EnvSecretPlan | None:
        """Build the next credential Secret revision without cluster mutation."""

        from cli.commands.runtime_support import sensitive_env_names  # noqa: PLC0415

        return build_env_secret_plan(
            namespace=self.namespace,
            release_name=self.release_name,
            env_vars=env_vars,
            sensitive_names=sensitive_env_names(config_file),
        )

    def _create_env_secret(self, plan: EnvSecretPlan) -> None:
        """Create one immutable credential Secret revision from stdin."""

        cmd = [
            *self._kubectl_base_cmd(),
            "create",
            "-f",
            "-",
        ]
        log.info(
            f"Staging Kubernetes credential Secret revision with {plan.key_count} key(s)"
        )
        self._run(cmd, input_text=plan.manifest)

    def _ensure_namespace(self) -> None:
        get_cmd = [
            *self._kubectl_base_cmd(),
            "get",
            "namespace",
            self.namespace,
            "--ignore-not-found",
            "-o",
            "name",
        ]
        current = self._run(get_cmd, check=False, capture_output=True)
        if current.returncode != 0:
            raise RuntimeError(
                "Failed to inspect the Kubernetes namespace before deployment"
            )
        if current.stdout.strip():
            return

        create_cmd = [
            *self._kubectl_base_cmd(),
            "create",
            "namespace",
            self.namespace,
        ]
        created = self._run(create_cmd, check=False, capture_output=True)
        if created.returncode == 0:
            return

        # Another actor may have created it after the initial read. Re-read instead
        # of applying a minimal Namespace object over user-managed metadata.
        raced = self._run(get_cmd, check=False, capture_output=True)
        if raced.returncode == 0 and raced.stdout.strip():
            return
        raise RuntimeError("Failed to create the Kubernetes namespace")

    def _bind_env_secret_revision(self, values: dict, secret_name: str | None) -> None:
        """Bind the pod template to the selected non-secret revision name."""

        configured_secret_names = values.get("envFromSecrets", [])
        if not isinstance(configured_secret_names, list):
            raise ValueError("Helm envFromSecrets must be a list")
        if not all(isinstance(name, str) and name for name in configured_secret_names):
            raise ValueError("Helm envFromSecrets entries must be non-empty strings")
        owner = env_secret_owner(self.namespace, self.release_name)
        secret_names = [
            name
            for name in configured_secret_names
            if name != LEGACY_ENV_SECRET_NAME
            and not (isinstance(name, str) and is_managed_env_secret_name(name, owner))
        ]
        if secret_name is not None and secret_name not in secret_names:
            secret_names.append(secret_name)
        if secret_names:
            values["envFromSecrets"] = secret_names
        else:
            values.pop("envFromSecrets", None)

        annotations = values.get("podAnnotations", {})
        if not isinstance(annotations, dict):
            raise ValueError("Helm podAnnotations must be a mapping")
        annotations = dict(annotations)
        if secret_name is None:
            annotations.pop(ENV_SECRET_REVISION_ANNOTATION, None)
        else:
            annotations[ENV_SECRET_REVISION_ANNOTATION] = secret_name
        if annotations:
            values["podAnnotations"] = annotations
        else:
            values.pop("podAnnotations", None)

    def _bind_dashboard_management_credential(
        self, values: dict, secret_plan: EnvSecretPlan | None
    ) -> None:
        """Expose only the Dashboard management key, never the whole Secret."""

        configured_dashboard = values.get("dashboard", {})
        if not isinstance(configured_dashboard, dict):
            raise ValueError("Helm dashboard values must be a mapping")
        dashboard = dict(configured_dashboard)
        configured_extra_env = dashboard.get("extraEnv", [])
        if not isinstance(configured_extra_env, list) or not all(
            isinstance(entry, dict) and isinstance(entry.get("name"), str)
            for entry in configured_extra_env
        ):
            raise ValueError("Helm dashboard.extraEnv must contain named mappings")

        owner = env_secret_owner(self.namespace, self.release_name)
        managed_secret_name = (
            secret_plan.name
            if secret_plan is not None
            and MANAGEMENT_CREDENTIAL_ENV in secret_plan.keys
            and dashboard.get("enabled") is not False
            else None
        )
        extra_env = _dashboard_management_extra_env(
            configured_extra_env,
            managed_secret_name=managed_secret_name,
            owner=owner,
        )
        if extra_env:
            dashboard["extraEnv"] = extra_env
        else:
            dashboard.pop("extraEnv", None)
        if dashboard:
            values["dashboard"] = dashboard
        else:
            values.pop("dashboard", None)

    def _current_release_env_secret_refs(self) -> set[str]:
        """Return only CLI-managed Secret refs used by this release."""

        cmd = [
            *self._kubectl_base_cmd(),
            "get",
            "deployment",
            "--namespace",
            self.namespace,
            "-l",
            f"app.kubernetes.io/instance={self.release_name},"
            "app.kubernetes.io/component=router",
            "-o",
            ENV_SECRET_REFS_JSONPATH,
        ]
        result = self._run(cmd, capture_output=True)
        owner = env_secret_owner(self.namespace, self.release_name)
        return {
            name
            for name in referenced_secret_names(result.stdout)
            if name == LEGACY_ENV_SECRET_NAME or is_managed_env_secret_name(name, owner)
        }

    def _list_managed_env_secrets(self) -> set[str]:
        """Return validated credential revisions owned by this release."""

        owner = env_secret_owner(self.namespace, self.release_name)
        cmd = [
            *self._kubectl_base_cmd(),
            "get",
            "secret",
            "--namespace",
            self.namespace,
            "-l",
            f"{ENV_SECRET_OWNER_LABEL}={owner},"
            f"{ENV_SECRET_MANAGER_LABEL}={ENV_SECRET_MANAGER_VALUE}",
            "-o",
            'jsonpath={range .items[*]}{.metadata.name}{"\\n"}{end}',
        ]
        result = self._run(cmd, capture_output=True)
        return managed_env_secret_names(result.stdout, owner)

    def _namespace_references_secret(self, secret_name: str) -> bool:
        """Check every deployment before removing the legacy shared Secret."""

        cmd = [
            *self._kubectl_base_cmd(),
            "get",
            "deployment",
            "--namespace",
            self.namespace,
            "-o",
            ENV_SECRET_REFS_JSONPATH,
        ]
        result = self._run(cmd, capture_output=True)
        return secret_name in referenced_secret_names(result.stdout)

    def _verify_committed_env_secret(self, secret_name: str | None) -> None:
        """Verify Helm committed exactly the CLI-managed reference we planned."""

        expected = {secret_name} if secret_name is not None else set()
        if self._current_release_env_secret_refs() != expected:
            raise RuntimeError(
                "Helm deployment completed, but the Kubernetes credential Secret "
                "reference could not be verified; existing Secrets were preserved"
            )

    def _cleanup_obsolete_env_secrets(
        self,
        *,
        active_secret_name: str | None,
        previous_secret_refs: set[str],
        previous_managed_secrets: set[str],
    ) -> None:
        """Delete stale owned revisions only after Helm commits successfully."""

        owner = env_secret_owner(self.namespace, self.release_name)
        obsolete = set(previous_managed_secrets)
        obsolete.update(
            name
            for name in previous_secret_refs
            if is_managed_env_secret_name(name, owner)
        )
        if active_secret_name is not None:
            obsolete.discard(active_secret_name)
        obsolete = {
            name for name in obsolete if not self._namespace_references_secret(name)
        }
        if (
            LEGACY_ENV_SECRET_NAME in previous_secret_refs
            and not self._namespace_references_secret(LEGACY_ENV_SECRET_NAME)
        ):
            obsolete.add(LEGACY_ENV_SECRET_NAME)
        for name in sorted(obsolete):
            self._delete_secret_if_exists(name, required=True)

    def _delete_secret_if_exists(self, name: str, *, required: bool = False) -> bool:
        cmd = [
            *self._kubectl_base_cmd(),
            "delete",
            "secret",
            name,
            "--namespace",
            self.namespace,
            "--ignore-not-found",
        ]
        result = self._run(cmd, check=False)
        if result.returncode == 0:
            return True
        if required:
            raise RuntimeError(
                "Failed to delete an obsolete Kubernetes credential Secret: "
                f"kubectl exited with code {result.returncode}"
            )
        return False

    def _discard_staged_env_secret(self, name: str) -> bool:
        """Best-effort rollback that never replaces the deployment exception."""

        try:
            if self._namespace_references_secret(name):
                return False
            return self._delete_secret_if_exists(name)
        except Exception:
            return False

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
            raise ValueError(
                f"Kubernetes logs/status do not support service '{service}'"
            )
        return f"app.kubernetes.io/instance={self.release_name},{label}"

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
        result = self._run(cmd, check=False)
        if result.returncode != 0:
            # A failed wait means pods never became ready: reporting success
            # here would hide broken rollouts behind a green summary.
            self._log_failed_pod_diagnostics()
            raise SystemExit(result.returncode)

    def _log_failed_pod_diagnostics(self) -> None:
        log.error("kubectl wait failed; current pod status:")
        self._run_display(
            [
                *self._kubectl_base_cmd(),
                "get",
                "pods",
                "-l",
                f"app.kubernetes.io/instance={self.release_name}",
                "--namespace",
                self.namespace,
                "-o",
                "wide",
            ]
        )

    def _log_k8s_summary(self, client_port: int) -> None:
        success("Kubernetes deployment is ready")
        heading("Commands")
        echo("  vllm-sr status --target kubernetes")
        echo("  vllm-sr logs router --target kubernetes [-f]")
        echo("  vllm-sr stop --target kubernetes")
        echo()
        heading("Local access")
        fields(
            (
                (
                    "Port forward",
                    f"kubectl port-forward -n {self.namespace} "
                    f"svc/{self.release_name} {client_port}:{client_port}",
                ),
            )
        )

    @staticmethod
    def _require_tool(name: str) -> None:
        if shutil.which(name) is None:
            raise SystemExit(
                f"'{name}' is required for Kubernetes deployment but was not "
                "found on PATH."
            )

    def _resolve_chart_dir(self) -> str:
        """--chart-dir, a repository checkout's chart, or the published chart
        that runs this CLI's images."""
        if not self.chart_dir:
            self.chart_dir = self._find_chart_dir() or self._fetch_published_chart()
        return self.chart_dir

    @staticmethod
    def _find_chart_dir() -> str | None:
        candidates = [
            CHART_REL_PATH,
            os.path.join(os.getcwd(), CHART_REL_PATH),
        ]
        for path in candidates:
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "Chart.yaml")):
                return os.path.abspath(path)
        return None

    def _fetch_published_chart(self) -> str:
        version = published_chart_version()
        self._fetched_chart = tempfile.TemporaryDirectory(prefix="vllm-sr-chart-")
        log.info(f"No chart in {CHART_REL_PATH}; using {PUBLISHED_CHART} {version}")
        pulled = self._run(
            [
                *self._helm_base_cmd(),
                "pull",
                PUBLISHED_CHART,
                "--version",
                version,
                "--untar",
                "--untardir",
                self._fetched_chart.name,
            ],
            check=False,
            capture_output=True,
        )
        chart = os.path.join(self._fetched_chart.name, os.path.basename(CHART_REL_PATH))
        if pulled.returncode != 0 or not os.path.isfile(
            os.path.join(chart, "Chart.yaml")
        ):
            detail = (pulled.stderr or pulled.stdout or "").strip()
            raise SystemExit(
                f"Could not fetch {PUBLISHED_CHART} {version}"
                f"{': ' + detail if detail else ''}. "
                "Pass --chart-dir with the chart directory of a repository checkout."
            )
        return chart

    @staticmethod
    def _run(
        cmd: list[str],
        check: bool = True,
        *,
        input_text: str | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        log.debug(f"Running: {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            check=check,
            input=input_text,
            text=input_text is not None or capture_output,
            capture_output=capture_output,
        )

    @staticmethod
    def _run_display(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, check=False)
