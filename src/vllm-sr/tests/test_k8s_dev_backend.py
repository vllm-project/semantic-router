"""Focused tests for the managed Kind-backed k8s dev backend path."""

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.commands import runtime as rt  # noqa: E402
from cli.config_translator import translate_config_to_helm_values  # noqa: E402
from cli.k8s_backend import (  # noqa: E402
    HELM_WAIT_TIMEOUT,
    K8S_SERVICE_TO_LABEL,
    K8sBackend,
)
from cli.kind_cluster import KindClusterManager  # noqa: E402
from cli.main import main  # noqa: E402


def _minimal_valid_config(*, listener_port: int = 8888) -> dict:
    return {
        "version": "v0.3",
        "listeners": [
            {
                "name": "public",
                "address": "0.0.0.0",
                "port": listener_port,
                "timeout": "60s",
            }
        ],
        "providers": {
            "defaults": {
                "default_model": "test-model",
                "default_reasoning_effort": "medium",
            },
            "models": [
                {
                    "name": "test-model",
                    "provider_model_id": "test-model",
                    "backend_refs": [
                        {
                            "name": "primary",
                            "endpoint": "llm-katan.default.svc.cluster.local:8000",
                            "protocol": "http",
                            "weight": 100,
                        }
                    ],
                }
            ],
        },
        "routing": {
            "modelCards": [{"name": "test-model"}],
            "decisions": [
                {
                    "name": "default-route",
                    "description": "Default route for k8s backend tests",
                    "priority": 100,
                    "rules": {"operator": "AND", "conditions": []},
                    "modelRefs": [{"model": "test-model"}],
                }
            ],
        },
        "global": {
            "stores": {
                "semantic_cache": {
                    "enabled": False,
                }
            },
            "model_catalog": {
                "embeddings": {
                    "semantic": {
                        "mmbert_model_path": "",
                        "qwen3_model_path": "",
                        "gemma_model_path": "",
                        "bert_model_path": "",
                        "multimodal_model_path": "",
                    }
                },
                "modules": {
                    "prompt_guard": {
                        "enabled": False,
                        "model_ref": "",
                        "model_id": "",
                        "jailbreak_mapping_path": "",
                        "use_mmbert_32k": False,
                    },
                    "classifier": {
                        "domain": {
                            "model_ref": "",
                            "model_id": "",
                            "category_mapping_path": "",
                            "use_mmbert_32k": False,
                        },
                        "pii": {
                            "model_ref": "",
                            "model_id": "",
                            "pii_mapping_path": "",
                            "use_mmbert_32k": False,
                        },
                    },
                    "feedback_detector": {
                        "enabled": False,
                        "model_ref": "",
                        "model_id": "",
                        "use_mmbert_32k": False,
                    },
                },
            },
        },
    }


class TestConfigTranslator:
    def test_canonical_config_tree_is_passed_through(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump(_minimal_valid_config()))

        values = translate_config_to_helm_values(str(config))

        assert values["config"]["version"] == "v0.3"
        assert (
            values["config"]["providers"]["defaults"]["default_model"] == "test-model"
        )
        assert values["config"]["routing"]["decisions"][0]["name"] == "default-route"
        assert yaml.safe_load(values["configRaw"]) == _minimal_valid_config()

    def test_runtime_image_overrides_share_the_same_pull_policy(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump(_minimal_valid_config()))

        values = translate_config_to_helm_values(
            str(config),
            image="repo/router:v2",
            envoy_image="repo/envoy:v4",
            dashboard_image="repo/dashboard:v3",
            pull_policy="never",
            router_service_host="semantic-router.vllm-semantic-router-system.svc.cluster.local",
        )

        assert values["image"] == {
            "repository": "repo/router",
            "tag": "v2",
            "pullPolicy": "Never",
        }
        assert values["envoy"]["image"] == {
            "repository": "repo/envoy",
            "tag": "v4",
            "pullPolicy": "Never",
        }
        assert values["dashboard"]["image"] == {
            "repository": "repo/dashboard",
            "tag": "v3",
            "pullPolicy": "Never",
        }
        assert values["envoy"]["enabled"] is True
        assert values["envoy"]["listeners"] == [
            {
                "name": "public",
                "address": "0.0.0.0",
                "port": 8888,
                "timeout": "60s",
            }
        ]
        assert (
            "semantic-router.vllm-semantic-router-system.svc.cluster.local"
            in values["envoy"]["config"]
        )
        assert "/dev/stdout" in values["envoy"]["config"]

    def test_k8s_backend_normalizes_runtime_paths_to_chart_mount(self):
        env_vars = {
            "VLLM_SR_SOURCE_CONFIG_PATH": "/app/config.yaml",
            "VLLM_SR_RUNTIME_CONFIG_PATH": "/app/.vllm-sr/runtime-config.yaml",
            "SR_LOG_LEVEL": "debug",
        }

        assert K8sBackend._normalize_chart_env_vars(env_vars) == {
            "VLLM_SR_SOURCE_CONFIG_PATH": "/app/config.yaml",
            "VLLM_SR_RUNTIME_CONFIG_PATH": "/app/config.yaml",
            "SR_LOG_LEVEL": "debug",
        }

    def test_router_logs_use_router_component_label(self):
        assert K8S_SERVICE_TO_LABEL["router"] == "app.kubernetes.io/component=router"


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm is required")
def test_rendered_chart_separates_runtime_component_selectors(tmp_path):
    chart_dir = PROJECT_ROOT.parents[1] / "deploy" / "helm" / "semantic-router"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_minimal_valid_config()))
    values = translate_config_to_helm_values(
        str(config),
        router_service_host="semantic-router.vllm-semantic-router-system.svc.cluster.local",
    )
    values.setdefault("dashboard", {})["enabled"] = True
    values_path = tmp_path / "values.yaml"
    values_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "test-release",
            str(chart_dir),
            "-f",
            str(values_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resources = {
        (doc["kind"], doc["metadata"]["name"]): doc
        for doc in yaml.safe_load_all(rendered.stdout)
        if (
            isinstance(doc, dict)
            and doc.get("kind")
            and doc.get("metadata", {}).get("name")
        )
    }

    router_deployment = resources[("Deployment", "test-release-semantic-router")]
    router_service = resources[("Service", "test-release-semantic-router")]
    router_metrics_service = resources[
        ("Service", "test-release-semantic-router-metrics")
    ]
    router_config = resources[("ConfigMap", "test-release-semantic-router-config")]
    envoy_deployment = resources[("Deployment", "test-release-semantic-router-envoy")]
    envoy_service = resources[("Service", "test-release-semantic-router-envoy")]
    dashboard_deployment = resources[
        ("Deployment", "test-release-semantic-router-dashboard")
    ]
    dashboard_service = resources[("Service", "test-release-semantic-router-dashboard")]

    assert router_deployment["spec"]["selector"]["matchLabels"] == {
        "app.kubernetes.io/name": "semantic-router",
        "app.kubernetes.io/instance": "test-release",
        "app": "semantic-router",
        "app.kubernetes.io/component": "router",
    }
    assert (
        router_service["spec"]["selector"]
        == router_deployment["spec"]["selector"]["matchLabels"]
    )
    assert (
        router_metrics_service["spec"]["selector"]
        == router_deployment["spec"]["selector"]["matchLabels"]
    )

    assert envoy_deployment["spec"]["selector"]["matchLabels"] == {
        "app.kubernetes.io/name": "semantic-router",
        "app.kubernetes.io/instance": "test-release",
        "app": "semantic-router",
        "app.kubernetes.io/component": "envoy",
    }
    assert (
        envoy_service["spec"]["selector"]
        == envoy_deployment["spec"]["selector"]["matchLabels"]
    )
    assert [port["port"] for port in envoy_service["spec"]["ports"]] == [8888, 9901]

    assert dashboard_deployment["spec"]["selector"]["matchLabels"] == {
        "app.kubernetes.io/name": "semantic-router",
        "app.kubernetes.io/instance": "test-release",
        "app": "semantic-router",
        "app.kubernetes.io/component": "dashboard",
    }
    assert (
        dashboard_service["spec"]["selector"]
        == dashboard_deployment["spec"]["selector"]["matchLabels"]
    )
    env = {
        entry["name"]: entry["value"]
        for entry in dashboard_deployment["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
        if "value" in entry
    }
    assert env["TARGET_ROUTER_API_URL"] == "http://test-release-semantic-router:8080"
    assert env["TARGET_ENVOY_URL"] == "http://test-release-semantic-router-envoy:8888"
    assert (
        env["TARGET_ENVOY_ADMIN_URL"]
        == "http://test-release-semantic-router-envoy:9901"
    )
    assert (
        yaml.safe_load(router_config["data"]["config.yaml"]) == _minimal_valid_config()
    )


class TestKindClusterManager:
    def test_context_name_defaults_from_cluster_name(self):
        manager = KindClusterManager(cluster_name="dev-stack")

        assert manager.context_name == "kind-dev-stack"

    def test_ensure_noops_when_cluster_exists(self, monkeypatch):
        manager = KindClusterManager(cluster_name="dev-stack")
        monkeypatch.setattr(manager, "exists", lambda: True)
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: pytest.fail("kind create should not run"),
        )

        manager.ensure()

    def test_load_image_invokes_kind_load(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        manager = KindClusterManager(cluster_name="dev-stack")
        manager.load_image("repo/router:dev")

        assert calls == [
            (
                [
                    "kind",
                    "load",
                    "docker-image",
                    "repo/router:dev",
                    "--name",
                    "dev-stack",
                ],
                {"check": True},
            )
        ]


class TestManagedKindBackend:
    def test_managed_kind_deploy_ensures_cluster_and_loads_images(
        self, monkeypatch, tmp_path
    ):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"version": "v0.3"}))
        translate_calls = {}
        helm_runs = []
        kind_instances = []

        class FakeKind:
            def __init__(self, cluster_name=None):
                self.cluster_name = cluster_name or "vllm-sr"
                self.context_name = f"kind-{self.cluster_name}"
                self.ensure_calls = 0
                self.loaded_images = []
                kind_instances.append(self)

            def ensure(self):
                self.ensure_calls += 1

            def load_image(self, image_name):
                self.loaded_images.append(image_name)

            def delete(self):
                pytest.fail("delete should not be called during deploy")

        def fake_translate(config_file, **kwargs):
            translate_calls["config_file"] = config_file
            translate_calls["kwargs"] = kwargs
            return {}

        monkeypatch.setattr("cli.k8s_backend.KindClusterManager", FakeKind)
        monkeypatch.setattr(
            "cli.k8s_backend.get_runtime_images",
            lambda **kwargs: {
                "router": "repo/router:dev",
                "envoy": "envoyproxy/envoy:v1",
                "dashboard": "repo/dashboard:dev",
            },
        )
        monkeypatch.setattr(
            "cli.k8s_backend.load_profile_values",
            lambda *args: {"dashboard": {"enabled": True}},
        )
        monkeypatch.setattr(
            "cli.k8s_backend.translate_config_to_helm_values", fake_translate
        )
        monkeypatch.setattr(
            "cli.k8s_backend.write_helm_values_file", lambda values: "/tmp/values.yaml"
        )
        monkeypatch.setattr(K8sBackend, "_sync_env_secret", lambda self, env_vars: None)
        monkeypatch.setattr(K8sBackend, "_wait_for_pods", lambda self: None)
        monkeypatch.setattr(K8sBackend, "_log_k8s_summary", lambda self: None)
        monkeypatch.setattr(
            K8sBackend, "_require_tool", staticmethod(lambda name: None)
        )
        monkeypatch.setattr(
            K8sBackend,
            "_run",
            staticmethod(
                lambda cmd, check=True: helm_runs.append((cmd, check))
                or MagicMock(returncode=0)
            ),
        )

        backend = K8sBackend(profile="dev", chart_dir="/tmp/chart")
        backend.deploy(str(config), pull_policy="never", enable_observability=True)

        assert len(kind_instances) == 1
        fake_kind = kind_instances[0]
        assert fake_kind.ensure_calls == 1
        assert fake_kind.loaded_images == [
            "repo/router:dev",
            "envoyproxy/envoy:v1",
            "repo/dashboard:dev",
        ]
        assert translate_calls["config_file"] == str(config)
        assert translate_calls["kwargs"]["image"] == "repo/router:dev"
        assert translate_calls["kwargs"]["envoy_image"] == "envoyproxy/envoy:v1"
        assert translate_calls["kwargs"]["dashboard_image"] == "repo/dashboard:dev"
        assert helm_runs[0][0][:4] == [
            "helm",
            "--kube-context",
            "kind-vllm-sr",
            "upgrade",
        ]
        assert helm_runs[0][0][-2:] == ["--timeout", HELM_WAIT_TIMEOUT]

    def test_managed_kind_skips_dashboard_image_when_profile_does_not_enable_it(
        self, monkeypatch, tmp_path
    ):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"version": "v0.3"}))
        kind_instances = []
        runtime_image_kwargs = {}

        class FakeKind:
            def __init__(self, cluster_name=None):
                self.cluster_name = cluster_name or "vllm-sr"
                self.context_name = f"kind-{self.cluster_name}"
                self.loaded_images = []
                kind_instances.append(self)

            def ensure(self):
                return None

            def load_image(self, image_name):
                self.loaded_images.append(image_name)

            def delete(self):
                pytest.fail("delete should not be called during deploy")

        monkeypatch.setattr("cli.k8s_backend.KindClusterManager", FakeKind)
        monkeypatch.setattr(
            "cli.k8s_backend.get_runtime_images",
            lambda **kwargs: runtime_image_kwargs.update(kwargs)
            or {
                "router": "repo/router:dev",
                "envoy": "envoyproxy/envoy:v1",
                "dashboard": "",
            },
        )
        monkeypatch.setattr("cli.k8s_backend.load_profile_values", lambda *args: {})
        monkeypatch.setattr(
            "cli.k8s_backend.translate_config_to_helm_values",
            lambda config_file, **kwargs: {},
        )
        monkeypatch.setattr(
            "cli.k8s_backend.write_helm_values_file", lambda values: "/tmp/values.yaml"
        )
        monkeypatch.setattr(K8sBackend, "_sync_env_secret", lambda self, env_vars: None)
        monkeypatch.setattr(K8sBackend, "_wait_for_pods", lambda self: None)
        monkeypatch.setattr(K8sBackend, "_log_k8s_summary", lambda self: None)
        monkeypatch.setattr(
            K8sBackend, "_require_tool", staticmethod(lambda name: None)
        )
        monkeypatch.setattr(
            K8sBackend,
            "_run",
            staticmethod(lambda cmd, check=True: MagicMock(returncode=0)),
        )

        backend = K8sBackend(chart_dir="/tmp/chart")
        backend.deploy(str(config), pull_policy="never", enable_observability=True)

        assert runtime_image_kwargs["include_dashboard"] is False
        assert kind_instances[0].loaded_images == [
            "repo/router:dev",
            "envoyproxy/envoy:v1",
        ]

    def test_explicit_context_skips_managed_kind_bootstrap(self, monkeypatch, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"version": "v0.3"}))
        translate_calls = {}
        kind_instances = []

        class FakeKind:
            def __init__(self, cluster_name=None):
                self.cluster_name = cluster_name or "vllm-sr"
                self.context_name = f"kind-{self.cluster_name}"
                self.ensure_calls = 0
                self.loaded_images = []
                kind_instances.append(self)

            def ensure(self):
                self.ensure_calls += 1

            def load_image(self, image_name):
                self.loaded_images.append(image_name)

            def delete(self):
                self.ensure_calls += 100

        monkeypatch.setattr("cli.k8s_backend.KindClusterManager", FakeKind)
        monkeypatch.setattr(
            "cli.k8s_backend.get_runtime_images",
            lambda **kwargs: pytest.fail(
                "existing-cluster deploy should not resolve local images"
            ),
        )
        monkeypatch.setattr("cli.k8s_backend.load_profile_values", lambda *args: {})
        monkeypatch.setattr(
            "cli.k8s_backend.translate_config_to_helm_values",
            lambda config_file, **kwargs: translate_calls.update(kwargs) or {},
        )
        monkeypatch.setattr(
            "cli.k8s_backend.write_helm_values_file", lambda values: "/tmp/values.yaml"
        )
        monkeypatch.setattr(K8sBackend, "_sync_env_secret", lambda self, env_vars: None)
        monkeypatch.setattr(K8sBackend, "_wait_for_pods", lambda self: None)
        monkeypatch.setattr(K8sBackend, "_log_k8s_summary", lambda self: None)
        monkeypatch.setattr(
            K8sBackend, "_require_tool", staticmethod(lambda name: None)
        )
        monkeypatch.setattr(
            K8sBackend,
            "_run",
            staticmethod(lambda cmd, check=True: MagicMock(returncode=0)),
        )

        backend = K8sBackend(context="existing-cluster", chart_dir="/tmp/chart")
        backend.deploy(
            str(config),
            image="repo/router:stable",
            envoy_image="repo/envoy:stable",
            dashboard_image="repo/dashboard:stable",
            pull_policy="ifnotpresent",
        )

        assert len(kind_instances) == 1
        fake_kind = kind_instances[0]
        assert fake_kind.ensure_calls == 0
        assert fake_kind.loaded_images == []
        assert translate_calls["image"] == "repo/router:stable"
        assert translate_calls["envoy_image"] == "repo/envoy:stable"
        assert translate_calls["dashboard_image"] == ""

    def test_teardown_deletes_managed_kind_cluster(self, monkeypatch):
        delete_calls = []

        class FakeKind:
            def __init__(self, cluster_name=None):
                self.cluster_name = cluster_name or "vllm-sr"
                self.context_name = f"kind-{self.cluster_name}"

            def delete(self):
                delete_calls.append(True)

        monkeypatch.setattr("cli.k8s_backend.KindClusterManager", FakeKind)
        monkeypatch.setattr(
            K8sBackend, "_require_tool", staticmethod(lambda name: None)
        )
        monkeypatch.setattr(
            K8sBackend, "_delete_secret_if_exists", lambda self, name: None
        )
        monkeypatch.setattr(
            K8sBackend,
            "_run",
            staticmethod(lambda cmd, check=True: MagicMock(returncode=0)),
        )

        backend = K8sBackend(chart_dir="/tmp/chart")
        backend.teardown()

        assert delete_calls == [True]


class TestRuntimeCommand:
    def test_serve_passes_platform_into_k8s_backend(self, monkeypatch):
        deployed = {}

        class FakeK8s:
            def deploy(self, **kwargs):
                deployed.update(kwargs)

        monkeypatch.setattr(rt, "_build_backend", lambda *args, **kwargs: FakeK8s())
        monkeypatch.setattr(
            rt,
            "ensure_bootstrap_workspace",
            lambda _: MagicMock(config_path=Path("/dev/null"), setup_mode=False),
        )
        monkeypatch.setattr(
            rt,
            "resolve_effective_config_path",
            lambda *args: Path("/dev/null"),
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "serve",
                "--config",
                "/dev/null",
                "--target",
                "k8s",
                "--platform",
                "amd",
                "--image-pull-policy",
                "never",
            ],
        )

        assert result.exit_code == 0
        assert deployed["platform"] == "amd"
