"""Tests for the deployment backend abstraction and K8s / Docker wiring."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_translator import (  # noqa: E402
    _deep_merge,
    translate_config_to_helm_values,
    write_helm_values_file,
)
from cli.deployment_backend import DEFAULT_TARGET, resolve_target  # noqa: E402
from cli.docker_backend import DockerBackend  # noqa: E402
from cli.k8s_backend import K8sBackend  # noqa: E402

# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


class TestResolveTarget:
    def test_none_returns_default(self):
        assert resolve_target(None) == DEFAULT_TARGET

    def test_docker(self):
        assert resolve_target("docker") == "docker"

    def test_k8s(self):
        assert resolve_target("k8s") == "k8s"

    def test_case_insensitive(self):
        assert resolve_target("K8S") == "k8s"
        assert resolve_target("Docker") == "docker"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid deployment target"):
            resolve_target("aws")


# ---------------------------------------------------------------------------
# DockerBackend
# ---------------------------------------------------------------------------


class TestDockerBackend:
    def test_deploy_delegates_to_start_vllm_sr(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "cli.docker_backend.start_vllm_sr",
            lambda *a, **kw: captured.update(kw),
        )

        backend = DockerBackend()
        backend.deploy(
            config_file="/tmp/config.yaml",
            source_config_file="/tmp/source-config.yaml",
            runtime_config_file="/tmp/runtime-config.yaml",
            env_vars={"A": "B"},
            image="test:img",
            router_image="test:router",
            envoy_image="test:envoy",
            dashboard_image="test:dashboard",
            topology="split",
            pull_policy="always",
            enable_observability=False,
        )

        assert captured["source_config_file"] == "/tmp/source-config.yaml"
        assert captured["runtime_config_file"] == "/tmp/runtime-config.yaml"
        assert captured["image"] == "test:img"
        assert captured["router_image"] == "test:router"
        assert captured["envoy_image"] == "test:envoy"
        assert captured["dashboard_image"] == "test:dashboard"
        assert captured["topology"] == "split"
        assert captured["pull_policy"] == "always"
        assert captured["enable_observability"] is False

    def test_teardown_delegates_to_stop_vllm_sr(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            "cli.docker_backend.stop_vllm_sr",
            lambda: called.append(True),
        )
        DockerBackend().teardown()
        assert called

    def test_is_running_false_when_not_found(self, monkeypatch):
        monkeypatch.setattr(
            "cli.docker_backend.docker_container_status",
            lambda _: "not found",
        )
        assert DockerBackend().is_running() is False

    def test_is_running_true_when_split_dashboard_container_running(self, monkeypatch):
        monkeypatch.setattr(
            "cli.docker_backend.docker_container_status",
            lambda name: "running" if "dashboard" in name else "not found",
        )
        assert DockerBackend().is_running() is True

    def test_get_dashboard_url_prefers_split_dashboard_container(self, monkeypatch):
        monkeypatch.setattr(
            "cli.docker_backend.docker_container_status",
            lambda name: "running" if "dashboard" in name else "not found",
        )
        assert DockerBackend().get_dashboard_url() == "http://localhost:8700"


# ---------------------------------------------------------------------------
# Config translator
# ---------------------------------------------------------------------------


class TestConfigTranslator:
    def test_image_override(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": [{"port": 8899}]}))

        values = translate_config_to_helm_values(
            str(config),
            image="myrepo/myimage:v2",
        )
        assert values["image"]["repository"] == "myrepo/myimage"
        assert values["image"]["tag"] == "v2"

    def test_pull_policy_normalised(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(
            str(config),
            pull_policy="always",
        )
        assert values["image"]["pullPolicy"] == "Always"

    def test_observability_flags(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        enabled = translate_config_to_helm_values(
            str(config), enable_observability=True
        )
        assert enabled["dependencies"]["observability"]["jaeger"]["enabled"] is True

        disabled = translate_config_to_helm_values(
            str(config), enable_observability=False
        )
        assert disabled["dependencies"]["observability"]["jaeger"]["enabled"] is False

    def test_config_sections_pass_through(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "listeners": [{"port": 8899}],
                    "decisions": [{"name": "default"}],
                    "bert_model": {"model_id": "test"},
                }
            )
        )

        values = translate_config_to_helm_values(str(config))
        assert values["config"]["listeners"] == [{"port": 8899}]
        assert values["config"]["decisions"] == [{"name": "default"}]
        assert values["config"]["bert_model"]["model_id"] == "test"

    def test_write_helm_values_file(self, tmp_path):
        expected_replicas = 3
        values = {"replicaCount": expected_replicas, "image": {"tag": "v1"}}
        path = write_helm_values_file(values, str(tmp_path))

        written = yaml.safe_load(Path(path).read_text())
        assert written["replicaCount"] == expected_replicas

    def test_sensitive_env_vars_excluded_from_plain_env(self, tmp_path):
        """Sensitive vars (masked=True in PASSTHROUGH_ENV_RULES) must not leak into plain env."""
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        env_vars = {
            "HF_TOKEN": "hf_test123",
            "OPENAI_API_KEY": "sk-test",
            "HF_ENDPOINT": "https://huggingface.co",
        }
        values = translate_config_to_helm_values(str(config), env_vars=env_vars)
        env_list = values.get("env", [])
        names = {e["name"] for e in env_list}
        assert "HF_TOKEN" not in names, "Sensitive var leaked into plain env"
        assert "OPENAI_API_KEY" not in names, "Sensitive var leaked into plain env"
        assert "HF_ENDPOINT" in names, "Non-sensitive var should be in plain env"

    def test_env_secret_name_added_to_values(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(
            str(config), env_vars={"HF_ENDPOINT": "x"}, env_secret_name="my-secret"
        )
        assert "my-secret" in values["envFromSecrets"]

    def test_env_vars_none_produces_no_env_key(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(str(config), env_vars=None)
        assert "env" not in values
        assert "envFromSecrets" not in values

    def test_deep_merge(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        overrides = {"a": {"b": 99}, "e": 4}
        merged = _deep_merge(base, overrides)
        assert merged == {"a": {"b": 99, "c": 2}, "d": 3, "e": 4}


# ---------------------------------------------------------------------------
# K8sBackend (unit tests — no real cluster needed)
# ---------------------------------------------------------------------------


class TestK8sBackend:
    def test_require_tool_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(SystemExit):
            K8sBackend._require_tool("helm")

    def test_helm_base_cmd_includes_context(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.context = "my-ctx"
        assert backend._helm_base_cmd() == ["helm", "--kube-context", "my-ctx"]

    def test_helm_base_cmd_without_context(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.context = None
        assert backend._helm_base_cmd() == ["helm"]

    def test_label_for_service_router(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.release_name = "sr"
        assert "semantic-router" in backend._label_for_service("router")

    def test_label_for_service_all(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.release_name = "sr"
        assert "sr" in backend._label_for_service("all")

    def test_sync_env_secret_returns_none_when_no_sensitive_vars(self, monkeypatch):
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        result = backend._sync_env_secret({"HF_ENDPOINT": "https://hf.co"})
        assert result is None

    def test_sync_env_secret_returns_none_when_empty(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        assert backend._sync_env_secret(None) is None
        assert backend._sync_env_secret({}) is None

    def test_sync_env_secret_creates_secret(self, monkeypatch):
        cmds_run = []

        def fake_run(cmd, **kwargs):
            cmds_run.append(cmd)
            return MagicMock(returncode=0, stdout="")

        monkeypatch.setattr("subprocess.run", fake_run)

        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        result = backend._sync_env_secret({"HF_TOKEN": "hf_secret123"})

        assert result == "vllm-sr-env-secrets"
        create_cmds = [c for c in cmds_run if "create" in c and "secret" in c]
        assert len(create_cmds) == 1
        assert "--from-literal=HF_TOKEN=hf_secret123" in create_cmds[0]


# ---------------------------------------------------------------------------
# CLI integration — serve --target routes to correct backend
# ---------------------------------------------------------------------------


from cli.commands import runtime as rt  # noqa: E402
from cli.main import main  # noqa: E402
from click.testing import CliRunner  # noqa: E402


class TestCLITargetRouting:
    def test_serve_default_target_builds_docker_backend(self, monkeypatch):
        built = []

        class _FakeDocker:
            def deploy(self, **kw):
                pass

        def _fake_build(target, **kw):
            built.append(resolve_target(target))
            return _FakeDocker()

        monkeypatch.setattr(rt, "_build_backend", _fake_build)
        monkeypatch.setattr(
            rt,
            "ensure_bootstrap_workspace",
            lambda _: MagicMock(config_path=Path("/dev/null"), setup_mode=False),
        )
        monkeypatch.setattr(
            rt,
            "resolve_effective_config_path",
            lambda *a: Path("/dev/null"),
        )

        runner = CliRunner()
        runner.invoke(
            main,
            ["serve", "--config", "/dev/null", "--image-pull-policy", "never"],
        )

        assert built and built[0] == "docker"

    def test_stop_target_k8s_builds_k8s_backend(self, monkeypatch):
        built = []

        class _FakeK8s:
            def teardown(self):
                pass

        def _fake_build(target, **kw):
            built.append(resolve_target(target))
            return _FakeK8s()

        monkeypatch.setattr(rt, "_build_backend", _fake_build)

        runner = CliRunner()
        runner.invoke(main, ["stop", "--target", "k8s"])

        assert built and built[0] == "k8s"
