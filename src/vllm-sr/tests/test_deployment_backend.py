"""Tests for deployment target resolution, config translation, and containers."""

import stat
import sys
from pathlib import Path

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
from cli.container_backend import ContainerBackend  # noqa: E402
from cli.deployment_backend import DEFAULT_TARGET, resolve_target  # noqa: E402

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
# ContainerBackend
# ---------------------------------------------------------------------------


class TestContainerBackend:
    def test_deploy_delegates_to_start_vllm_sr(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "cli.container_backend.start_vllm_sr",
            lambda *a, **kw: captured.update(kw),
        )

        backend = ContainerBackend()
        backend.deploy(
            config_file="/tmp/config.yaml",
            source_config_file="/tmp/source-config.yaml",
            runtime_config_file="/tmp/runtime-config.yaml",
            env_vars={"A": "B"},
            image="test:img",
            router_image="test:router",
            envoy_image="test:envoy",
            dashboard_image="test:dashboard",
            sim_image="test:sim",
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
        assert captured["sim_image"] == "test:sim"
        assert captured["topology"] == "split"
        assert captured["pull_policy"] == "always"
        assert captured["enable_observability"] is False

    def test_teardown_delegates_to_stop_vllm_sr(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            "cli.container_backend.stop_vllm_sr",
            lambda: called.append(True),
        )
        ContainerBackend().teardown()
        assert called

    def test_is_running_false_when_not_found(self, monkeypatch):
        monkeypatch.setattr(
            "cli.container_backend.container_status",
            lambda _: "not found",
        )
        assert ContainerBackend().is_running() is False

    def test_is_running_true_when_split_dashboard_container_running(self, monkeypatch):
        monkeypatch.setattr(
            "cli.container_backend.container_status",
            lambda name: "running" if "dashboard" in name else "not found",
        )
        assert ContainerBackend().is_running() is True

    def test_get_dashboard_url_prefers_split_dashboard_container(self, monkeypatch):
        monkeypatch.setattr(
            "cli.container_backend.container_status",
            lambda name: "running" if "dashboard" in name else "not found",
        )
        assert ContainerBackend().get_dashboard_url() == "http://localhost:8700"


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
        assert stat.S_IMODE(Path(path).stat().st_mode) == 0o600

    def test_write_helm_values_file_keeps_caller_directory(self, tmp_path):
        caller_directory = tmp_path / "caller-owned"
        caller_directory.mkdir(mode=0o755)
        caller_directory.chmod(0o755)
        existing_values_path = caller_directory / "values-override.yaml"
        existing_values_path.write_text("replicaCount: 99\n")
        existing_values_path.chmod(0o644)

        values_path = Path(
            write_helm_values_file({"replicaCount": 1}, str(caller_directory))
        )

        assert caller_directory.is_dir()
        assert stat.S_IMODE(caller_directory.stat().st_mode) == 0o755
        assert stat.S_IMODE(values_path.stat().st_mode) == 0o600
        assert yaml.safe_load(values_path.read_text())["replicaCount"] == 1

    def test_write_helm_values_file_creates_private_directory(self):
        values_path = Path(write_helm_values_file({"replicaCount": 1}))
        values_directory = values_path.parent
        try:
            assert stat.S_IMODE(values_directory.stat().st_mode) == 0o700
            assert stat.S_IMODE(values_path.stat().st_mode) == 0o600
        finally:
            values_path.unlink(missing_ok=True)
            values_directory.rmdir()

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

    def test_router_log_level_env_vars_are_included_in_plain_env(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(
            str(config),
            env_vars={"SR_LOG_LEVEL": "debug", "SR_LOG_ENCODING": "console"},
        )
        env_entries = {entry["name"]: entry["value"] for entry in values["env"]}

        assert env_entries["SR_LOG_LEVEL"] == "debug"
        assert env_entries["SR_LOG_ENCODING"] == "console"

    def test_env_secret_name_added_to_values(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(
            str(config), env_vars={"HF_ENDPOINT": "x"}, env_secret_name="my-secret"
        )
        assert "my-secret" in values["envFromSecrets"]

    def test_profile_env_secrets_are_preserved_before_cli_secret_append(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(
            str(config),
            profile_values={
                "envFromSecrets": ["operator-managed", "cli-runtime"],
            },
            env_secret_name="cli-runtime",
        )

        assert values["envFromSecrets"] == ["operator-managed", "cli-runtime"]

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
