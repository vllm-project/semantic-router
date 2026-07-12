"""Tests for the deployment backend abstraction and K8s / Docker wiring."""

import base64
import json
import subprocess
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
from cli.container_backend import ContainerBackend  # noqa: E402
from cli.deployment_backend import DEFAULT_TARGET, resolve_target  # noqa: E402
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


# K8sBackend (unit tests — no real cluster needed)


def _k8s_backend(release_name: str = "alpha") -> K8sBackend:
    return K8sBackend(
        namespace="shared-ns",
        context=None,
        release_name=release_name,
        chart_dir="/chart",
    )


def _managed_secret_payload(
    name: str,
    release_name: str,
    data: dict[str, str],
) -> dict:
    encoded = {
        key: base64.b64encode(value.encode()).decode() for key, value in data.items()
    }
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "labels": {
                "app.kubernetes.io/managed-by": "vllm-sr-cli",
                "app.kubernetes.io/instance": release_name,
                "semantic-router.vllm.ai/runtime-env-secret": "true",
            },
        },
        "immutable": True,
        "data": encoded,
    }


def _deployment_with_secret(release_name: str, secret_name: str) -> dict:
    return {
        "metadata": {
            "name": release_name,
            "labels": {"app.kubernetes.io/instance": release_name},
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "router",
                            "envFrom": [{"secretRef": {"name": secret_name}}],
                        }
                    ]
                }
            }
        },
    }


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

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("[]", False),
            ('[{"name":"alpha","status":"deployed"}]', True),
        ],
    )
    def test_helm_release_state_requires_exact_structured_match(
        self, monkeypatch, payload, expected
    ):
        backend = _k8s_backend()
        calls = []
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kwargs: (
                calls.append((cmd, kwargs))
                or MagicMock(returncode=0, stdout=payload, stderr="")
            ),
        )

        assert backend._helm_release_exists() is expected
        assert calls[0][0][1:3] == ["list", "--all"]
        assert calls[0][1]["capture_output"] is True

    @pytest.mark.parametrize(
        "result",
        [
            MagicMock(returncode=1, stdout="", stderr="sensitive-server-output"),
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="not-json", stderr=""),
            MagicMock(returncode=0, stdout="{}", stderr=""),
            MagicMock(
                returncode=0,
                stdout='[{"name":"beta","status":"deployed"}]',
                stderr="",
            ),
            MagicMock(returncode=0, stdout='[{"name":"alpha"}]', stderr=""),
        ],
    )
    def test_helm_release_state_fails_closed_without_output_leak(
        self, monkeypatch, caplog, result
    ):
        backend = _k8s_backend()
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: result)

        with pytest.raises(RuntimeError, match="Helm"):
            backend._helm_release_exists()

        assert "sensitive-server-output" not in caplog.text

    def test_label_for_service_router(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.release_name = "sr"
        assert "semantic-router" in backend._label_for_service("router")

    def test_label_for_service_all(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.release_name = "sr"
        assert "sr" in backend._label_for_service("all")

    def test_secret_plan_generates_looper_key_for_empty_inputs(self, monkeypatch):
        backend = _k8s_backend()
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])

        for env_vars in (None, {}, {"HF_ENDPOINT": "https://hf.co"}):
            plan = backend._plan_env_secret(env_vars)
            data = json.loads(plan.new_manifest)["data"]
            generated = base64.b64decode(data["VLLM_SR_LOOPER_SHARED_SECRET"]).decode()
            assert len(generated) == 64 and int(generated, 16) >= 0
            assert plan.creates_secret is True
            assert plan.recreate_for_looper_rotation is True

    def test_create_uses_immutable_stdin_manifest_without_argv_leak(
        self, monkeypatch, caplog
    ):
        backend = _k8s_backend()
        calls = []
        secret_value = "hf_secret123"
        generated_looper = "a" * 64
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            "cli.k8s_backend.secrets.token_hex",
            lambda size: generated_looper if size == 32 else "b" * 12,
        )
        monkeypatch.setattr(backend, "_ensure_namespace", lambda: None)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: (
                calls.append((cmd, kwargs)) or MagicMock(returncode=0)
            ),
        )

        plan = backend._plan_env_secret({"HF_TOKEN": secret_value})
        created_name = backend._create_planned_secret(plan)

        assert created_name == plan.active_name
        assert plan.active_name.startswith("alpha-vsr-env-")
        assert secret_value not in repr(plan)
        cmd, kwargs = calls[0]
        assert cmd[-3:] == ["create", "-f", "-"]
        manifest = json.loads(kwargs["input_text"])
        assert manifest["immutable"] is True
        assert manifest["data"] == backend._encode_secret_data(
            {
                "HF_TOKEN": secret_value,
                "VLLM_SR_LOOPER_SHARED_SECRET": generated_looper,
            }
        )
        assert manifest["metadata"]["labels"]["app.kubernetes.io/instance"] == "alpha"
        leaked = [secret_value, generated_looper, *manifest["data"].values()]
        assert all(
            value not in " ".join(cmd) and value not in caplog.text for value in leaked
        )

    def test_empty_explicit_looper_secret_is_rejected(self, monkeypatch):
        backend = _k8s_backend()
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])

        with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
            backend._plan_env_secret({"VLLM_SR_LOOPER_SHARED_SECRET": ""})

    def test_unchanged_exact_data_reuses_current_generation(self, monkeypatch):
        backend = _k8s_backend()
        current_name = "alpha-vsr-env-current"
        payload = _managed_secret_payload(
            current_name,
            "alpha",
            {
                "HF_TOKEN": "same",
                "OPENAI_API_KEY": "same-key",
                "VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64,
            },
        )
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)

        plan = backend._plan_env_secret(
            {"HF_TOKEN": "same", "OPENAI_API_KEY": "same-key"}
        )

        assert plan.active_name == current_name
        assert plan.creates_secret is False
        assert plan.recreate_for_looper_rotation is True

    def test_exact_key_set_change_creates_new_generation(self, monkeypatch):
        backend = _k8s_backend()
        current_name = "alpha-vsr-env-current"
        payload = _managed_secret_payload(
            current_name,
            "alpha",
            {
                "HF_TOKEN": "same",
                "OPENAI_API_KEY": "removed",
                "VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64,
            },
        )
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)

        plan = backend._plan_env_secret({"HF_TOKEN": "same"})
        manifest = json.loads(plan.new_manifest)

        assert plan.active_name != current_name
        assert manifest["data"] == backend._encode_secret_data(
            {"HF_TOKEN": "same", "VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64}
        )
        assert plan.recreate_for_looper_rotation is True

    def test_release_scoped_secrets_do_not_cross_releases(self, monkeypatch):
        alpha = _k8s_backend("alpha")
        beta = _k8s_backend("beta")
        monkeypatch.setattr(alpha, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(beta, "_current_router_secret_refs", lambda: [])

        alpha_plan = alpha._plan_env_secret({"HF_TOKEN": "alpha-token"})
        beta_plan = beta._plan_env_secret({"HF_TOKEN": "beta-token"})
        alpha_manifest = json.loads(alpha_plan.new_manifest)
        beta_manifest = json.loads(beta_plan.new_manifest)

        assert alpha_plan.active_name != beta_plan.active_name
        assert (
            alpha_manifest["metadata"]["labels"]["app.kubernetes.io/instance"]
            == "alpha"
        )
        assert (
            beta_manifest["metadata"]["labels"]["app.kubernetes.io/instance"] == "beta"
        )

    def test_deploy_orders_create_atomic_helm_then_bounded_gc(
        self, monkeypatch, tmp_path
    ):
        backend = _k8s_backend()
        config = tmp_path / "config.yaml"
        config.write_text("listeners: []\n")
        current_name = "alpha-vsr-env-current"
        old = "a" * 64
        new = "b" * 64
        payload = _managed_secret_payload(
            current_name,
            "alpha",
            {"VLLM_SR_LOOPER_SHARED_SECRET": old},
        )
        events = []
        written_values = {}
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)
        monkeypatch.setattr(
            "cli.k8s_backend.write_helm_values_file",
            lambda values: written_values.update(values) or "/tmp/values.yaml",
        )

        def create(plan):
            events.append(("create", plan.active_name))
            return plan.active_name

        def run(cmd, **kwargs):
            events.append(("helm", cmd))
            return MagicMock(returncode=0)

        monkeypatch.setattr(backend, "_create_planned_secret", create)
        monkeypatch.setattr(backend, "_run", run)
        monkeypatch.setattr(
            backend,
            "_garbage_collect_managed_secrets",
            lambda: events.append(("gc", None)) or True,
        )
        monkeypatch.setattr(backend, "_wait_for_pods", lambda: None)
        monkeypatch.setattr(backend, "_log_k8s_summary", lambda: None)

        backend.deploy(
            str(config),
            env_vars={"VLLM_SR_LOOPER_SHARED_SECRET": new},
        )

        assert [event[0] for event in events] == ["create", "helm", "gc"]
        helm_cmd = events[1][1]
        assert "--atomic" in helm_cmd
        assert "--wait" in helm_cmd
        assert helm_cmd[helm_cmd.index("--history-max") + 1] == "10"
        assert written_values["runtimeCredentials"]["recreateForLooperRotation"] is True
        serialized_values = json.dumps(written_values)
        assert old not in serialized_values
        assert new not in serialized_values

    def test_atomic_failure_deletes_only_new_secret_and_retains_old(
        self, monkeypatch, tmp_path
    ):
        backend = _k8s_backend()
        config = tmp_path / "config.yaml"
        config.write_text("listeners: []\n")
        current_name = "alpha-vsr-env-current"
        payload = _managed_secret_payload(
            current_name,
            "alpha",
            {"HF_TOKEN": "old"},
        )
        events = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)
        monkeypatch.setattr(
            "cli.k8s_backend.write_helm_values_file", lambda _: "/tmp/values.yaml"
        )

        def create(plan):
            events.append(("create", plan.active_name))
            return plan.active_name

        def fail_helm(cmd, **kwargs):
            events.append(("helm", cmd))
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(backend, "_create_planned_secret", create)
        monkeypatch.setattr(backend, "_run", fail_helm)
        monkeypatch.setattr(
            backend,
            "_delete_managed_secret_if_unreferenced",
            lambda name: events.append(("delete", name)) or True,
        )

        with pytest.raises(subprocess.CalledProcessError):
            backend.deploy(str(config), env_vars={"HF_TOKEN": "new"})

        assert [event[0] for event in events] == ["create", "helm", "delete"]
        assert events[-1][1] == events[0][1]
        assert current_name not in [
            event[1] for event in events if event[0] == "delete"
        ]

    def test_cleanup_rechecks_latest_rollback_references(self, monkeypatch):
        backend = _k8s_backend()
        name = "alpha-vsr-env-concurrent"
        refs = [name]
        deleted = []
        monkeypatch.setattr(backend, "_protected_secret_refs", lambda: set(refs))
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value: deleted.append(value) or True,
        )

        backend._delete_managed_secret_if_unreferenced(name)
        refs.clear()
        backend._delete_managed_secret_if_unreferenced(name)

        assert deleted == [name]

    def test_omitted_optional_credentials_reuse_looper_generation(
        self, monkeypatch, tmp_path
    ):
        backend = _k8s_backend()
        config = tmp_path / "config.yaml"
        config.write_text("listeners: []\n")
        current_name = "alpha-vsr-env-current"
        payload = _managed_secret_payload(
            current_name,
            "alpha",
            {"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64},
        )
        events = []
        values = {}
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)
        monkeypatch.setattr(
            "cli.k8s_backend.write_helm_values_file",
            lambda rendered: values.update(rendered) or "/tmp/values.yaml",
        )
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: events.append("helm") or MagicMock(returncode=0),
        )
        monkeypatch.setattr(
            backend,
            "_garbage_collect_managed_secrets",
            lambda: events.append("gc") or True,
        )
        monkeypatch.setattr(backend, "_wait_for_pods", lambda: None)
        monkeypatch.setattr(backend, "_log_k8s_summary", lambda: None)

        backend.deploy(str(config), env_vars=None)

        assert events == ["helm", "gc"]
        assert values["runtimeCredentials"] == {
            "generation": current_name,
            "recreateForLooperRotation": True,
        }

    def test_teardown_uninstalls_before_release_scoped_cleanup(
        self, monkeypatch, caplog
    ):
        backend = _k8s_backend()
        events = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: ["vllm-sr-env-secrets"]
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_names",
            lambda: ["alpha-vsr-env-old"],
        )
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: (
                events.append(("helm", cmd)) or MagicMock(returncode=0)
            ),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda name: events.append(("delete", name)) or True,
        )
        backend.teardown()

        assert [event[0] for event in events] == ["helm", "delete"]
        assert events[1][1] == "alpha-vsr-env-old"
        assert "--wait" in events[0][1]
        assert "never deletes this unowned compatibility Secret" in caplog.text
        assert "all live workload references" in caplog.text
        assert "retained Helm revisions for every release" in caplog.text

    def test_failed_teardown_retains_all_secrets(self, monkeypatch):
        backend = _k8s_backend()
        deleted = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: ["vllm-sr-env-secrets"]
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_names",
            lambda: ["alpha-vsr-env-old"],
        )
        monkeypatch.setattr(
            backend, "_run", lambda cmd, **kwargs: MagicMock(returncode=1)
        )
        monkeypatch.setattr(backend, "_delete_secret_if_exists", deleted.append)

        with pytest.raises(RuntimeError, match="Helm uninstall failed"):
            backend.teardown()

        assert deleted == []

    def test_teardown_delete_failure_tries_every_secret_then_fails(self, monkeypatch):
        backend = _k8s_backend()
        attempted = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_names",
            lambda: ["alpha-vsr-env-fails", "alpha-vsr-env-succeeds"],
        )
        monkeypatch.setattr(
            backend, "_run", lambda *args, **kwargs: MagicMock(returncode=0)
        )

        def delete(name):
            attempted.append(name)
            return name.endswith("succeeds")

        monkeypatch.setattr(backend, "_delete_secret_if_exists", delete)

        with pytest.raises(RuntimeError, match="cleanup was incomplete"):
            backend.teardown()

        assert attempted == ["alpha-vsr-env-fails", "alpha-vsr-env-succeeds"]

    def test_second_stop_retries_cleanup_after_release_is_proven_absent(
        self, monkeypatch
    ):
        backend = _k8s_backend()
        release_states = iter([True, False])
        delete_results = iter([False, True])
        attempted = []
        uninstall_calls = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_helm_release_exists", lambda: next(release_states)
        )
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_names",
            lambda: ["alpha-vsr-env-stale"],
        )
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: (
                uninstall_calls.append(args[0]) or MagicMock(returncode=0)
            ),
        )

        def delete(name):
            attempted.append(name)
            return next(delete_results)

        monkeypatch.setattr(backend, "_delete_secret_if_exists", delete)

        with pytest.raises(RuntimeError, match="cleanup was incomplete"):
            backend.teardown()
        backend.teardown()

        assert len(uninstall_calls) == 1
        assert attempted == ["alpha-vsr-env-stale", "alpha-vsr-env-stale"]

    def test_absent_release_does_not_delete_a_still_referenced_secret(
        self, monkeypatch
    ):
        backend = _k8s_backend()
        live_name = "alpha-vsr-env-live"
        deleted = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: False)
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [live_name])
        monkeypatch.setattr(
            backend, "_list_release_managed_secret_names", lambda: [live_name]
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda name: deleted.append(name) or True,
        )

        with pytest.raises(RuntimeError, match="cleanup was incomplete"):
            backend.teardown()

        assert deleted == []

    def test_unknown_helm_release_state_never_starts_cleanup(self, monkeypatch):
        backend = _k8s_backend()
        events = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend,
            "_helm_release_exists",
            lambda: (_ for _ in ()).throw(RuntimeError("unknown state")),
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_names",
            lambda: events.append("list") or [],
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda name: events.append(("delete", name)) or True,
        )

        with pytest.raises(RuntimeError, match="unknown state"):
            backend.teardown()

        assert events == []

    def test_unknown_initial_kubectl_state_never_uninstalls_or_deletes(
        self, monkeypatch
    ):
        backend = _k8s_backend()
        events = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(
            backend,
            "_current_router_secret_refs",
            lambda: (_ for _ in ()).throw(RuntimeError("unknown kubectl state")),
        )
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: events.append("uninstall"),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda name: events.append(("delete", name)) or True,
        )

        with pytest.raises(RuntimeError, match="unknown kubectl state"):
            backend.teardown()

        assert events == []

    def test_release_secret_listing_filters_other_releases(self, monkeypatch):
        backend = _k8s_backend()
        alpha = _managed_secret_payload("alpha-vsr-env-one", "alpha", {})
        beta = _managed_secret_payload("beta-vsr-env-one", "beta", {})
        monkeypatch.setattr(
            backend,
            "_get_json",
            lambda cmd, **kwargs: {"items": [alpha, beta]},
        )

        assert backend._list_release_managed_secret_names() == ["alpha-vsr-env-one"]

    def test_teardown_never_deletes_namespace_global_legacy_secret(
        self, monkeypatch, caplog
    ):
        backend = _k8s_backend()
        commands = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        # Even a mistakenly release-labelled legacy Secret remains unowned and safe.
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_names",
            lambda: ["vllm-sr-env-secrets"],
        )
        monkeypatch.setattr(
            backend,
            "_helm_retained_secret_refs",
            lambda: pytest.fail("teardown must not infer global legacy ownership"),
        )
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: commands.append(cmd) or MagicMock(returncode=0),
        )

        backend.teardown()

        assert len(commands) == 1
        assert commands[0][1] == "uninstall"
        assert "namespace-global legacy runtime Secret" in caplog.text
        assert "vllm-sr-env-secrets" in caplog.text

    def test_kubectl_inspection_failure_is_fail_closed_without_output_leak(
        self, monkeypatch, caplog
    ):
        backend = _k8s_backend()
        sensitive_output = "server-returned-sensitive-output"
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: MagicMock(
                returncode=1,
                stdout="",
                stderr=sensitive_output,
            ),
        )

        with pytest.raises(RuntimeError, match="kubectl query failed"):
            backend._current_router_secret_refs()

        assert sensitive_output not in caplog.text


# CLI integration — serve --target routes to correct backend


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

    def test_stop_target_k8s_cleanup_failure_exits_nonzero(self, monkeypatch, caplog):
        class _FailingK8s:
            def teardown(self):
                raise RuntimeError("Runtime credential Secret cleanup was incomplete")

        monkeypatch.setattr(rt, "_build_backend", lambda *args, **kwargs: _FailingK8s())

        result = CliRunner().invoke(main, ["stop", "--target", "k8s"])

        assert result.exit_code == 1
        assert "cleanup was incomplete" in caplog.text
