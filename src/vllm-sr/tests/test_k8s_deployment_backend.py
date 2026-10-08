"""Tests for Kubernetes deployment backend and CLI target routing."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.commands import runtime as rt  # noqa: E402
from cli.deployment_backend import resolve_target  # noqa: E402
from cli.k8s_backend import (  # noqa: E402
    PUBLISHED_CHART,
    K8sBackend,
    client_service_port,
    published_chart_version,
)
from cli.main import main  # noqa: E402
from click.testing import CliRunner  # noqa: E402

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

    @pytest.mark.parametrize(
        "secret_value",
        [
            "literal-cluster-secret-canary",
            {"value": "literal-cluster-secret-canary"},
            ["literal-cluster-secret-canary"],
            42,
        ],
    )
    def test_literal_config_secret_fails_before_cluster_mutation(
        self, monkeypatch, tmp_path, secret_value
    ):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "providers": {"models": [{"api_key": secret_value}]},
                }
            ),
            encoding="utf-8",
        )
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.chart_dir = str(tmp_path)
        backend.release_name = "sr"
        backend.profile = None

        cluster_mutation = MagicMock(
            side_effect=AssertionError("cluster mutation was attempted")
        )
        monkeypatch.setattr(backend, "_require_tool", lambda _name: None)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)
        monkeypatch.setattr(backend, "_plan_env_secret", lambda *_args: None)
        monkeypatch.setattr(backend, "_ensure_namespace", cluster_mutation)
        monkeypatch.setattr(backend, "_run", cluster_mutation)

        with pytest.raises(ValueError, match=r"config.providers.models\[0\].api_key"):
            backend.deploy(config_file=str(config))

        assert cluster_mutation.call_count == 0

    @pytest.mark.parametrize("environment_name", ["HOME", "PATH", "VLLM_SR_PLATFORM"])
    def test_reserved_management_token_env_fails_before_cluster_mutation(
        self, monkeypatch, tmp_path, environment_name
    ):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "global": {
                        "services": {
                            "management_api": {
                                "auth": {
                                    "mode": "bearer",
                                    "tokens": [
                                        {"env": environment_name, "role": "viewer"}
                                    ],
                                }
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.chart_dir = str(tmp_path)
        backend.release_name = "sr"
        backend.profile = None

        cluster_mutation = MagicMock(
            side_effect=AssertionError("cluster mutation was attempted")
        )
        monkeypatch.setattr(backend, "_require_tool", lambda _name: None)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)
        monkeypatch.setattr(backend, "_ensure_namespace", cluster_mutation)
        monkeypatch.setattr(backend, "_run", cluster_mutation)

        with pytest.raises(ValueError, match="env name is invalid"):
            backend.deploy(
                config_file=str(config),
                env_vars={environment_name: "predictable-host-value"},
            )

        assert cluster_mutation.call_count == 0

    @pytest.mark.parametrize("env_field", ["env", "extraEnv"])
    def test_sensitive_profile_env_fails_before_cluster_mutation(
        self, monkeypatch, tmp_path, env_field
    ):
        canary = "profile-literal-secret-canary"
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "providers": {"models": [{"api_key": "${GEMINI_API_KEY}"}]},
                }
            ),
            encoding="utf-8",
        )
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.chart_dir = str(tmp_path)
        backend.release_name = "sr"
        backend.profile = "prod"

        cluster_mutation = MagicMock(
            side_effect=AssertionError("cluster mutation was attempted")
        )
        monkeypatch.setattr(backend, "_require_tool", lambda _name: None)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)
        monkeypatch.setattr(backend, "_plan_env_secret", lambda *_args: None)
        monkeypatch.setattr(
            "cli.k8s_backend.load_profile_values",
            lambda *_args: {env_field: [{"name": "GEMINI_API_KEY", "value": canary}]},
        )
        monkeypatch.setattr(backend, "_ensure_namespace", cluster_mutation)
        monkeypatch.setattr(backend, "_run", cluster_mutation)

        with pytest.raises(ValueError, match="GEMINI_API_KEY") as exc:
            backend.deploy(
                config_file=str(config),
                env_vars={"GEMINI_API_KEY": "host-secret-canary"},
            )

        assert canary not in str(exc.value)
        assert cluster_mutation.call_count == 0

    def test_existing_namespace_is_never_mutated(self, monkeypatch):
        backend = K8sBackend.__new__(K8sBackend)
        backend.context = None
        backend.namespace = "user-managed"
        commands = []

        def run(cmd, **kwargs):
            commands.append((cmd, kwargs))
            return subprocess.CompletedProcess(
                cmd, 0, stdout="namespace/user-managed\n"
            )

        monkeypatch.setattr(backend, "_run", run)
        backend._ensure_namespace()

        assert len(commands) == 1
        assert commands[0][0][1:4] == ["get", "namespace", "user-managed"]
        assert "apply" not in commands[0][0]
        assert commands[0][1] == {"check": False, "capture_output": True}

    def test_missing_namespace_is_created_without_apply(self, monkeypatch):
        backend = K8sBackend.__new__(K8sBackend)
        backend.context = None
        backend.namespace = "new-ns"
        commands = []
        results = iter(
            [
                subprocess.CompletedProcess([], 0, stdout=""),
                subprocess.CompletedProcess([], 0, stdout="namespace/new-ns\n"),
            ]
        )

        def run(cmd, **kwargs):
            commands.append((cmd, kwargs))
            return next(results)

        monkeypatch.setattr(backend, "_run", run)
        backend._ensure_namespace()

        assert [command[0][1] for command in commands] == ["get", "create"]
        assert all("apply" not in command for command, _kwargs in commands)

    def test_namespace_create_race_is_verified_and_other_failures_close(
        self, monkeypatch
    ):
        backend = K8sBackend.__new__(K8sBackend)
        backend.context = None
        backend.namespace = "raced-ns"
        results = iter(
            [
                subprocess.CompletedProcess([], 0, stdout=""),
                subprocess.CompletedProcess([], 1, stdout=""),
                subprocess.CompletedProcess([], 0, stdout="namespace/raced-ns\n"),
            ]
        )
        monkeypatch.setattr(backend, "_run", lambda *_args, **_kwargs: next(results))
        backend._ensure_namespace()

        failed_results = iter(
            [
                subprocess.CompletedProcess([], 0, stdout=""),
                subprocess.CompletedProcess([], 1, stdout=""),
                subprocess.CompletedProcess([], 1, stdout=""),
            ]
        )
        monkeypatch.setattr(
            backend, "_run", lambda *_args, **_kwargs: next(failed_results)
        )
        with pytest.raises(RuntimeError, match="Failed to create"):
            backend._ensure_namespace()

    def test_teardown_writes_final_result_to_stdout(self, monkeypatch, capsys):
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.release_name = "sr"

        monkeypatch.setattr(backend, "_require_tool", lambda _name: None)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=0),
        )
        monkeypatch.setattr(backend, "_current_release_env_secret_refs", set)
        monkeypatch.setattr(backend, "_list_managed_env_secrets", set)
        cleanup = MagicMock()
        monkeypatch.setattr(backend, "_cleanup_obsolete_env_secrets", cleanup)

        backend.teardown()

        captured = capsys.readouterr()
        assert captured.out == "✓ Helm release uninstalled\n"
        assert "Uninstalling Helm release: sr" in captured.err
        cleanup.assert_called_once_with(
            active_secret_name=None,
            previous_secret_refs=set(),
            previous_managed_secrets=set(),
        )

    def test_teardown_does_not_report_success_when_helm_fails(
        self, monkeypatch, capsys
    ):
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.release_name = "sr"

        monkeypatch.setattr(backend, "_require_tool", lambda _name: None)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )
        monkeypatch.setattr(backend, "_current_release_env_secret_refs", set)
        monkeypatch.setattr(backend, "_list_managed_env_secrets", set)

        with pytest.raises(RuntimeError, match="uninstall failed"):
            backend.teardown()

        assert "✓ Helm release uninstalled" not in capsys.readouterr().out

    def test_logs_propagate_kubectl_failure(self, monkeypatch):
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.release_name = "sr"

        monkeypatch.setattr(backend, "_require_tool", lambda _name: None)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=23),
        )

        with pytest.raises(SystemExit) as raised:
            backend.logs("router")
        assert raised.value.code == 23

    def test_status_propagates_display_command_failure(self, monkeypatch):
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.release_name = "sr"

        monkeypatch.setattr(backend, "_require_tool", lambda _name: None)
        results = iter((0, 17))
        monkeypatch.setattr(
            backend,
            "_run_display",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                args=[], returncode=next(results)
            ),
        )

        with pytest.raises(SystemExit) as raised:
            backend.status()
        assert raised.value.code == 17

    def test_wait_for_pods_failure_raises_and_prints_diagnostics(self, monkeypatch):
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.release_name = "sr"

        diagnostic_calls = []
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                args=[], returncode=1
            ),
        )
        monkeypatch.setattr(backend, "_run_display", diagnostic_calls.append)

        with pytest.raises(SystemExit) as raised:
            backend._wait_for_pods()
        assert raised.value.code == 1
        assert diagnostic_calls, "failed wait must print pod diagnostics"
        assert any("get" in str(part) for part in diagnostic_calls[0])

    def test_wait_for_pods_success_stays_silent(self, monkeypatch):
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.release_name = "sr"

        diagnostic_calls = []
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                args=[], returncode=0
            ),
        )
        monkeypatch.setattr(backend, "_run_display", diagnostic_calls.append)

        backend._wait_for_pods()
        assert diagnostic_calls == []

    def test_label_for_service_router(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.release_name = "sr"
        assert backend._label_for_service("router") == (
            "app.kubernetes.io/instance=sr,app.kubernetes.io/component=router"
        )

    def test_label_for_service_all(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.release_name = "sr"
        assert "sr" in backend._label_for_service("all")

    def test_status_and_dashboard_url_are_release_scoped(self, monkeypatch):
        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.release_name = "router-b"
        monkeypatch.setattr(backend, "_require_tool", lambda _name: None)

        display_commands = []

        def display(cmd):
            display_commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(backend, "_run_display", display)
        backend.status("dashboard")
        pod_cmd = display_commands[0]
        assert pod_cmd[pod_cmd.index("-l") + 1] == (
            "app.kubernetes.io/instance=router-b,app.kubernetes.io/component=dashboard"
        )

        subprocess_commands = []

        def run(cmd, **_kwargs):
            subprocess_commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="10.0.0.1:8700")

        monkeypatch.setattr(subprocess, "run", run)
        assert backend.get_dashboard_url() == "http://10.0.0.1:8700"
        service_cmd = subprocess_commands[0]
        assert service_cmd[service_cmd.index("-l") + 1] == (
            "app.kubernetes.io/instance=router-b,app.kubernetes.io/component=dashboard"
        )

    def test_deploy_checks_sensitivity_against_source_not_effective_config(
        self, monkeypatch, tmp_path
    ):
        """deploy() must classify env vars using source_config_file, not the
        (possibly rewritten) effective config_file it hands to Helm."""
        source_config = tmp_path / "config.yaml"
        source_config.write_text(
            yaml.safe_dump(
                {"providers": {"models": [{"api_key_env": "GEMINI_API_KEY"}]}}
            )
        )
        effective_config = tmp_path / "runtime-config.yaml"
        effective_config.write_text(yaml.safe_dump({"listeners": []}))

        backend = K8sBackend.__new__(K8sBackend)
        backend.namespace = "test-ns"
        backend.context = None
        backend.chart_dir = str(tmp_path)
        backend.release_name = "sr"
        backend.profile = None

        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr("cli.k8s_backend.print_vllm_logo", lambda: None)
        monkeypatch.setattr(backend, "_ensure_namespace", lambda: None)
        monkeypatch.setattr(backend, "_current_release_env_secret_refs", set)
        monkeypatch.setattr(backend, "_list_managed_env_secrets", set)
        monkeypatch.setattr(
            backend, "_cleanup_obsolete_env_secrets", lambda **_kwargs: None
        )
        monkeypatch.setattr(backend, "_run", lambda *a, **kw: None)
        monkeypatch.setattr(backend, "_wait_for_pods", lambda: None)
        monkeypatch.setattr(backend, "_log_k8s_summary", lambda _port: None)
        monkeypatch.setattr(
            "cli.k8s_backend.load_profile_values", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            "cli.config_translator.write_helm_values_file",
            lambda *a, **kw: str(tmp_path),
        )

        secret_cmds = []
        monkeypatch.setattr(
            backend,
            "_plan_env_secret",
            lambda env_vars, config_file=None: secret_cmds.append(config_file) or None,
        )

        backend.deploy(
            config_file=str(effective_config),
            source_config_file=str(source_config),
            env_vars={"GEMINI_API_KEY": "gk-test"},
        )

        assert secret_cmds == [str(source_config)]


# ---------------------------------------------------------------------------
# CLI integration — serve --target routes to correct backend
# ---------------------------------------------------------------------------


class TestCLITargetRouting:
    def test_serve_default_target_builds_docker_backend(self, monkeypatch, tmp_path):
        built = []
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "version: v0.3\nlisteners:\n  - name: http\n    address: 0.0.0.0\n    port: 8899\n",
            encoding="utf-8",
        )

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
            lambda _: MagicMock(config_path=config_path, setup_mode=False),
        )

        runner = CliRunner()
        runner.invoke(
            main,
            [
                "serve",
                "--config",
                str(config_path),
                "--image-pull-policy",
                "never",
            ],
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

        assert built and built[0] == "kubernetes"

    def test_k8s_runtime_mode_flags_reach_backend(self, monkeypatch, tmp_path):
        captured = {}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "version: v0.3\nlisteners:\n  - name: http\n    address: 0.0.0.0\n    port: 8899\n",
            encoding="utf-8",
        )

        class _FakeK8s:
            def deploy(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(rt, "_build_backend", lambda *_args, **_kwargs: _FakeK8s())
        monkeypatch.setattr(
            rt,
            "ensure_bootstrap_workspace",
            lambda _: MagicMock(config_path=config_path, setup_mode=False),
        )

        result = CliRunner().invoke(
            main,
            [
                "serve",
                "--config",
                str(config_path),
                "--target",
                "k8s",
                "--minimal",
                "--readonly",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["minimal"] is True
        assert captured["readonly"] is True
        assert captured["enable_observability"] is False

    @pytest.mark.parametrize("config_state", ["missing", "setup"])
    def test_k8s_requires_existing_non_setup_config_without_workspace_mutation(
        self, monkeypatch, tmp_path, config_state
    ):
        config_path = tmp_path / "config.yaml"
        if config_state == "setup":
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "version": "v0.3",
                        "setup": {"mode": True, "state": "bootstrap"},
                    }
                ),
                encoding="utf-8",
            )
        before = {path.name for path in tmp_path.iterdir()}
        bootstrap = MagicMock(side_effect=AssertionError("bootstrap was called"))
        backend_builder = MagicMock(side_effect=AssertionError("backend was called"))
        monkeypatch.setattr(rt, "ensure_bootstrap_workspace", bootstrap)
        monkeypatch.setattr(rt, "_build_backend", backend_builder)

        result = CliRunner().invoke(
            main,
            ["serve", "--config", str(config_path), "--target", "k8s"],
        )

        assert result.exit_code == 1
        assert "Kubernetes deployment" in result.output
        bootstrap.assert_not_called()
        backend_builder.assert_not_called()
        assert {path.name for path in tmp_path.iterdir()} == before
        assert not (tmp_path / ".vllm-sr").exists()

    def test_serve_k8s_does_not_inject_local_runtime_defaults(
        self, monkeypatch, tmp_path
    ):
        captured = {}
        config_path = tmp_path / "config.yaml"
        source = {
            "version": "v0.3",
            "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
            "global": {
                "services": {},
                "stores": {},
                "integrations": {"looper": {}},
            },
        }
        config_path.write_text(yaml.safe_dump(source), encoding="utf-8")

        class _FakeK8s:
            def deploy(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(rt, "_build_backend", lambda *_args, **_kwargs: _FakeK8s())
        monkeypatch.setattr(
            rt,
            "ensure_bootstrap_workspace",
            lambda _: MagicMock(config_path=config_path, setup_mode=False),
        )

        result = CliRunner().invoke(
            main,
            ["serve", "--config", str(config_path), "--target", "k8s"],
        )

        assert result.exit_code == 0, result.output
        assert Path(captured["config_file"]).resolve() == config_path.resolve()
        effective = captured["config_document"]
        assert effective["global"]["services"] == {}
        assert effective["global"]["stores"] == {}
        assert effective["global"]["integrations"] == {"looper": {}}
        assert effective["global"]["router"]["enabled"] is True
        assert effective["global"]["model_catalog"]["system"]["decision_model"] == {
            "deployment": "primary"
        }
        assert yaml.safe_load(config_path.read_text()) == source
        assert not (tmp_path / ".vllm-sr").exists()

    def test_k8s_overrides_never_publish_local_runtime_state(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.delenv("VLLM_SR_ROCM_FORCE_GPU", raising=False)
        monkeypatch.delenv("VLLM_SR_ROCM_PRESERVE_CPU", raising=False)
        source = {
            "version": "v0.3",
            "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
            "routing": {
                "decisions": [{"name": "route", "algorithm": {"type": "multi_factor"}}]
            },
            "global": {
                "model_catalog": {"embeddings": {"semantic": {"use_cpu": True}}}
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
        runtime_dir = tmp_path / ".vllm-sr"
        runtime_dir.mkdir(mode=0o700)
        active_path = runtime_dir / "runtime-config.yaml"
        provenance_path = runtime_dir / "runtime-config.provenance.json"
        active_path.write_text("dashboard_owned: do-not-clobber\n", encoding="utf-8")
        provenance_path.write_text('{"dashboard":"receipt"}\n', encoding="utf-8")
        captured = {}

        class _FakeK8s:
            def deploy(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(rt, "_build_backend", lambda *_args, **_kwargs: _FakeK8s())
        monkeypatch.setattr(
            rt,
            "ensure_bootstrap_workspace",
            lambda _: MagicMock(config_path=config_path, setup_mode=False),
        )

        result = CliRunner().invoke(
            main,
            [
                "serve",
                "--config",
                str(config_path),
                "--target",
                "k8s",
                "--algorithm",
                "static",
            ],
        )

        assert result.exit_code == 0, result.output
        value = captured["config_document"]
        for component in ("routing", "decisions", 0, "algorithm", "type"):
            value = value[component]
        assert value == "static"
        assert active_path.read_text() == "dashboard_owned: do-not-clobber\n"
        assert provenance_path.read_text() == '{"dashboard":"receipt"}\n'
        assert {path.name for path in runtime_dir.iterdir()} == {
            active_path.name,
            provenance_path.name,
        }

    @pytest.mark.parametrize(
        ("override_args", "env_name", "env_value", "expected"),
        [
            (["--platform", "rocm"], None, None, "rocm"),
            ([], "VLLM_SR_PLATFORM", "cuda", "cuda"),
            ([], "DASHBOARD_PLATFORM", "rocm", "rocm"),
        ],
    )
    def test_kubernetes_gpu_platform_reaches_the_backend(
        self,
        monkeypatch,
        tmp_path,
        override_args,
        env_name,
        env_value,
        expected,
    ):
        monkeypatch.delenv("VLLM_SR_PLATFORM", raising=False)
        monkeypatch.delenv("DASHBOARD_PLATFORM", raising=False)
        if env_name is not None:
            monkeypatch.setenv(env_name, env_value)
        captured = {}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "version: v0.3\nlisteners:\n  - name: http\n    address: 0.0.0.0\n    port: 8899\n",
            encoding="utf-8",
        )

        class _FakeK8s:
            def deploy(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(rt, "_build_backend", lambda *_args, **_kwargs: _FakeK8s())

        result = CliRunner().invoke(
            main,
            [
                "serve",
                "--config",
                str(config_path),
                "--target",
                "kubernetes",
                *override_args,
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["platform"] == expected
        assert captured["gateway"] == "standalone"


def _values(mode=None, listeners=None, api_port=None):
    values = {"configOverride": {"listeners": listeners or []}}
    if mode is not None:
        values["gateway"] = {"mode": mode}
    if api_port is not None:
        values["service"] = {"api": {"port": api_port}}
    return values


_TLS = {"cert_file": "certs/tls.crt", "key_file": "certs/tls.key"}


@pytest.mark.parametrize(
    ("values", "port"),
    [
        (_values(listeners=[{"name": "http", "port": 8899}]), 8899),
        (_values("standalone", [{"name": "edge", "port": 9000}]), 9000),
        (
            _values(
                "standalone",
                [
                    {"name": "https", "port": 8443, "tls": _TLS},
                    {"name": "http", "port": 8899},
                ],
            ),
            8899,
        ),
        (_values("standalone", [{"name": "https", "port": 8443, "tls": _TLS}]), 8443),
        (_values("standalone"), 8899),
        (_values("extproc", [{"name": "http", "port": 8899}]), 8080),
        (_values("extproc", api_port=18080), 18080),
    ],
)
def test_client_service_port_follows_the_charts_client_port(values, port):
    assert client_service_port(values) == port


def test_kubernetes_summary_port_forwards_the_client_port(capsys):
    backend = K8sBackend.__new__(K8sBackend)
    backend.namespace = "test-ns"
    backend.release_name = "sr"

    backend._log_k8s_summary(8899)

    out = capsys.readouterr().out
    assert "kubectl port-forward -n test-ns svc/sr 8899:8899" in out
    assert "8080:8080" not in out


class TestK8sChartResolution:
    """Only deploy needs a chart; without a checkout it runs the published one."""

    @staticmethod
    def _pull_into_untardir(calls: list[list[str]], returncode: int = 0):
        def run(cmd, check=True, **_kwargs):
            calls.append(cmd)
            if returncode == 0:
                chart = Path(cmd[cmd.index("--untardir") + 1]) / "semantic-router"
                chart.mkdir(parents=True)
                (chart / "Chart.yaml").write_text("name: semantic-router\n")
            return subprocess.CompletedProcess(
                cmd, returncode, "", "Error: unauthorized"
            )

        return staticmethod(run)

    def test_a_backend_away_from_a_checkout_needs_no_chart(self, tmp_path, monkeypatch):
        # status, logs, stop and dashboard build the backend too (#4710).
        monkeypatch.chdir(tmp_path)

        backend = rt._build_backend("kubernetes", namespace="sr")

        assert backend.chart_dir is None

    def test_deploy_fetches_the_published_chart_once(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls: list[list[str]] = []
        monkeypatch.setattr(K8sBackend, "_run", self._pull_into_untardir(calls))
        backend = K8sBackend()

        chart = backend._resolve_chart_dir()

        assert backend._resolve_chart_dir() == chart
        assert (Path(chart) / "Chart.yaml").is_file()
        assert len(calls) == 1
        assert calls[0][:5] == [
            "helm",
            "pull",
            PUBLISHED_CHART,
            "--version",
            published_chart_version(),
        ]

    def test_a_failed_fetch_names_the_way_out(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            K8sBackend, "_run", self._pull_into_untardir([], returncode=1)
        )

        with pytest.raises(SystemExit, match=r"unauthorized\. Pass --chart-dir"):
            K8sBackend()._resolve_chart_dir()

    def test_a_chart_dir_or_a_checkout_is_used_as_is(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            K8sBackend, "_run", self._pull_into_untardir([], returncode=1)
        )
        assert (
            K8sBackend(chart_dir="/charts/mine")._resolve_chart_dir() == "/charts/mine"
        )

        checkout = tmp_path / "deploy" / "helm" / "semantic-router"
        checkout.mkdir(parents=True)
        (checkout / "Chart.yaml").write_text("name: semantic-router\n")
        monkeypatch.chdir(tmp_path)
        assert K8sBackend()._resolve_chart_dir() == str(checkout)

    def test_the_published_chart_follows_the_image_channel(self, monkeypatch):
        assert published_chart_version("0.5.0.dev20261007101233") == "0.0.0-latest"
        monkeypatch.setattr(
            "cli.k8s_backend.image_tag_for_cli_version", lambda version: f"v{version}"
        )
        assert published_chart_version("0.5.0") == "0.5.0"
