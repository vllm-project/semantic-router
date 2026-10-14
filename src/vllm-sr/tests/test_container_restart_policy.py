import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import container_services, container_support_services  # noqa: E402
from cli.bootstrap import BootstrapResult  # noqa: E402
from cli.commands import runtime as runtime_commands  # noqa: E402
from cli.container_run_command import (  # noqa: E402
    RESTART_POLICIES,
    RESTART_POLICY_ENV,
    build_base_run_command,
    restart_policy_args,
)
from cli.main import main  # noqa: E402
from cli.runtime_stack import resolve_runtime_stack  # noqa: E402
from click.testing import CliRunner  # noqa: E402


def _restart_value(command):
    index = command.index("--restart")
    return command[index + 1]


def test_no_policy_keeps_containers_without_a_restart_policy(monkeypatch):
    monkeypatch.delenv(RESTART_POLICY_ENV, raising=False)
    assert restart_policy_args() == []
    monkeypatch.setenv(RESTART_POLICY_ENV, "no")
    assert restart_policy_args() == []


@pytest.mark.parametrize("policy", [p for p in RESTART_POLICIES if p != "no"])
def test_policy_becomes_a_restart_argument(monkeypatch, policy):
    monkeypatch.setenv(RESTART_POLICY_ENV, policy.upper())
    assert restart_policy_args() == ["--restart", policy]


def test_unknown_policy_is_refused(monkeypatch):
    monkeypatch.setenv(RESTART_POLICY_ENV, "sometimes")
    with pytest.raises(ValueError, match=RESTART_POLICY_ENV):
        restart_policy_args()


@pytest.mark.parametrize("start_immediately", [True, False])
def test_router_envoy_and_dashboard_commands_carry_the_policy(
    monkeypatch, start_immediately
):
    monkeypatch.setenv(RESTART_POLICY_ENV, "unless-stopped")
    cmd = build_base_run_command(
        "docker",
        65536,
        "net",
        "vllm-sr-router-container",
        start_immediately=start_immediately,
    )
    assert _restart_value(cmd) == "unless-stopped"
    assert cmd[: 2 if start_immediately else 1] == (
        ["docker", "run"] if start_immediately else ["docker"]
    )


def _storage_start_environment(monkeypatch, commands):
    monkeypatch.setattr(container_services, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_services, "container_status", lambda _name: "not found"
    )
    monkeypatch.setattr(container_services, "_is_port_in_use", lambda _port: False)
    monkeypatch.setattr(
        container_services,
        "_run_service_start",
        lambda cmd, _label: commands.append(cmd) or (0, "", ""),
    )


def test_storage_services_carry_the_policy(monkeypatch, tmp_path):
    monkeypatch.setenv(RESTART_POLICY_ENV, "always")
    commands = []
    _storage_start_environment(monkeypatch, commands)
    layout = resolve_runtime_stack()
    conf = tmp_path / "redis.conf"
    conf.write_text("requirepass x\n")
    password = tmp_path / "postgres-password"
    password.write_text("x")

    container_services.container_start_redis(
        "net", layout, redis_conf_file=str(conf), data_volume="redis-data"
    )
    container_services.container_start_postgres(
        "net", layout, postgres_password_file=str(password), data_volume="pg-data"
    )
    container_services.container_start_milvus(
        "net",
        layout,
        state_root_dir=str(tmp_path),
        host_hidden_state_dir=str(tmp_path / ".vllm-sr"),
    )

    assert len(commands) == 3
    assert [_restart_value(cmd) for cmd in commands] == ["always"] * 3


def test_observability_services_carry_the_policy(monkeypatch, tmp_path):
    monkeypatch.setenv(RESTART_POLICY_ENV, "on-failure")
    commands = []
    monkeypatch.setattr(
        container_support_services, "get_container_runtime", lambda: "docker"
    )
    monkeypatch.setattr(
        container_support_services, "_replace_existing_container", lambda _name: None
    )
    monkeypatch.setattr(
        container_support_services,
        "_run_service_start",
        lambda cmd, _service: commands.append(cmd) or (0, "", ""),
    )
    layout = resolve_runtime_stack()

    container_support_services.container_start_jaeger(stack_layout=layout)
    container_support_services.container_start_prometheus(
        config_dir=str(tmp_path), stack_layout=layout
    )
    container_support_services.container_start_grafana(
        config_dir=str(tmp_path), stack_layout=layout
    )

    assert [_restart_value(cmd) for cmd in commands] == ["on-failure"] * 3


def _capture_serve(monkeypatch, tmp_path):
    monkeypatch.setattr("cli.runtime_lifecycle.get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        "cli.runtime_lifecycle.container_status_strict", lambda _name: "not found"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "http-8899", "address": "0.0.0.0", "port": 8899}
                ],
                "routing": {"decisions": [{"name": "default", "priority": 1}]},
            },
            sort_keys=False,
        )
    )
    bootstrap = BootstrapResult(
        config_path=config_path, output_dir=tmp_path / ".vllm-sr", setup_mode=False
    )
    seen = {}

    class _StubBackend:
        def deploy(self, **_kwargs):
            # What every container builder of this start reads.
            seen["args"] = restart_policy_args()

    monkeypatch.setattr(
        runtime_commands, "ensure_bootstrap_workspace", lambda _: bootstrap
    )
    monkeypatch.setattr(
        runtime_commands, "_build_backend", lambda *a, **k: _StubBackend()
    )
    return config_path, seen


def test_serve_restart_policy_reaches_the_container_builders(monkeypatch, tmp_path):
    monkeypatch.delenv(RESTART_POLICY_ENV, raising=False)
    config_path, seen = _capture_serve(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        main,
        ["serve", "--config", str(config_path), "--restart-policy", "unless-stopped"],
    )

    assert result.exit_code == 0, result.output
    assert seen["args"] == ["--restart", "unless-stopped"]


def test_serve_without_a_policy_adds_none(monkeypatch, tmp_path):
    monkeypatch.delenv(RESTART_POLICY_ENV, raising=False)
    config_path, seen = _capture_serve(monkeypatch, tmp_path)

    result = CliRunner().invoke(main, ["serve", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert seen["args"] == []


def test_serve_restart_policy_is_a_docker_target_option(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("a misplaced option reached workspace or backend mutation")

    monkeypatch.setattr(runtime_commands, "_resolve_serve_config", unexpected)
    monkeypatch.setattr(runtime_commands, "_build_backend", unexpected)
    result = CliRunner().invoke(
        main, ["serve", "--target", "kubernetes", "--restart-policy", "always"]
    )

    assert result.exit_code == 2
    assert "--restart-policy applies to the docker target" in result.output


def test_serve_refuses_an_unknown_restart_policy():
    result = CliRunner().invoke(main, ["serve", "--restart-policy", "sometimes"])
    assert result.exit_code == 2
    assert "--restart-policy" in result.output
