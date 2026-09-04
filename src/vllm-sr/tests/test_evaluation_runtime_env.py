from __future__ import annotations

import os
import stat
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from cli import container_cli, container_start, evaluation_deployment_snapshot
from cli.commands import runtime_paths
from cli.evaluation_runtime_env import (
    EVALUATION_DASHBOARD_CONFIG_ENV_NAMES,
    EVALUATION_DEPLOYMENTS_CONTAINER_DIR,
    EVALUATION_DEPLOYMENTS_DIR_ENV,
    EVALUATION_ENABLED_ENV,
    configure_dashboard_evaluation_deployments,
    configure_dashboard_evaluation_env,
    evaluation_dashboard_secret_env_names,
)

ORDINARY_DASHBOARD_GID = os.getgid() or 1000


@pytest.fixture
def immutable_staging_root():
    with tempfile.TemporaryDirectory(
        prefix="vllm-sr-evaluation-staging-", dir="/tmp"
    ) as directory:
        yield Path(directory)


def _router_config(tmp_path, token_env: str = "ROUTER_EVAL_TOKEN"):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "services": {
                        "management_api": {
                            "auth": {
                                "mode": "bearer",
                                "tokens": [{"env": token_env, "role": "viewer"}],
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _production_evaluation_environment() -> dict[str, str]:
    return {
        EVALUATION_ENABLED_ENV: "true",
        "EVALUATION_ROUTER_API_KEY_ENV": "ROUTER_EVAL_TOKEN",
        "EVALUATION_ENVOY_API_KEY_ENV": "ENVOY_EVAL_TOKEN",
        "EVALUATION_AGENT_TASK_LEDGER_URL": "https://agent-task.internal",
        "EVALUATION_AGENT_TASK_LEDGER_API_KEY_ENV": "AGENT_TASK_TOKEN",
        "EVALUATION_AGENT_TASK_LEDGER_TIMEOUT": "30s",
        "EVALUATION_FAULT_RECOVERY_LEDGER_URL": "https://fault.internal",
        "EVALUATION_FAULT_RECOVERY_LEDGER_API_KEY_ENV": "FAULT_TOKEN",
        "EVALUATION_FAULT_RECOVERY_LEDGER_TIMEOUT": "30s",
        "EVALUATION_HARD_POLICY_LEDGER_URL": "https://policy.internal",
        "EVALUATION_HARD_POLICY_LEDGER_API_KEY_ENV": "POLICY_TOKEN",
        "EVALUATION_HARD_POLICY_LEDGER_TIMEOUT": "45s",
        "EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_URL": "https://experiment.internal",
        "EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_API_KEY_ENV": "EXPERIMENT_TOKEN",
        "EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_TIMEOUT": "2m",
        "ROUTER_EVAL_TOKEN": "router-secret-value",
        "ENVOY_EVAL_TOKEN": "envoy-secret-value",
        "AGENT_TASK_TOKEN": "agent-task-secret-value",
        "FAULT_TOKEN": "fault-secret-value",
        "POLICY_TOKEN": "policy-secret-value",
        "EXPERIMENT_TOKEN": "experiment-secret-value",
    }


def _capture_run_commands(monkeypatch):
    captured = []

    def fake_run(cmd, capture_output, text, check, env=None):
        captured.append(cmd)
        return SimpleNamespace(stdout="container-id\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        container_start, "_render_split_envoy_config", lambda *args, **kwargs: None
    )
    return captured


def _find_container_run_cmd(commands, container_name):
    for command in commands:
        if "--name" not in command:
            continue
        if command[command.index("--name") + 1] == container_name:
            return command
    raise AssertionError(
        f"container command for {container_name} not found: {commands!r}"
    )


def _option_values(command, option):
    return [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == option
    ]


def _stub_valid_container_cli(monkeypatch, tmp_path):
    docker_bin = tmp_path / "docker"
    docker_bin.write_text("")
    monkeypatch.setattr(
        container_start,
        "resolve_container_cli_path",
        lambda preferred_path=None: str(docker_bin),
    )


def test_zero_evaluation_runtime_configuration_forwards_nothing(tmp_path):
    dashboard_env: dict[str, str] = {}

    names = configure_dashboard_evaluation_env(
        dashboard_env,
        source_config_path=str(_router_config(tmp_path)),
        host_env={},
    )

    assert names == set()
    assert dashboard_env == {}


def test_split_runtime_forwards_production_evaluation_config_without_secret_argv(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_SR_TOPOLOGY", "split")
    monkeypatch.setattr(runtime_paths, "_current_posix_user_id", lambda: None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: v0.3\n"
        "listeners:\n  - name: public\n    address: 0.0.0.0\n    port: 8899\n"
        "global:\n  services:\n    management_api:\n"
        "      bind_address: 0.0.0.0\n      port: 8080\n"
        "      auth:\n        mode: bearer\n        tokens:\n"
        "          - env: ROUTER_EVAL_TOKEN\n            role: viewer\n",
        encoding="utf-8",
    )
    evaluation_env = _production_evaluation_environment()
    evaluation_env["EVALUATION_AGENT_TASK_LEDGER_TIMEOUT"] = "20s"
    for name, value in evaluation_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(container_start, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        container_start,
        "get_runtime_images",
        lambda **_kwargs: {
            "router": "router-image",
            "envoy": "envoy-image",
            "dashboard": "dashboard-image",
        },
    )
    captured = _capture_run_commands(monkeypatch)
    _stub_valid_container_cli(monkeypatch, tmp_path)

    rc, _, _ = container_cli.container_start_vllm_sr(
        str(config_path),
        {"ROUTER_EVAL_TOKEN": "router-secret-value"},
        [{"name": "public", "address": "0.0.0.0", "port": 8899}],
        network_name="vllm-sr-network",
        minimal=False,
    )

    assert rc == 0
    router_cmd = _find_container_run_cmd(captured, "vllm-sr-router-container")
    envoy_cmd = _find_container_run_cmd(captured, "vllm-sr-envoy-container")
    dashboard_cmd = _find_container_run_cmd(captured, "vllm-sr-dashboard-container")
    dashboard_env = set(_option_values(dashboard_cmd, "-e"))
    router_env = set(_option_values(router_cmd, "-e"))
    envoy_env = set(_option_values(envoy_cmd, "-e"))
    assert "EVALUATION_ENABLED=true" in dashboard_env
    assert not any(value.startswith("EVALUATION_ENABLED=") for value in router_env)
    assert not any(value.startswith("EVALUATION_ENABLED=") for value in envoy_env)
    assert "EVALUATION_ROUTER_API_KEY_ENV=ROUTER_EVAL_TOKEN" in dashboard_env
    assert "EVALUATION_AGENT_TASK_LEDGER_TIMEOUT=20s" in dashboard_env
    assert "EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_TIMEOUT=2m" in dashboard_env
    assert "ROUTER_EVAL_TOKEN" in dashboard_env
    assert "ROUTER_EVAL_TOKEN" in router_env
    for name in (
        "ENVOY_EVAL_TOKEN",
        "AGENT_TASK_TOKEN",
        "FAULT_TOKEN",
        "POLICY_TOKEN",
        "EXPERIMENT_TOKEN",
    ):
        assert name in dashboard_env
        assert name not in router_env
        assert name not in envoy_env
    rendered_commands = repr(captured)
    for secret in (
        "router-secret-value",
        "envoy-secret-value",
        "agent-task-secret-value",
        "fault-secret-value",
        "policy-secret-value",
        "experiment-secret-value",
    ):
        assert secret not in rendered_commands


def test_evaluation_runtime_configuration_is_dashboard_scoped_and_secret_safe(tmp_path):
    host_env = _production_evaluation_environment()
    dashboard_env: dict[str, str] = {}

    names = configure_dashboard_evaluation_env(
        dashboard_env,
        source_config_path=str(_router_config(tmp_path)),
        host_env=host_env,
    )

    expected_names = {
        "ROUTER_EVAL_TOKEN",
        "ENVOY_EVAL_TOKEN",
        "AGENT_TASK_TOKEN",
        "FAULT_TOKEN",
        "POLICY_TOKEN",
        "EXPERIMENT_TOKEN",
    }
    assert names == expected_names
    assert evaluation_dashboard_secret_env_names(dashboard_env) == expected_names
    assert set(EVALUATION_DASHBOARD_CONFIG_ENV_NAMES) <= set(dashboard_env)
    for name in expected_names:
        assert dashboard_env[name] == ""
    rendered = repr(dashboard_env)
    for secret in (
        "router-secret-value",
        "envoy-secret-value",
        "agent-task-secret-value",
        "fault-secret-value",
        "policy-secret-value",
        "experiment-secret-value",
    ):
        assert secret not in rendered


@pytest.mark.parametrize("value", ["TRUE", "False", "1", " true", "false ", ""])
def test_evaluation_enabled_requires_an_exact_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="must be exactly 'true' or 'false'"):
        configure_dashboard_evaluation_env(
            {},
            source_config_path=str(_router_config(tmp_path)),
            host_env={EVALUATION_ENABLED_ENV: value},
        )


def test_disabled_evaluation_ignores_stale_runtime_configuration(tmp_path):
    dashboard_env = {
        "EVALUATION_AGENT_TASK_LEDGER_URL": "https://stale.internal/path",
        "EVALUATION_AGENT_TASK_LEDGER_API_KEY_ENV": "MISSING_TOKEN",
    }

    names = configure_dashboard_evaluation_env(
        dashboard_env,
        source_config_path=str(_router_config(tmp_path)),
        host_env={EVALUATION_ENABLED_ENV: "false"},
    )

    assert names == set()
    assert dashboard_env == {EVALUATION_ENABLED_ENV: "false"}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda env: env.pop("FAULT_TOKEN"),
            "no non-empty host value: FAULT_TOKEN",
        ),
        (
            lambda env: env.__setitem__(
                "EVALUATION_HARD_POLICY_LEDGER_API_KEY_ENV", "FAULT_TOKEN"
            ),
            "credential references must be distinct",
        ),
        (
            lambda env: env.__setitem__(
                "EVALUATION_ROUTER_API_KEY_ENV", "VLLM_SR_DASHBOARD_RECIPE_TOKEN"
            ),
            "cannot reuse the Dashboard management credential",
        ),
        (
            lambda env: env.__setitem__(
                "EVALUATION_FAULT_RECOVERY_LEDGER_API_KEY_ENV", "literal-secret"
            ),
            "must name one uppercase",
        ),
    ],
)
def test_evaluation_runtime_credentials_fail_closed(tmp_path, mutate, match):
    host_env = _production_evaluation_environment()
    mutate(host_env)

    with pytest.raises(ValueError, match=match):
        configure_dashboard_evaluation_env(
            {},
            source_config_path=str(_router_config(tmp_path)),
            host_env=host_env,
        )


def test_router_evaluation_credential_must_exist_in_router_auth_config(tmp_path):
    with pytest.raises(ValueError, match="must be declared"):
        configure_dashboard_evaluation_env(
            {},
            source_config_path=str(_router_config(tmp_path, "OTHER_ROUTER_TOKEN")),
            host_env=_production_evaluation_environment(),
        )


def test_evaluation_deployments_are_readable_by_ordinary_dashboard_gid(
    tmp_path, immutable_staging_root
):
    registry_root = tmp_path / "deployments"
    registry_root.mkdir()
    baseline_root = registry_root / "baseline"
    baseline_root.mkdir()
    (baseline_root / "config.yaml").write_text("version: v0.3\n", encoding="utf-8")
    (registry_root / "registry.json").write_text(
        '{"schema_version":"evaluation-deployments.v1","deployments":['
        '{"id":"baseline","name":"Baseline","config_file":"baseline/config.yaml",'
        '"router_origin":"https://router.internal",'
        '"envoy_origin":"https://envoy.internal"}]}',
        encoding="utf-8",
    )
    dashboard_env: dict[str, str] = {}
    mounts: list[str] = []
    staging_root = immutable_staging_root

    configure_dashboard_evaluation_deployments(
        dashboard_env,
        mounts,
        staging_root=str(staging_root),
        readable_gid=ORDINARY_DASHBOARD_GID,
        host_env={EVALUATION_DEPLOYMENTS_DIR_ENV: str(registry_root)},
    )

    assert dashboard_env[EVALUATION_DEPLOYMENTS_DIR_ENV] == (
        EVALUATION_DEPLOYMENTS_CONTAINER_DIR
    )
    mounted_root = mounts[0].split(":", 1)[0]
    assert mounted_root != str(registry_root)
    assert mounted_root.startswith(str(staging_root))
    assert mounts == [f"{mounted_root}:{EVALUATION_DEPLOYMENTS_CONTAINER_DIR}:ro,z"]
    assert (Path(mounted_root) / "registry.json").read_bytes() == (
        registry_root / "registry.json"
    ).read_bytes()
    assert stat.S_IMODE(Path(mounted_root).stat().st_mode) == 0o550
    assert stat.S_IMODE((Path(mounted_root) / "registry.json").stat().st_mode) == 0o440
    assert stat.S_IMODE((Path(mounted_root) / "baseline").stat().st_mode) == 0o550
    assert (
        stat.S_IMODE((Path(mounted_root) / "baseline" / "config.yaml").stat().st_mode)
        == 0o440
    )
    for path in (
        Path(mounted_root),
        Path(mounted_root) / "registry.json",
        Path(mounted_root) / "baseline",
        Path(mounted_root) / "baseline" / "config.yaml",
    ):
        assert path.stat().st_gid == ORDINARY_DASHBOARD_GID

    original_registry = (Path(mounted_root) / "registry.json").read_bytes()
    original_config = (Path(mounted_root) / "baseline" / "config.yaml").read_bytes()
    (registry_root / "registry.json").write_text(
        '{"schema_version":"evaluation-deployments.v1","deployments":[]}',
        encoding="utf-8",
    )
    (baseline_root / "config.yaml").write_text(
        "attacker: substituted\n", encoding="utf-8"
    )

    assert (Path(mounted_root) / "registry.json").read_bytes() == original_registry
    assert (
        Path(mounted_root) / "baseline" / "config.yaml"
    ).read_bytes() == original_config


def test_root_runtime_snapshot_assigns_dashboard_gid_before_freezing(
    tmp_path, monkeypatch
):
    snapshot = tmp_path / "snapshot"
    nested = snapshot / "nested"
    nested.mkdir(parents=True)
    (snapshot / "registry.json").write_text("{}", encoding="utf-8")
    (nested / "config.yaml").write_text("version: v0.3\n", encoding="utf-8")

    assigned: set[int] = set()
    original_fchmod = os.fchmod

    def record_assignment(descriptor, owner, gid):
        assert owner == -1
        assert gid == 65532
        assigned.add(os.fstat(descriptor).st_ino)

    def require_assignment_before_freeze(descriptor, mode):
        assert os.fstat(descriptor).st_ino in assigned
        original_fchmod(descriptor, mode)

    monkeypatch.setattr(os, "fchown", record_assignment)
    monkeypatch.setattr(os, "fchmod", require_assignment_before_freeze)

    evaluation_deployment_snapshot._freeze_snapshot_permissions(snapshot, 65532)

    entries = (snapshot, snapshot / "registry.json", nested, nested / "config.yaml")
    assert assigned == {path.stat().st_ino for path in entries}
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o550
    assert stat.S_IMODE(nested.stat().st_mode) == 0o550
    assert stat.S_IMODE((snapshot / "registry.json").stat().st_mode) == 0o440
    assert stat.S_IMODE((nested / "config.yaml").stat().st_mode) == 0o440


def test_snapshot_identity_is_scoped_to_dashboard_runtime_gid():
    files = {"registry.json": b"{}"}

    assert evaluation_deployment_snapshot._snapshot_digest(
        files, 1000
    ) != evaluation_deployment_snapshot._snapshot_digest(files, 65532)


def test_evaluation_deployment_mount_is_zero_config_by_default():
    dashboard_env = {
        EVALUATION_DEPLOYMENTS_DIR_ENV: EVALUATION_DEPLOYMENTS_CONTAINER_DIR
    }
    mounts: list[str] = []

    configure_dashboard_evaluation_deployments(
        dashboard_env,
        mounts,
        staging_root="/unused-with-zero-config",
        readable_gid=ORDINARY_DASHBOARD_GID,
        host_env={},
    )

    assert EVALUATION_DEPLOYMENTS_DIR_ENV not in dashboard_env
    assert mounts == []


def test_evaluation_deployment_mount_fails_closed_for_root_symlink(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "registry.json").write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="components must not be symlinks"):
        configure_dashboard_evaluation_deployments(
            {},
            [],
            staging_root=str(tmp_path / "state"),
            readable_gid=ORDINARY_DASHBOARD_GID,
            host_env={EVALUATION_DEPLOYMENTS_DIR_ENV: str(linked)},
        )


def test_evaluation_deployment_mount_fails_closed_for_parent_symlink(tmp_path):
    actual_parent = tmp_path / "actual-parent"
    actual_root = actual_parent / "deployments"
    actual_root.mkdir(parents=True)
    (actual_root / "registry.json").write_text("{}", encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="components must not be symlinks"):
        configure_dashboard_evaluation_deployments(
            {},
            [],
            staging_root=str(tmp_path / "state"),
            readable_gid=ORDINARY_DASHBOARD_GID,
            host_env={
                EVALUATION_DEPLOYMENTS_DIR_ENV: str(linked_parent / "deployments")
            },
        )


def test_evaluation_deployment_mount_fails_closed_for_registry_symlink(tmp_path):
    registry_root = tmp_path / "deployments"
    registry_root.mkdir()
    actual_registry = tmp_path / "registry.json"
    actual_registry.write_text("{}", encoding="utf-8")
    (registry_root / "registry.json").symlink_to(actual_registry)

    with pytest.raises(ValueError, match="contain no symlinks"):
        configure_dashboard_evaluation_deployments(
            {},
            [],
            staging_root=str(tmp_path / "state"),
            readable_gid=ORDINARY_DASHBOARD_GID,
            host_env={EVALUATION_DEPLOYMENTS_DIR_ENV: str(registry_root)},
        )


def test_evaluation_deployment_mount_fails_closed_for_config_symlink(tmp_path):
    registry_root = tmp_path / "deployments"
    registry_root.mkdir()
    actual_config = tmp_path / "config.yaml"
    actual_config.write_text("secret: outside\n", encoding="utf-8")
    (registry_root / "config.yaml").symlink_to(actual_config)
    (registry_root / "registry.json").write_text(
        '{"schema_version":"evaluation-deployments.v1","deployments":['
        '{"id":"baseline","name":"Baseline","config_file":"config.yaml",'
        '"router_origin":"https://router.internal",'
        '"envoy_origin":"https://envoy.internal"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contain no symlinks"):
        configure_dashboard_evaluation_deployments(
            {},
            [],
            staging_root=str(tmp_path / "state"),
            readable_gid=ORDINARY_DASHBOARD_GID,
            host_env={EVALUATION_DEPLOYMENTS_DIR_ENV: str(registry_root)},
        )


def test_evaluation_deployment_mount_rejects_unrecognized_registry_data(tmp_path):
    registry_root = tmp_path / "deployments"
    registry_root.mkdir()
    (registry_root / "config.yaml").write_text("version: v0.3\n", encoding="utf-8")
    (registry_root / "registry.json").write_text(
        '{"schema_version":"evaluation-deployments.v1","deployments":['
        '{"id":"baseline","name":"Baseline","config_file":"config.yaml",'
        '"router_origin":"https://router.internal",'
        '"envoy_origin":"https://envoy.internal","api_key":"must-not-stage"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown fields"):
        configure_dashboard_evaluation_deployments(
            {},
            [],
            staging_root=str(tmp_path / "state"),
            readable_gid=ORDINARY_DASHBOARD_GID,
            host_env={EVALUATION_DEPLOYMENTS_DIR_ENV: str(registry_root)},
        )
