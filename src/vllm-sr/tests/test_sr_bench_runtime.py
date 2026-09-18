"""The benchmark worker survives UI/runtime reloads without resending work."""
from __future__ import annotations

import json
from contextlib import nullcontext
import stat
import subprocess
from types import SimpleNamespace

import pytest
from cli import container_start, container_start_runner, core, runtime_lifecycle
from cli.runtime_stack import resolve_runtime_stack
from cli.sr_bench_runtime import (
    BENCH_IDENTITY_LABEL, dashboard_bench_env, prepare_bench_runtime,
    reuse_bench_container,
)


def test_managed_service_token_is_private_stable_and_not_in_common_mounts(tmp_path):
    stack = resolve_runtime_stack(stack_name="bench-test", port_offset=1000)
    first = prepare_bench_runtime(str(tmp_path), stack, {})
    second = prepare_bench_runtime(str(tmp_path), stack, {})
    assert first == second
    assert first.store == tmp_path / ".sr-bench/bench-test/store"
    assert len(first.secrets[first.token_env]) >= 32
    assert stat.S_IMODE((first.store.parent / "service-token").stat().st_mode) == 0o600
    assert str(first.store).find(".vllm-sr") == -1
    assert stack.sr_bench_port == 9090
    assert stack.sr_bench_container_name not in stack.runtime_container_names
    assert dashboard_bench_env(first)[first.token_env] == ""


def test_external_service_does_not_create_or_restart_a_container(tmp_path):
    stack = resolve_runtime_stack()
    configured = prepare_bench_runtime(str(tmp_path), stack, {"SR_BENCH_URL": "https://bench.example", "SR_BENCH_TOKEN_ENV": "BENCH_ACCESS", "BENCH_ACCESS": "private"})
    assert not configured.managed
    assert configured.origin == "https://bench.example"
    assert not (tmp_path / ".sr-bench").exists()
    with pytest.raises(ValueError, match="missing"):
        prepare_bench_runtime(str(tmp_path), stack, {"SR_BENCH_URL": "https://bench.example"})


def test_target_credentials_are_loaded_only_from_registered_references(tmp_path):
    stack = resolve_runtime_stack()
    initial = prepare_bench_runtime(str(tmp_path), stack, {})
    (initial.store / "targets.json").write_text(json.dumps([{"id": "single", "api_key_env": "MODEL_TOKEN"}]))
    with pytest.raises(ValueError, match="Missing"):
        prepare_bench_runtime(str(tmp_path), stack, {})
    configured = prepare_bench_runtime(str(tmp_path), stack, {"MODEL_TOKEN": "model-secret"})
    assert configured.secrets["MODEL_TOKEN"] == "model-secret"
    assert "MODEL_TOKEN" not in dashboard_bench_env(configured)


def test_bench_container_has_no_runtime_socket_or_gpu_and_token_never_enters_argv(tmp_path):
    stack = resolve_runtime_stack()
    bench = prepare_bench_runtime(str(tmp_path), stack, {})
    name, container, commands = container_start._build_bench_runtime_spec(runtime="docker", image="dashboard@sha256:test", nofile_limit=10000, network_name=stack.network_name, stack_layout=stack, bench=bench)
    command, health = commands
    assert name == "sr-bench" and container == stack.sr_bench_container_name
    assert "127.0.0.1:8090:8090" in command
    assert "--user" in command
    assert "--device" not in command and "--gpus" not in command
    assert not any("docker.sock" in item for item in command)
    assert "SR_BENCH_TOKEN" in command
    assert bench.secrets[bench.token_env] not in " ".join(command + health)
    assert ":ro" not in next(item for item in command if item.startswith(str(bench.store) + ":"))


def test_reuse_requires_running_matching_worker_and_never_starts_a_stopped_one(monkeypatch):
    command = ["docker", "run", "--label", f"{BENCH_IDENTITY_LABEL}=same"]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout='"running" '+json.dumps({BENCH_IDENTITY_LABEL:"same"})))
    assert reuse_bench_container(command, "bench")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout='"exited" '+json.dumps({BENCH_IDENTITY_LABEL:"same"})))
    with pytest.raises(ValueError, match="stopped"):
        reuse_bench_container(command, "bench")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout='"running" '+json.dumps({BENCH_IDENTITY_LABEL:"other"})))
    with pytest.raises(ValueError, match="different"):
        reuse_bench_container(command, "bench")


def test_reused_worker_is_not_owned_by_failed_dashboard_start_rollback(monkeypatch):
    commands=[]; removed=[]
    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            raise subprocess.CalledProcessError(125, command, stderr="name conflict")
        return SimpleNamespace(stdout="", stderr="")
    monkeypatch.setattr(subprocess,"run",run)
    monkeypatch.setattr(container_start_runner,"reuse_bench_container",lambda *a: True)
    monkeypatch.setattr(container_start_runner,"_cleanup_started_containers", lambda names: removed.extend(names))
    status, _, _ = container_start_runner.run_container_specs([
        ("sr-bench","bench",(["docker","run"],["docker","exec","bench","health"])),
        ("dashboard","dashboard",(["docker","run"],)),
    ], storage_secret_values={}, bench_secret_values={"SR_BENCH_TOKEN":"secret"})
    assert status == 125
    assert removed == []
    assert commands.count(["docker","run"]) == 2


def test_runtime_reload_preserves_bench_but_explicit_stop_accounts_for_it(monkeypatch):
    stack = resolve_runtime_stack()
    observed=[]
    monkeypatch.setattr(core,"container_status",lambda name: observed.append(name) or "not found")
    core._managed_container_statuses(stack)
    assert stack.sr_bench_container_name in observed
    assert stack.sr_bench_container_name not in stack.runtime_container_names
    stopped=[]
    monkeypatch.setattr(runtime_lifecycle,"acquire_runtime_lifecycle_lock",lambda **kwargs: nullcontext())
    monkeypatch.setattr(runtime_lifecycle,"get_container_runtime",lambda: "docker")
    monkeypatch.setattr(runtime_lifecycle,"container_status_strict",lambda name: "exited" if name in stopped else "running")
    monkeypatch.setattr(runtime_lifecycle,"container_stop_container",lambda name: stopped.append(name) or True)
    runtime_lifecycle.stop_runtime_before_config_replacement(stack)
    assert set(stopped) == set(stack.runtime_container_names)
    assert stack.sr_bench_container_name not in stopped
