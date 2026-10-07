"""Pending activations: the CLI applies what the Dashboard saved, for setup or a restart."""

import json
import os
import signal
import time
from pathlib import Path

import pytest
from cli import core, pending_activation, runtime_lifecycle
from cli.runtime_config_lock import (
    RuntimeConfigLockError,
    acquire_runtime_config_lock,
)
from cli.runtime_stack import resolve_runtime_stack


def _record(runtime_config: Path, reason: str = "setup", detail: str = "") -> None:
    pending_activation.record_file(runtime_config).write_text(
        json.dumps(
            {"reason": reason, "recorded_at": "2026-10-07T00:00:00Z", "detail": detail}
        )
    )


def test_hand_off_files_sit_beside_the_runtime_config():
    config = Path("/app/.vllm-sr/runtime-config.lane.yaml")
    assert pending_activation.heartbeat_file(config) == Path(
        "/app/.vllm-sr/runtime-config.lane.serve-heartbeat.json"
    )
    assert pending_activation.record_file(config) == Path(
        "/app/.vllm-sr/runtime-config.lane.pending-activation.json"
    )


def test_the_heartbeat_covers_the_block(tmp_path):
    config = tmp_path / "runtime-config.yaml"
    heartbeat = pending_activation.heartbeat_file(config)

    with pending_activation.serve_heartbeat(config, interval=0.01):
        assert json.loads(heartbeat.read_text()) == {
            "pid": os.getpid(),
            "state": "starting",
        }
        first = heartbeat.stat().st_mtime_ns
        deadline = time.monotonic() + 5
        while heartbeat.stat().st_mtime_ns == first and time.monotonic() < deadline:
            time.sleep(0.01)
        assert heartbeat.stat().st_mtime_ns > first

    assert not heartbeat.exists()


def test_the_heartbeat_says_what_the_cli_does(tmp_path):
    config = tmp_path / "runtime-config.yaml"
    heartbeat = pending_activation.heartbeat_file(config)

    with pending_activation.serve_heartbeat(config, interval=0.01) as state:
        state(pending_activation.SERVE_WAITING)
        assert json.loads(heartbeat.read_text())["state"] == "waiting"
        time.sleep(0.05)
        # The beat keeps the state it was given.
        assert json.loads(heartbeat.read_text())["state"] == "waiting"
        state(pending_activation.SERVE_STARTING)
        assert json.loads(heartbeat.read_text())["state"] == "starting"


def test_the_wait_ends_when_the_dashboard_records_the_activation(tmp_path):
    config = tmp_path / "runtime-config.yaml"
    polls = []
    activated_on = 3

    def dashboard_running():
        polls.append(True)
        if len(polls) == activated_on:
            _record(config)
        return True

    assert pending_activation.wait_for_pending_activation(
        config, dashboard_running=dashboard_running, interval=0
    ) == pending_activation.PendingActivation(reason="setup")
    assert len(polls) == activated_on


@pytest.mark.parametrize("stop", ["sigterm", "ctrl-c"])
def test_stopping_the_wait_leaves_setup_mode(tmp_path, stop):
    config = tmp_path / "runtime-config.yaml"

    def dashboard_running():
        if stop == "sigterm":
            os.kill(os.getpid(), signal.SIGTERM)
        raise KeyboardInterrupt

    assert (
        pending_activation.wait_for_pending_activation(
            config, dashboard_running=dashboard_running, interval=0
        )
        is None
    )
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_a_dashboard_that_exits_ends_the_wait(tmp_path):
    with pytest.raises(RuntimeError, match="The Dashboard exited during setup"):
        pending_activation.wait_for_pending_activation(
            tmp_path / "runtime-config.yaml",
            dashboard_running=lambda: False,
            interval=0,
        )


def test_the_lock_is_free_while_released_and_held_again_after(tmp_path):
    runtime_config = tmp_path / "runtime-config.yaml"
    kwargs = {
        "runtime_config_path": runtime_config,
        "state_root_dir": tmp_path,
        "stack_name": "vllm-sr",
    }
    with acquire_runtime_config_lock(**kwargs) as lock:
        with pytest.raises(RuntimeConfigLockError):
            acquire_runtime_config_lock(**kwargs, timeout_seconds=0)
        with lock.released():
            acquire_runtime_config_lock(**kwargs, timeout_seconds=0).close()
        with pytest.raises(RuntimeConfigLockError):
            acquire_runtime_config_lock(**kwargs, timeout_seconds=0)


@pytest.fixture
def setup_stack(monkeypatch, tmp_path):
    """`core.start_vllm_sr` in setup mode, with every container call recorded."""

    runtime_config = tmp_path / "runtime-config.yaml"
    runtime_config.write_text("listeners: []\n")
    calls = []
    configs = iter(
        [
            {
                "listeners": [{"name": "http-8899", "port": 8899}],
                "setup": {"mode": True},
            },
            {"listeners": [{"name": "http-9000", "port": 9000}]},
        ]
    )

    def record(name, ret=(0, "", "")):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return ret

        return call

    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(core, "load_config", lambda _path: next(configs))
    monkeypatch.setattr(core, "container_status_strict", lambda _name: "not found")
    monkeypatch.setattr(core, "ensure_clean_runtime_container", record("ensure_clean"))
    monkeypatch.setattr(core, "provision_storage_backends", lambda *a, **k: set())
    monkeypatch.setattr(
        core, "_prepare_runtime_network", lambda *a, **k: ("net", str(tmp_path))
    )
    monkeypatch.setattr(core, "container_start_vllm_sr", record("start"))
    monkeypatch.setattr(core, "connect_runtime_container", lambda *a: None)
    monkeypatch.setattr(core, "maybe_finish_setup_mode", lambda *a, **k: True)
    monkeypatch.setattr(core, "container_status", lambda _name: "running")
    monkeypatch.setattr(core, "_wait_and_verify_runtime", record("wait_ready"))
    monkeypatch.setattr(core, "log_runtime_summary", record("summary"))
    monkeypatch.setattr(runtime_lifecycle, "container_status", lambda _name: "running")
    return runtime_config, calls


def _serve_setup(runtime_config, gateway):
    core.start_vllm_sr(
        str(runtime_config),
        env_vars={"VLLM_SR_SETUP_MODE": "true", "DASHBOARD_SETUP_MODE": "true"},
        enable_observability=False,
        source_config_file=str(runtime_config.with_name("config.yaml")),
        runtime_config_file=str(runtime_config),
        gateway=gateway,
    )


@pytest.mark.parametrize(
    ("gateway", "services"),
    [("standalone", ("router",)), ("extproc", ("router", "envoy"))],
)
def test_serve_starts_the_router_once_setup_is_activated(
    setup_stack, monkeypatch, gateway, services
):
    runtime_config, calls = setup_stack
    stack = resolve_runtime_stack()
    waited_unlocked = []

    started = []

    def start(*args, **kwargs):
        # The Dashboard sees a CLI that starts the stack.
        heartbeat = pending_activation.heartbeat_file(runtime_config)
        started.append(json.loads(heartbeat.read_text())["state"])
        calls.append(("start", args, kwargs))
        return (0, "", "")

    monkeypatch.setattr(core, "container_start_vllm_sr", start)

    def wait(config, *, dashboard_running):
        assert dashboard_running()
        # The Dashboard sees a CLI that waits to apply the activation.
        heartbeat = pending_activation.heartbeat_file(config)
        assert json.loads(heartbeat.read_text())["state"] == "waiting"
        # The Dashboard's activation needs the runtime config lock.
        acquire_runtime_config_lock(
            runtime_config_path=config,
            state_root_dir=runtime_config.parent,
            stack_name=stack.stack_name,
            timeout_seconds=0,
        ).close()
        waited_unlocked.append(config)
        _record(runtime_config)
        return pending_activation.PendingActivation(reason="setup")

    monkeypatch.setattr(core, "wait_for_pending_activation", wait)

    _serve_setup(runtime_config, gateway)

    assert waited_unlocked == [str(runtime_config)]
    names = [call[0] for call in calls]
    first = names.index("start")
    # Setup's own start, then the Router (and Envoy) recreated from the activated config.
    assert names[first:] == [
        "start",
        *["ensure_clean"] * len(services),
        "start",
        "wait_ready",
        "summary",
    ]
    _, _, activated = calls[names.index("start", first + 1)]
    assert activated["services"] == services
    assert activated["listeners"] == [{"name": "http-9000", "port": 9000}]
    assert "VLLM_SR_SETUP_MODE" not in activated["env_vars"]
    assert "DASHBOARD_SETUP_MODE" not in activated["env_vars"]
    assert calls[-1][1][0] == [{"name": "http-9000", "port": 9000}]
    assert pending_activation.read_pending_activation(runtime_config) is None
    assert started == ["starting", "starting"]
    assert not pending_activation.heartbeat_file(runtime_config).exists()


def test_serve_that_stops_waiting_leaves_the_router_stopped(setup_stack, monkeypatch):
    runtime_config, calls = setup_stack
    monkeypatch.setattr(core, "wait_for_pending_activation", lambda *a, **k: None)

    _serve_setup(runtime_config, "standalone")

    names = [call[0] for call in calls]
    assert names[names.index("start") :] == ["start"]


def test_status_reports_a_setup_that_waits_for_serve(monkeypatch, tmp_path):
    stack = resolve_runtime_stack()
    runtime_config = tmp_path / "runtime-config.yaml"
    statuses = {stack.router_container_name: "created"}
    monkeypatch.setattr(
        core, "container_status", lambda name: statuses.get(name, "running")
    )
    monkeypatch.setattr(
        core,
        "inspect_container_mounts",
        lambda _name: [{"Source": str(tmp_path), "Destination": "/app/.vllm-sr"}],
    )

    assert core._activation_state(stack).startswith("Setup mode: activate a config")
    _record(runtime_config)
    assert core._activation_state(stack) == (
        "Setup is complete; run `vllm-sr serve` to start the Router."
    )
    statuses[stack.router_container_name] = "running"
    assert core._activation_state(stack) is None


def test_status_reports_a_change_that_waits_for_a_restart(monkeypatch, tmp_path):
    stack = resolve_runtime_stack()
    runtime_config = tmp_path / "runtime-config.yaml"
    monkeypatch.setattr(core, "container_status", lambda _name: "running")
    monkeypatch.setattr(
        core,
        "inspect_container_mounts",
        lambda _name: [{"Source": str(tmp_path), "Destination": "/app/.vllm-sr"}],
    )
    _record(runtime_config, "restart", "listeners[http].port changes")

    assert core._activation_state(stack) == (
        "Restart required: a change saved in the Dashboard needs `vllm-sr serve` "
        "to apply it (listeners[http].port changes)."
    )
    assert (
        pending_activation.read_pending_activation(runtime_config).reason == "restart"
    )


def test_a_record_the_cli_cannot_read_counts_as_setup(tmp_path):
    config = tmp_path / "runtime-config.yaml"
    pending_activation.record_file(config).write_text("not json")

    assert pending_activation.read_pending_activation(
        config
    ) == pending_activation.PendingActivation(reason="setup")
