"""A restart must not feed the departing router a new watched configuration."""

import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml
from cli import runtime_lifecycle
from cli.commands import runtime_paths, runtime_serve_config
from cli.parser import ConfigParseError
from cli.runtime_stack import resolve_runtime_stack


def prepare_replacement(tmp_path, monkeypatch):
    source = tmp_path / "config.yaml"
    source.write_text(
        "version: v0.3\nlisteners:\n- name: main\n  address: 0.0.0.0\n  port: 8888\n"
    )
    old = source.read_bytes()
    active = runtime_paths.materialize_runtime_config(source, old)
    changed = yaml.safe_load(old)
    changed["listeners"][0]["port"] = 8899
    candidate = yaml.safe_dump(changed).encode()
    monkeypatch.setattr(
        runtime_serve_config, "build_effective_config_bytes", lambda *args: candidate
    )
    return source, active, old, candidate


@pytest.mark.parametrize("replace_active", [False, True])
def test_serve_stops_old_runtime_before_watched_config_changes(
    tmp_path, monkeypatch, replace_active
):
    source, active, old, candidate = prepare_replacement(tmp_path, monkeypatch)
    stack = resolve_runtime_stack()
    running = set(stack.runtime_container_names)
    events = []
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_status_strict",
        lambda name: "running" if name in running else "exited",
    )

    def stop(name):
        assert active.read_bytes() == old
        events.append(("stop", name))
        running.remove(name)
        return True

    monkeypatch.setattr(runtime_lifecycle, "container_stop_container", stop)
    monkeypatch.setattr(runtime_lifecycle, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        runtime_lifecycle,
        "acquire_runtime_lifecycle_lock",
        lambda **kwargs: nullcontext(),
    )
    real_replace = runtime_paths.os.replace

    def replace(source_path, target):
        if Path(target) == active:
            assert not running, "active file reached an old live config watcher"
            events.append(("publish", active))
        return real_replace(source_path, target)

    monkeypatch.setattr(runtime_paths.os, "replace", replace)
    _path, _setup, lock = runtime_serve_config._prepare_docker_runtime_config(
        source, None, False, None, (), replace_active
    )
    lock.close()
    assert active.read_bytes() == candidate
    assert events == [("stop", name) for name in stack.runtime_container_names] + [
        ("publish", active)
    ]


@pytest.mark.parametrize("failure", ["stop", "still_running", "inspect"])
def test_failed_restart_preserves_active_and_provenance(tmp_path, monkeypatch, failure):
    source, active, old, _candidate = prepare_replacement(tmp_path, monkeypatch)
    provenance = runtime_paths._runtime_config_provenance_path(active)
    old_provenance = provenance.read_bytes()
    monkeypatch.setattr(runtime_lifecycle, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        runtime_lifecycle,
        "acquire_runtime_lifecycle_lock",
        lambda **kwargs: nullcontext(),
    )

    def inspect(_name):
        if failure == "inspect":
            raise RuntimeError("inspection unavailable")
        return "running"

    monkeypatch.setattr(runtime_lifecycle, "container_status_strict", inspect)
    stopped = []

    def stop(name):
        stopped.append(name)
        return failure != "stop"

    monkeypatch.setattr(runtime_lifecycle, "container_stop_container", stop)
    with pytest.raises(RuntimeError):
        runtime_serve_config._prepare_docker_runtime_config(
            source, None, False, None, (), True
        )
    assert active.read_bytes() == old
    assert provenance.read_bytes() == old_provenance
    if failure == "inspect":
        assert stopped == []


def test_invalid_candidate_does_not_stop_or_publish(tmp_path, monkeypatch):
    source, active, old, _candidate = prepare_replacement(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime_serve_config,
        "build_effective_config_bytes",
        lambda *args: b"version: v0.3\nunknown_surface: true\n",
    )
    stopped = []
    monkeypatch.setattr(
        runtime_serve_config, "stop_runtime_before_config_replacement", stopped.append
    )
    with pytest.raises(ConfigParseError, match="schema validation"):
        runtime_serve_config._prepare_docker_runtime_config(
            source, None, False, None, (), True
        )
    assert active.read_bytes() == old
    assert stopped == []


def test_missing_deployment_does_not_stop_or_publish(tmp_path, monkeypatch):
    source, active, old, candidate = prepare_replacement(tmp_path, monkeypatch)
    document = yaml.safe_load(candidate)
    document["routing"] = {
        "model_bindings": {
            "embedding": {
                "deployment": "missing",
                "contract": "embedding.v1",
                "adapter": "sentence_embedding",
            }
        }
    }
    monkeypatch.setattr(
        runtime_serve_config,
        "build_effective_config_bytes",
        lambda *args: yaml.safe_dump(document).encode(),
    )
    stopped = []
    monkeypatch.setattr(
        runtime_serve_config, "stop_runtime_before_config_replacement", stopped.append
    )
    with pytest.raises(ValueError, match="Unknown model deployment"):
        runtime_serve_config._prepare_docker_runtime_config(
            source, None, False, None, (), True
        )
    assert active.read_bytes() == old
    assert stopped == []


@pytest.mark.parametrize("flag", ["minimal", "readonly"])
def test_setup_flag_conflict_does_not_stop_or_publish(tmp_path, monkeypatch, flag):
    source, active, old, candidate = prepare_replacement(tmp_path, monkeypatch)
    document = yaml.safe_load(candidate)
    document["setup"] = {"mode": True}
    monkeypatch.setattr(
        runtime_serve_config,
        "build_effective_config_bytes",
        lambda *args: yaml.safe_dump(document).encode(),
    )
    stopped = []
    monkeypatch.setattr(
        runtime_serve_config, "stop_runtime_before_config_replacement", stopped.append
    )
    with pytest.raises(ValueError, match="Setup mode requires"):
        runtime_serve_config._prepare_effective_serve_config(
            source,
            resolved_target="docker",
            algorithm=None,
            source_setup_mode=True,
            platform=None,
            recipe_env_bindings=(),
            replace_active_config=True,
            **{flag: True}
        )
    assert active.read_bytes() == old
    assert stopped == []


@pytest.mark.parametrize("preserve", ["identical", "dashboard"])
def test_preserved_document_does_not_stop_during_preparation(
    tmp_path, monkeypatch, preserve
):
    source, active, old, _candidate = prepare_replacement(tmp_path, monkeypatch)
    if preserve == "identical":
        monkeypatch.setattr(
            runtime_serve_config, "build_effective_config_bytes", lambda *args: old
        )
    else:
        active.write_bytes(old + b"# Dashboard edit\n")
    expected = active.read_bytes()
    stopped = []
    monkeypatch.setattr(
        runtime_serve_config, "stop_runtime_before_config_replacement", stopped.append
    )
    _path, _setup, lock = runtime_serve_config._prepare_docker_runtime_config(
        source, None, False, None, (), False
    )
    lock.close()
    assert active.read_bytes() == expected
    assert stopped == []


def test_restart_does_not_remove_container_after_failed_stop(monkeypatch):
    monkeypatch.setattr(runtime_lifecycle, "container_status", lambda name: "running")
    monkeypatch.setattr(
        runtime_lifecycle, "container_stop_container", lambda name: False
    )
    removed = []
    monkeypatch.setattr(runtime_lifecycle, "container_remove_container", removed.append)
    with pytest.raises(RuntimeError, match="stop"):
        runtime_lifecycle.ensure_clean_runtime_container("test-router")
    assert removed == []


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX SIGTERM")
def test_restart_finishes_sigterm_before_old_process_can_reload(tmp_path, monkeypatch):
    source, active, _old, candidate = prepare_replacement(tmp_path, monkeypatch)
    ready = tmp_path / "ready"
    observed = tmp_path / "observed"
    worker = tmp_path / "watcher.py"
    worker.write_text(
        "import signal, sys, time\n"
        "from pathlib import Path\n"
        "active, ready, observed = map(Path, sys.argv[1:])\n"
        "original = active.read_bytes()\n"
        "def stop(*args):\n"
        "    time.sleep(0.05)\n"
        "    observed.write_text('closed')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "ready.touch()\n"
        "while True:\n"
        "    if active.read_bytes() != original:\n"
        "        observed.write_text('reload')\n"
        "        raise SystemExit(1)\n"
        "    time.sleep(0.005)\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(worker), str(active), str(ready), str(observed)]
    )
    try:
        deadline = time.monotonic() + 3
        while not ready.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.005)
        stack = resolve_runtime_stack()
        monkeypatch.setattr(
            runtime_lifecycle, "get_container_runtime", lambda: "docker"
        )
        monkeypatch.setattr(
            runtime_lifecycle,
            "acquire_runtime_lifecycle_lock",
            lambda **kwargs: nullcontext(),
        )
        monkeypatch.setattr(
            runtime_lifecycle,
            "container_status_strict",
            lambda name: (
                "running"
                if name == stack.router_container_name and process.poll() is None
                else "not found"
            ),
        )

        def stop(_name):
            process.terminate()
            return process.wait(timeout=3) == 0

        monkeypatch.setattr(runtime_lifecycle, "container_stop_container", stop)
        _path, _setup, lock = runtime_serve_config._prepare_docker_runtime_config(
            source, None, False, None, (), True
        )
        lock.close()
        assert process.wait(timeout=3) == 0
        assert observed.read_text() == "closed"
        assert active.read_bytes() == candidate
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
