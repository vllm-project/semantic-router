"""A restart must not feed the departing router a new watched configuration."""

import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml
from cli import (
    container_images,
    pending_activation,
    router_validation,
    runtime_lifecycle,
)
from cli.commands import runtime_paths, runtime_serve_config
from cli.parser import ConfigParseError
from cli.runtime_stack import resolve_runtime_stack


def prepare_replacement(tmp_path, monkeypatch):
    source = tmp_path / "config.yaml"
    source.write_text(
        "version: v0.5\nlisteners:\n  - name: main\n    address: 0.0.0.0\n    port: 8888\nglobal:\n  services:\n    observability:\n      tracing:\n        enabled: false\n"
    )
    old = source.read_bytes()
    active = runtime_paths.materialize_runtime_config(source, old)
    changed = yaml.safe_load(old)
    changed["listeners"][0]["port"] = 8899
    candidate = yaml.safe_dump(changed).encode()
    source.write_bytes(candidate)
    monkeypatch.setattr(
        runtime_serve_config,
        "build_effective_config_bytes",
        lambda *args: source.read_bytes(),
    )
    return source, active, old, candidate


# #7222: a change `config apply` staged waits outside the watched file until
# serve stops the Router, then serve makes exactly that document active, even
# over an API apply in between.
@pytest.mark.parametrize("stack_name", ["", "team-a"])
@pytest.mark.parametrize(
    ("replace_active", "staged"),
    [(False, None), (True, None), (False, "staged"), (False, "after an API apply")],
)
def test_serve_stops_old_runtime_before_watched_config_changes(
    tmp_path, monkeypatch, replace_active, staged, stack_name
):
    monkeypatch.setenv("VLLM_SR_STACK_NAME", stack_name)
    source, active, old, candidate = prepare_replacement(tmp_path, monkeypatch)
    if staged:
        stage_restart(source, candidate)
        assert active.read_bytes() == old
    if staged == "after an API apply":
        active.write_bytes(old + b"# applied through the management API\n")
    before = active.read_bytes()
    stack = resolve_runtime_stack()
    running = set(stack.runtime_container_names)
    events = []
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_status_strict",
        lambda name: "running" if name in running else "exited",
    )

    def stop(name):
        assert active.read_bytes() == before
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
    serve(source, replace_active)
    assert active.read_bytes() == candidate
    assert events == [("stop", name) for name in stack.runtime_container_names] + [
        ("publish", active)
    ]
    assert not runtime_paths.pending_restart_path(active).exists()


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
        lambda *args: b"version: v0.5\nunknown_surface: true\n",
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


def test_a_semantic_error_does_not_stop_or_publish(tmp_path, monkeypatch):
    source, active, old, candidate = prepare_replacement(tmp_path, monkeypatch)
    document = yaml.safe_load(candidate)
    document["entrypoints"] = [{"model_names": ["private"], "recipe": "missing"}]
    monkeypatch.setattr(
        runtime_serve_config,
        "build_effective_config_bytes",
        lambda *args: yaml.safe_dump(document).encode(),
    )
    stopped = []
    monkeypatch.setattr(
        runtime_serve_config, "stop_runtime_before_config_replacement", stopped.append
    )
    with pytest.raises(ValueError, match="unknown recipe 'missing'"):
        runtime_serve_config._prepare_docker_runtime_config(
            source, None, False, None, (), True
        )
    assert active.read_bytes() == old
    assert stopped == []


@pytest.mark.parametrize(
    ("listener_port", "management_api", "error"),
    [
        (65100, None, "listener main host port 65600"),
        (8899, {"bind_address": "0.0.0.0", "port": 65100}, "API host port 65600"),
        (8899, {"port": 8085}, "requires management_api.bind_address 0.0.0.0"),
        (8899, {"bind_address": "0.0.0.0", "port": 9190}, "a Router service port"),
    ],
    ids=["listener-host-port", "management-host-port", "bind-address", "service-port"],
)
def test_a_listener_the_start_refuses_does_not_stop_or_publish(
    tmp_path, monkeypatch, listener_port, management_api, error
):
    monkeypatch.setenv("VLLM_SR_PORT_OFFSET", "500")
    source, active, old, candidate = prepare_replacement(tmp_path, monkeypatch)
    document = yaml.safe_load(candidate)
    document["listeners"][0]["port"] = listener_port
    if management_api is not None:
        document["global"]["services"]["management_api"] = management_api
    monkeypatch.setattr(
        runtime_serve_config,
        "build_effective_config_bytes",
        lambda *args: yaml.safe_dump(document).encode(),
    )
    stopped = []
    monkeypatch.setattr(
        runtime_serve_config, "stop_runtime_before_config_replacement", stopped.append
    )
    with pytest.raises(ValueError, match=error):
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
            **{flag: True},
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


def stage_restart(source, document):
    """What `config apply` leaves when the Router answers RESTART_REQUIRED."""

    staged = runtime_paths.stage_pending_restart(source, document)
    active = runtime_paths._runtime_config_output_path(source)
    pending_activation.record_pending_activation(active, document, "restart")
    return staged


def record_replacements(monkeypatch):
    """The documents serve checks, and stops the runtime for, before publishing."""

    checked = []
    monkeypatch.setattr(
        runtime_serve_config,
        "_prepare_runtime_config_replacement",
        lambda data, *args, **kwargs: checked.append(data),
    )
    return checked


def serve(source, replace_active=False):
    _path, _setup, lock = runtime_serve_config._prepare_docker_runtime_config(
        source, None, False, None, (), replace_active
    )
    lock.close()


# A staged file no restart waits for, or of another shape, is dropped before
# it can promote a document or write a receipt, so the edited source serves.
@pytest.mark.parametrize(
    "case",
    [
        "no record",
        "dashboard record",
        "record for another config",
        "replace active",
        "torn",
        "extra key",
        "config not text",
        "bad digest",
        "symlink",
    ],
)
def test_serve_drops_a_staged_change_it_cannot_trust(tmp_path, monkeypatch, case):
    source, active, _old, candidate = prepare_replacement(tmp_path, monkeypatch)
    newer = candidate.replace(b"8899", b"8877")
    source.write_bytes(newer)
    checked = record_replacements(monkeypatch)
    staged = runtime_paths.pending_restart_path(active)
    stage = {
        "source_digest": runtime_paths._digest_bytes(newer),
        "config": candidate.decode(),
    }
    if case == "extra key":
        stage["origin"] = "cli"
    elif case == "config not text":
        stage["config"] = 5
    elif case == "bad digest":
        stage["source_digest"] = "sha256:not-a-digest"
    text = '{"schema_version": ' if case == "torn" else json.dumps(stage)
    if case == "symlink":
        (tmp_path / "elsewhere.json").write_text(text, encoding="utf-8")
        staged.symlink_to(tmp_path / "elsewhere.json")
    else:
        staged.write_text(text, encoding="utf-8")
    if case != "no record":
        pending_activation.record_pending_activation(
            active,
            _old if case == "record for another config" else candidate,
            "restart",
            origin="dashboard" if case == "dashboard record" else "cli",
        )

    serve(source, replace_active=case == "replace active")

    assert active.read_bytes() == newer
    assert checked == [newer]
    assert not staged.exists() and not staged.is_symlink()


# An edit to the source after the apply supersedes the staged change: serve
# checks and publishes only the source, even after an API apply in between.
@pytest.mark.parametrize("applied_in_between", [False, True])
def test_a_source_edit_after_the_apply_supersedes_the_staged_change(
    tmp_path, monkeypatch, applied_in_between
):
    source, active, old, candidate = prepare_replacement(tmp_path, monkeypatch)
    staged = stage_restart(source, candidate)
    if applied_in_between:
        active.write_bytes(old + b"# applied through the management API\n")
    newer = candidate.replace(b"8899", b"8877")
    source.write_bytes(newer)
    checked = record_replacements(monkeypatch)

    serve(source)

    assert active.read_bytes() == newer
    assert checked == [newer]
    assert not staged.exists()


def test_a_staged_change_serve_refuses_leaves_a_way_out(tmp_path, monkeypatch):
    source, active, old, candidate = prepare_replacement(tmp_path, monkeypatch)
    staged = stage_restart(source, candidate)

    def refuse_the_staged_change(data, *args, **kwargs):
        if data == candidate:
            raise ValueError("Invalid runtime config: refused")

    monkeypatch.setattr(
        runtime_serve_config,
        "_prepare_runtime_config_replacement",
        refuse_the_staged_change,
    )
    with pytest.raises(ValueError, match="refused"):
        serve(source)
    assert active.read_bytes() == old
    assert staged.exists()

    newer = candidate.replace(b"8899", b"8877")
    source.write_bytes(newer)
    serve(source)

    assert active.read_bytes() == newer
    assert not staged.exists()


def test_an_interrupted_promotion_promotes_again(tmp_path, monkeypatch):
    source, active, _old, candidate = prepare_replacement(tmp_path, monkeypatch)
    staged = stage_restart(source, candidate)
    record_replacements(monkeypatch)
    write = runtime_paths._atomic_write_private_bytes

    def fail_before_the_document(path, data):
        if path == active:
            raise OSError("No space left on device")
        write(path, data)

    monkeypatch.setattr(
        runtime_paths, "_atomic_write_private_bytes", fail_before_the_document
    )
    with pytest.raises(OSError):
        serve(source)
    assert staged.exists()
    monkeypatch.setattr(runtime_paths, "_atomic_write_private_bytes", write)

    serve(source)

    assert active.read_bytes() == candidate
    assert not staged.exists()


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


def _served_router(monkeypatch, *, verdict, pull=None):
    """Fake the host: the image the start would pull, and what it says."""

    ensured, validated = [], []
    real_run = subprocess.run

    def ensure(image, pull_policy):
        ensured.append((image, pull_policy))
        if pull is not None:
            pull()

    def run(command, **kwargs):
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, "2026-11-06T10:12:13Z", "")
        if "-validate-config" not in command:
            return real_run(command, **kwargs)
        validated.append(command[command.index("-validate-config") - 1])
        return subprocess.CompletedProcess(
            command, 0 if verdict["valid"] else 1, json.dumps(verdict).encode(), b""
        )

    monkeypatch.setattr(router_validation, "_container_runtime_installed", lambda: True)
    monkeypatch.setattr(router_validation, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(router_validation.subprocess, "run", run)
    monkeypatch.setattr(container_images, "_ensure_image_available", ensure)
    for name in ("VLLM_SR_ROUTER_IMAGE", "VLLM_SR_IMAGE"):
        monkeypatch.delenv(name, raising=False)
    return ensured, validated


def _record_stops(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        runtime_serve_config,
        "stop_runtime_before_config_replacement",
        stopped.append,
    )
    return stopped


@pytest.mark.router_image
def test_serve_keeps_old_runtime_when_router_refuses_new_config(tmp_path, monkeypatch):
    source, active, old, _candidate = prepare_replacement(tmp_path, monkeypatch)
    _served_router(
        monkeypatch,
        verdict={
            "valid": False,
            "error": "routing.model_bindings.pii: no deployment named redactor",
        },
    )
    stopped = _record_stops(monkeypatch)

    with pytest.raises(ValueError, match="no deployment named redactor"):
        runtime_serve_config._prepare_docker_runtime_config(
            source, None, False, None, (), True, image="router:new"
        )

    assert not stopped
    assert active.read_bytes() == old


@pytest.mark.router_image
@pytest.mark.parametrize(
    ("options", "environment", "expected"),
    [
        ({"image": "router:new"}, {}, "router:new"),
        ({"image": "base:new", "router_image": "router:new"}, {}, "router:new"),
        ({}, {"VLLM_SR_IMAGE": "pinned:new"}, "pinned:new"),
        ({}, {}, container_images.VLLM_SR_CONTAINER_IMAGE_DEFAULT),
    ],
)
def test_serve_checks_a_new_config_with_the_image_it_runs(
    tmp_path, monkeypatch, options, environment, expected
):
    source, active, _old, candidate = prepare_replacement(tmp_path, monkeypatch)
    ensured, validated = _served_router(monkeypatch, verdict={"valid": True})
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    # Neither a local binary nor the running stack's older image may judge it.
    monkeypatch.setenv(router_validation.ROUTER_BINARY_ENV, str(tmp_path / "router"))
    monkeypatch.setattr(
        router_validation, "_running_router", lambda: ("stack:old", "old-id")
    )
    stopped = _record_stops(monkeypatch)

    _path, _setup, lock = runtime_serve_config._prepare_docker_runtime_config(
        source, None, False, "cpu", (), True, image_pull_policy="never", **options
    )
    lock.close()

    assert validated == [expected]
    assert ensured == [(expected, "never")]
    assert stopped and active.read_bytes() == candidate


@pytest.mark.router_image
def test_a_failed_pull_of_the_served_image_leaves_the_old_runtime(
    tmp_path, monkeypatch
):
    source, active, old, _candidate = prepare_replacement(tmp_path, monkeypatch)

    def failed_pull():
        raise SystemExit(1)

    _ensured, validated = _served_router(
        monkeypatch, verdict={"valid": True}, pull=failed_pull
    )
    stopped = _record_stops(monkeypatch)

    with pytest.raises(SystemExit):
        runtime_serve_config._prepare_docker_runtime_config(
            source, None, False, "cpu", (), True, image="router:new"
        )

    assert not validated and not stopped
    assert active.read_bytes() == old


@pytest.mark.router_image
@pytest.mark.parametrize(
    ("checkout", "update"),
    [
        (None, ""),
        (
            Path("/src/semantic-router"),
            "rebuild it from this checkout (`make -C /src/semantic-router "
            "vllm-sr-build`) and serve it with `--image-pull-policy ifnotpresent` or ",
        ),
    ],
)
def test_a_moving_tag_serve_just_pulled_is_not_pulled_again(
    tmp_path, monkeypatch, checkout, update
):
    source, _active, _old, _candidate = prepare_replacement(tmp_path, monkeypatch)
    _served_router(monkeypatch, verdict={"valid": False, "error": "unknown field"})
    monkeypatch.setattr(router_validation, "_source_checkout", lambda: checkout)
    image = container_images.VLLM_SR_CONTAINER_IMAGE_DEFAULT.rsplit(":", 1)[0]
    monkeypatch.setattr(
        router_validation, "VLLM_SR_CONTAINER_IMAGE_DEFAULT", f"{image}:latest"
    )
    monkeypatch.setattr(container_images, "_PULLED_IMAGES", {f"{image}:latest"})
    _record_stops(monkeypatch)

    with pytest.raises(ValueError) as refused:
        runtime_serve_config._prepare_docker_runtime_config(
            source, None, False, "cpu", (), True, image=f"{image}:latest"
        )

    assert str(refused.value).endswith(
        "`latest` is a moving tag: this local copy was built on 2026-11-06 and may "
        f"be older than this CLI; {update}name the Router with `--image` or "
        "`VLLM_SR_IMAGE`."
    )
