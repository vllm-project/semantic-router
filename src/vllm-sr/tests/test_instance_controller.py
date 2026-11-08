"""State-machine contracts, including recovery before another cutover."""

import copy
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from cli import instance_paths
from cli.commands.instance import instance
from cli.instance_controller import ControllerHandler, ControllerServer
from cli.instance_paths import (
    MAX_UNIX_SOCKET_PATH_BYTES,
    instance_control_directory,
    instance_socket_address,
    prepare_instance_control_directory,
)
from cli.instance_runtime import ContainerInstanceBackend
from cli.instance_setup import attach_controller, controller_request, ensure_controller
from cli.instance_state import InstanceConflictError, InstanceController
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def short_root(monkeypatch):
    """A short private directory that stands in for /tmp, removed afterwards."""

    root = Path(tempfile.mkdtemp(prefix="vsr-", dir="/tmp"))
    monkeypatch.setattr(instance_paths, "_SHORT_TEMPORARY_ROOT", root)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def serve_controller(address, directory):
    server = ControllerServer(address, ControllerHandler)
    server.controller = InstanceController(directory, Backend())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return server, worker


def stop_controller(server, worker):
    server.shutdown()
    server.server_close()
    worker.join(3)


class Backend:
    def __init__(self):
        self.mode = "router"
        self.fail = False
        self.rollback_fail = False
        self.calls = 0

    def observe(self):
        return {"observed_mode": self.mode}

    def prepare(self, mode, deployment, operation):
        return {"mode": mode, "previous_mode": self.mode, "old": []}

    def activate(self, plan, progress):
        self.calls += 1
        progress("starting")
        self.mode = plan["mode"]
        if self.fail:
            raise RuntimeError("private credential must not escape")

    def rollback(self, plan):
        if self.rollback_fail:
            raise RuntimeError("rollback failed")
        self.mode = plan["previous_mode"]

    def native(self, path, payload=None):
        return 200, json.dumps({"object": "list", "data": []}).encode()


def test_control_state_is_outside_dashboard_writable_runtime_tree(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "host-private"))
    monkeypatch.setattr("cli.instance_paths.cli_user_share_gid", os.getgid)
    runtime = tmp_path / "recipe" / ".vllm-sr"
    directory = prepare_instance_control_directory(str(runtime), "one")
    assert not directory.is_relative_to(runtime)
    assert directory.stat().st_mode & 0o077 == 0
    assert (directory / "socket").stat().st_mode & 0o777 == 0o750
    assert instance_control_directory(str(runtime), "two") != directory
    (directory / "socket").rmdir()
    (directory / "socket").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        prepare_instance_control_directory(str(runtime), "one")


def test_unix_controller_serves_status_and_bounded_native_requests(tmp_path):
    directory = tmp_path / "controller"
    (directory / "socket").mkdir(parents=True)
    server, worker = serve_controller(instance_socket_address(directory), directory)
    try:
        assert controller_request(directory, "/status")["observed_mode"] == "router"
        assert controller_request(directory, "/models")["object"] == "list"
        with pytest.raises(ValueError, match="rejected"):
            controller_request(directory, "/systemone", {"url": "http://untrusted"})
        with pytest.raises(ValueError, match="rejected"):
            controller_request(
                directory, "/deploy", {"mode": "invalid", "request_id": "x"}
            )
    finally:
        stop_controller(server, worker)


def test_a_long_home_path_serves_its_controller_through_a_short_address(
    tmp_path, monkeypatch, short_root
):
    monkeypatch.setenv("HOME", str(tmp_path / ("a-long-home-directory-" * 4)))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr("cli.instance_paths.cli_user_share_gid", os.getgid)
    directory = prepare_instance_control_directory(str(tmp_path / "recipe"), "one")
    mounted = directory / "socket" / "control.sock"
    assert len(os.fsencode(mounted)) > MAX_UNIX_SOCKET_PATH_BYTES

    address = Path(instance_socket_address(directory))

    assert len(os.fsencode(address)) <= MAX_UNIX_SOCKET_PATH_BYTES
    assert address.parent.parent == short_root / f"vllm-sr-{os.geteuid()}"
    assert address.parent.parent.stat().st_mode & 0o777 == 0o700
    assert address.parent.resolve() == (directory / "socket").resolve()
    server, worker = serve_controller(str(address), directory)
    try:
        # The Dashboard mounts the socket directory; the socket is there.
        assert mounted.is_socket()
        assert controller_request(directory, "/status")["observed_mode"] == "router"
    finally:
        stop_controller(server, worker)
    assert instance_socket_address(directory) == str(address)


def test_the_short_address_prefers_a_private_xdg_runtime_dir(
    tmp_path, monkeypatch, short_root
):
    runtime = short_root / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    directory = tmp_path / ("d" * 100)

    address = Path(instance_socket_address(directory))
    assert address.parent.parent == runtime / "vllm-sr"
    assert address.parent.resolve() == (directory / "socket").resolve()

    # The spec requires a runtime directory only its user can enter.
    runtime.chmod(0o755)
    fallback = Path(instance_socket_address(directory)).parent.parent
    assert fallback == short_root / f"vllm-sr-{os.geteuid()}"


def test_a_short_state_directory_keeps_its_direct_socket_path(short_root):
    directory = short_root / "instance"

    assert instance_socket_address(directory) == str(
        directory / "socket" / "control.sock"
    )
    assert not (short_root / f"vllm-sr-{os.geteuid()}").exists()


def test_a_fallback_directory_another_user_owns_is_refused_with_the_way_out(
    tmp_path, monkeypatch, short_root
):
    user = os.geteuid()
    squatted = short_root / f"vllm-sr-{user + 1}"
    squatted.mkdir(mode=0o700)
    monkeypatch.setattr(os, "geteuid", lambda: user + 1)

    with pytest.raises(ValueError, match="owned by the current user") as refused:
        instance_socket_address(tmp_path / ("d" * 100))
    assert f"remove {squatted} or set XDG_RUNTIME_DIR" in str(refused.value)


def test_an_address_still_too_long_names_the_limit_and_the_setting(
    tmp_path, monkeypatch
):
    long_root = Path(tempfile.mkdtemp(prefix="vsr-", dir="/tmp")) / ("t" * 80)
    long_root.mkdir()
    monkeypatch.setattr(instance_paths, "_SHORT_TEMPORARY_ROOT", long_root)
    try:
        with pytest.raises(ValueError) as refused:
            instance_socket_address(tmp_path / ("d" * 100))
    finally:
        shutil.rmtree(long_root.parent)
    assert f"longer than the {MAX_UNIX_SOCKET_PATH_BYTES} bytes" in str(refused.value)
    assert "set XDG_RUNTIME_DIR to a shorter private directory" in str(refused.value)


def test_a_controller_that_cannot_start_says_why(short_root):
    directory = short_root / "instance"
    (directory / "socket").mkdir(parents=True)

    # Without a manifest the controller exits at once.
    with pytest.raises(RuntimeError) as failed:
        ensure_controller(directory)

    assert "failed to start: FileNotFoundError" in str(failed.value)
    assert f"its log is {directory / 'controller.log'}" in str(failed.value)


def request(controller, **kwargs):
    controller.submit(
        {"mode": "engine", "deployment": "primary", "request_id": "one", **kwargs}
    )
    controller.worker.join(3)
    assert not controller.worker.is_alive()
    return controller.status()


def test_mode_operation_persists_and_idempotent_retry(tmp_path):
    backend = Backend()
    controller = InstanceController(tmp_path, backend)
    state = request(controller)
    assert state["observed_mode"] == state["desired_mode"] == "engine"
    assert state["operation"]["phase"] == "ready"
    assert backend.calls == 1
    assert request(controller)["operation"]["phase"] == "ready"
    assert backend.calls == 1
    restored = InstanceController(tmp_path, backend)
    assert restored.status()["desired_mode"] == "engine"
    assert (tmp_path / "state.json").stat().st_mode & 0o077 == 0


def test_failed_cutover_restores_previous_mode_without_leaking_error(tmp_path):
    backend = Backend()
    backend.fail = True
    state = request(InstanceController(tmp_path, backend))
    assert state["observed_mode"] == state["desired_mode"] == "router"
    assert state["operation"]["rolled_back"] is True
    assert "credential" not in json.dumps(state)
    assert "rollback" not in state


def test_interrupted_and_failed_rollback_recover_before_new_work(tmp_path):
    backend = Backend()
    backend.fail = backend.rollback_fail = True
    controller = InstanceController(tmp_path, backend)
    state = request(controller)
    assert "can_switch" not in state
    with pytest.raises(InstanceConflictError):
        request(controller, request_id="two")
    backend.rollback_fail = False
    restored = InstanceController(tmp_path, backend)
    restored.recover()
    assert restored.status()["observed_mode"] == "router"
    assert "can_switch" not in restored.status()


def test_conflicting_work_is_rejected_and_retry_not_restarted(tmp_path):
    backend = Backend()
    gate = threading.Event()
    backend.activate = lambda plan, progress: gate.wait(3)
    controller = InstanceController(tmp_path, backend)
    original = {"mode": "engine", "deployment": "primary", "request_id": "one"}
    controller.submit(original)
    assert controller.submit(original)["operation"]["id"] == "one"
    with pytest.raises(InstanceConflictError):
        controller.submit({**original, "request_id": "other"})
    with pytest.raises(InstanceConflictError):
        controller.submit({**original, "mode": "router"})
    gate.set()
    controller.worker.join(3)


def test_preparation_failure_leaves_previous_desire(tmp_path):
    backend = Backend()

    def fail(*_args):
        raise ValueError("unconfigured artifact")

    backend.prepare = fail
    state = request(InstanceController(tmp_path, backend))
    assert state["desired_mode"] == "router"
    assert state["operation"]["phase"] == "failed"


def frontend_backend(tmp_path):
    document = {
        "version": "v0.5",
        "recipes": [{"name": "default", "decisions": [{"name": "kept-router-policy"}]}],
        "global": {
            "router": {"enabled": True},
            "model_catalog": {
                "system": {"decision_model": {"deployment": "primary"}},
                "deployments": {
                    "primary": {"provider": "model_runtime", "artifact": "example/vela"}
                },
            },
        },
    }
    backend = ContainerInstanceBackend(
        {"layout": vars(resolve_runtime_stack()), "runtime": "docker"}, tmp_path
    )
    backend.acquire = lambda: None
    backend.release = lambda: None
    state = {"document": document, "mode": "router", "failed": False, "calls": []}

    def api(path, payload=None, **kwargs):
        state["calls"].append((path, payload, kwargs))
        if path == "/api/v1/config":
            if payload is None:
                kwargs["response_headers"]["ETag"] = '"current"'
                return 200, json.dumps(state["document"]).encode()
            state["document"] = yaml.safe_load(payload["yaml"])
            if state["failed"]:
                state["failed"] = False
                return 200, b'{"activation_status":"failed"}'
            state["mode"] = (
                "router"
                if state["document"]["global"]["router"]["enabled"]
                else "engine"
            )
            return 200, b'{"activation_status":"active"}'
        if path == "/api/v1/instance":
            return (
                200,
                json.dumps(
                    {
                        "observed_mode": state["mode"],
                        "active_deployment": "primary",
                        "model": "example/vela",
                    }
                ).encode(),
            )
        if path == "/api/v1/config/hash":
            return 200, b'{"activation_status":"active"}'
        if path == "/api/v1/diagnostics/models/systemone":
            return (
                200,
                b'{"deployments":[{"id":"primary","artifact":"example/vela","ready":true,"surfaces":["decisions"]}]}',
            )
        raise AssertionError(path)

    backend.api = api
    return backend, state


def test_mode_only_publication_preserves_models_routing_and_container_lifetime(
    tmp_path,
):
    backend, state = frontend_backend(tmp_path)
    before = copy.deepcopy(state["document"])
    with patch("subprocess.run") as containers:
        controller = InstanceController(tmp_path, backend)
        result = request(controller, deployment=None)
        assert result["operation"]["phase"] == "ready"
        assert result["active_deployment"] == "primary"
        assert result["observed_mode"] == "engine"
        assert state["document"]["recipes"] == before["recipes"]
        assert (
            state["document"]["global"]["model_catalog"]
            == before["global"]["model_catalog"]
        )
        request(controller, request_id="back", mode="router", deployment=None)
        assert state["document"] == before
        containers.assert_not_called()
    assert all(path.startswith("/api/v1/") for path, _, _ in state["calls"])


def test_pending_publication_waits_for_active_generation_instead_of_rolling_back(
    tmp_path, monkeypatch
):
    backend, state = frontend_backend(tmp_path)
    original_api = backend.api
    activation_reads = 0

    def api(path, payload=None, **kwargs):
        nonlocal activation_reads
        if path == "/api/v1/config" and payload is not None:
            original_api(path, payload, **kwargs)
            state["mode"] = "router"
            return 202, b'{"activation_status":"pending"}'
        if path == "/api/v1/config/hash":
            activation_reads += 1
            if activation_reads == 1:
                return 200, b'{"activation_status":"pending"}'
            state["mode"] = "engine"
        return original_api(path, payload, **kwargs)

    backend.api = api
    monkeypatch.setattr("cli.instance_runtime.time.sleep", lambda _seconds: None)
    result = request(InstanceController(tmp_path, backend), deployment=None)
    assert result["operation"]["phase"] == "ready"
    assert result["observed_mode"] == "engine"
    assert activation_reads >= 2
    publications = [
        payload
        for path, payload, _ in state["calls"]
        if path == "/api/v1/config" and payload is not None
    ]
    assert len(publications) == 1
    assert result["operation"].get("rolled_back") is not True


def test_failed_generation_restores_previous_config_without_container_cutover(tmp_path):
    backend, state = frontend_backend(tmp_path)
    before = copy.deepcopy(state["document"])
    state["failed"] = True
    with patch("subprocess.run") as containers:
        result = request(InstanceController(tmp_path, backend))
        containers.assert_not_called()
    assert result["operation"]["rolled_back"] is True
    assert result["observed_mode"] == "router"
    assert state["document"] == before
    for name in ["candidate", "previous"]:
        assert next(tmp_path.glob(name + "-*.json")).stat().st_mode & 0o077 == 0


def test_recovery_does_not_overwrite_a_concurrent_configuration_editor(tmp_path):
    backend, state = frontend_backend(tmp_path)
    plan = backend.prepare("engine", None, "recovery")
    state["document"]["global"]["router"]["some_new_setting"] = "another writer"
    with pytest.raises(RuntimeError, match="Configuration changed"):
        backend.rollback(plan)
    assert state["document"]["global"]["router"]["some_new_setting"] == "another writer"


def test_engine_selects_external_deployment_without_owning_its_process(tmp_path):
    backend, state = frontend_backend(tmp_path)
    resource = state["document"]["global"]["model_catalog"]["deployments"]["primary"]
    resource["endpoint"] = "http://owned-by-operator.invalid"
    plan = backend.prepare("engine", "primary", "external")
    assert plan["resource"] == resource
    with pytest.raises(ValueError, match="configured"):
        backend.prepare("engine", "missing", "unknown")


def test_old_request_id_does_not_restart_after_later_operation(tmp_path):
    backend = Backend()
    controller = InstanceController(tmp_path, backend)
    request(controller)
    request(controller, request_id="two", mode="router", deployment=None)
    request(controller)
    assert backend.calls == 2
    assert controller.status()["desired_mode"] == "router"


def test_public_request_preserves_exact_deployment_and_artifact_guard(tmp_path):
    backend, state = frontend_backend(tmp_path)
    payload = {
        "deployment": "vela",
        "expected_artifact": "example/vela",
        "request": {"questions": []},
    }
    backend.native("/systemone", payload)
    path, forwarded, _ = state["calls"][-1]
    assert path == "/api/v1/diagnostics/models/systemone"
    assert forwarded == payload


def test_attach_preserves_actual_accelerator_scope_without_copying_credentials(
    tmp_path,
):
    directory = tmp_path / "controller"
    directory.mkdir()
    active = tmp_path / "active.yaml"
    active.write_text("version: v0.5\n")
    paths = {"effective_config_path": str(active), "models_dir": str(tmp_path)}

    def inspect(command, **_kwargs):
        if "{{json .Config.Env}}" in command:
            return SimpleNamespace(
                stdout=json.dumps(
                    [
                        "VLLM_SR_PLATFORM=rocm",
                        "ROCR_VISIBLE_DEVICES=7",
                        "API_KEY=private",
                    ]
                ),
                returncode=0,
            )
        return SimpleNamespace(stdout="sha256:pinned", returncode=0)

    with (
        patch(
            "cli.instance_setup._prepare_runtime_paths",
            return_value=(None, paths, None),
        ),
        patch(
            "cli.instance_setup.prepare_instance_control_directory",
            return_value=directory,
        ),
        patch("cli.instance_setup.get_container_runtime", return_value="docker"),
        patch("cli.instance_setup.subprocess.run", side_effect=inspect),
        patch("cli.instance_setup.ensure_controller"),
    ):
        attach_controller(str(active), str(active), {}, "standalone", 600)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["env"] == {"VLLM_SR_PLATFORM": "rocm", "ROCR_VISIBLE_DEVICES": "7"}
    assert manifest["router_image"] == manifest["dashboard_image"] == "sha256:pinned"


def test_instance_attach_and_models_use_supported_controller_commands(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text("version: v0.5\n")
    arguments = ["--config", str(source)]
    with (
        patch("cli.commands.instance.attach_controller") as attach,
        patch(
            "cli.commands.instance.controller_request",
            return_value={"object": "list", "data": []},
        ) as invoke,
    ):
        result = CliRunner().invoke(
            instance,
            [
                *arguments,
                "attach",
                "--runtime-config",
                str(source),
                "--gateway",
                "standalone",
            ],
        )
        assert result.exit_code == 0, result.output
        attach.assert_called_once_with(str(source), str(source), {}, "standalone", 600)
        result = CliRunner().invoke(instance, [*arguments, "models"])
        assert result.exit_code == 0, result.output
        assert invoke.call_args.args[1] == "/models"


def test_initial_engine_controller_preserves_already_serving_mode(tmp_path):
    backend = Backend()
    backend.mode = "engine"
    state = InstanceController(tmp_path, backend).status()
    assert state["desired_mode"] == state["observed_mode"] == "engine"
