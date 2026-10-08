"""State-machine contracts, including recovery before another cutover."""

import copy
import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cli.commands.instance import instance
from cli.instance_controller import ControllerHandler, ControllerServer
from cli.instance_paths import (
    instance_control_directory,
    prepare_instance_control_directory,
)
from cli.instance_runtime import ContainerInstanceBackend
from cli.instance_setup import attach_controller, controller_request
from cli.instance_state import InstanceConflictError, InstanceController
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner


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
    server = ControllerServer(
        str(directory / "socket" / "control.sock"), ControllerHandler
    )
    server.controller = InstanceController(directory, Backend())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        assert controller_request(directory, "/status")["observed_mode"] == "router"
        assert controller_request(directory, "/models")["object"] == "list"
        with pytest.raises(ValueError, match="rejected"):
            controller_request(directory, "/systemone", {"url": "http://untrusted"})
        with pytest.raises(ValueError, match="rejected"):
            controller_request(
                directory, "/deploy", {"mode": "engine", "request_id": "x"}
            )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(3)


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
    assert state["can_switch"] is False
    with pytest.raises(InstanceConflictError):
        request(controller, request_id="two")
    backend.rollback_fail = False
    restored = InstanceController(tmp_path, backend)
    restored.recover()
    assert restored.status()["observed_mode"] == "router"
    assert restored.status()["can_switch"] is True


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


def test_engine_uses_canonical_catalog_and_rejects_external_owner(tmp_path):
    layout = resolve_runtime_stack()
    manifest = {"layout": layout.__dict__, "runtime": "docker", "config": "config.yaml"}
    backend = ContainerInstanceBackend(manifest, tmp_path)
    backend.acquire = lambda: None
    backend.release = lambda: None
    backend.observe = lambda: {"observed_mode": "router"}
    backend.inspect = lambda _name: None
    resource = {"provider": "model_runtime", "artifact": "vllm-sr/example"}
    config = SimpleNamespace(
        global_={"model_catalog": {"deployments": {"primary": resource}}}
    )
    with patch("cli.instance_runtime.parse_user_config", return_value=config):
        plan = backend.prepare("engine", "primary", "id")
        assert plan["resource"] == resource
        resource["endpoint"] = "http://external.invalid"
        with pytest.raises(ValueError, match="externally owned"):
            backend.prepare("engine", "primary", "id")


def test_engine_command_preserves_gpu_scope_and_uses_private_port(
    tmp_path, monkeypatch
):
    layout = resolve_runtime_stack()
    manifest = {
        "layout": layout.__dict__,
        "runtime": "docker",
        "router_image": "sha256:pinned",
        "models_dir": str(tmp_path),
        "env": {"VLLM_SR_PLATFORM": "amd"},
    }
    backend = ContainerInstanceBackend(manifest, tmp_path)
    monkeypatch.setenv("VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES", "7")
    plan = {
        "deployment": "primary",
        "resource": {
            "artifact": "vllm-sr/example",
            "public_name": "friendly-public-id",
            "provider": "model_runtime",
            "device": "rocm",
        },
        "engine_models": "engine-one.yaml",
    }
    with patch(
        "cli.instance_runtime.subprocess.run",
        return_value=SimpleNamespace(returncode=0),
    ) as run:
        backend.start_engine(plan)
    command = run.call_args.args[0]
    assert "127.0.0.1::8100" in command
    assert "ROCR_VISIBLE_DEVICES=7" in command
    assert "--detach" in command and "--rm" not in command
    assert "sha256:pinned" in command and "vllm-srun" in command
    assert "unless-stopped" in command
    assert "vllm-sr.model=vllm-sr/example" in command
    assert "vllm-sr.model=friendly-public-id" not in command
    assert (tmp_path / "engine-one.yaml").stat().st_mode & 0o077 == 0


def test_container_failure_restores_known_router_config_and_leaves_dashboard(tmp_path):
    layout = resolve_runtime_stack()
    active = tmp_path / "active.yaml"
    active.write_text("pending-new-config")
    known = tmp_path / "router-ready.yaml"
    known.write_text("known-running-config")
    manifest = {
        "layout": layout.__dict__,
        "runtime": "docker",
        "config": str(active),
        "state_root": str(tmp_path),
        "env": {},
    }
    backend = ContainerInstanceBackend(manifest, tmp_path)
    containers = {
        "router-old": {
            "Id": "router-old",
            "Name": "/" + layout.router_container_name,
            "State": {"Running": True},
            "Config": {"Labels": {}},
        },
        "dashboard": {
            "Id": "dashboard",
            "Name": "/" + layout.dashboard_container_name,
            "State": {"Running": True},
            "Config": {"Labels": {}},
        },
    }

    def inspect(name):
        for item in containers.values():
            if name in {item["Id"], item["Name"].lstrip("/")}:
                return copy.deepcopy(item)
        return None

    def command(operation, *args, **_kwargs):
        if operation == "stop":
            containers[args[-1]]["State"]["Running"] = False
        elif operation == "rename":
            containers[args[0]]["Name"] = "/" + args[1]
        elif operation == "start":
            containers[args[0]]["State"]["Running"] = True
        elif operation == "rm":
            del containers[args[-1]]
        return SimpleNamespace(returncode=0)

    def failed_start(plan):
        containers["engine-new"] = {
            "Id": "engine-new",
            "Name": "/" + backend.engine_name,
            "State": {"Running": True},
            "Config": {"Labels": {"vllm-sr.operation": plan["operation"]}},
        }
        raise RuntimeError("startup failed")

    backend.inspect, backend.command = inspect, command
    backend.acquire = lambda: None
    backend.release = lambda: None
    backend.wait_ready = lambda _mode: None
    backend.start_engine = failed_start
    config = SimpleNamespace(
        global_={
            "model_catalog": {
                "deployments": {
                    "primary": {
                        "provider": "model_runtime",
                        "artifact": "vllm-sr/example",
                    }
                }
            }
        }
    )
    with patch("cli.instance_runtime.parse_user_config", return_value=config):
        state = request(InstanceController(tmp_path, backend))
    assert state["operation"]["rolled_back"]
    assert containers["router-old"]["State"]["Running"]
    assert containers["dashboard"]["State"]["Running"]
    assert containers["dashboard"]["Name"] == "/" + layout.dashboard_container_name
    assert "engine-new" not in containers
    assert active.read_text() == "known-running-config"
    assert next(tmp_path.glob("failed-*.yaml")).read_text() == "pending-new-config"


def test_old_request_id_does_not_restart_after_later_operation(tmp_path):
    backend = Backend()
    controller = InstanceController(tmp_path, backend)
    request(controller)
    request(controller, request_id="two", mode="router", deployment=None)
    request(controller)
    assert backend.calls == 2
    assert controller.status()["desired_mode"] == "router"


def test_engine_public_request_never_substitutes_active_model(tmp_path):
    backend = ContainerInstanceBackend(
        {"layout": vars(resolve_runtime_stack()), "runtime": "docker"}, tmp_path
    )
    backend.observe = lambda: {
        "observed_mode": "engine",
        "active_deployment": "kai",
        "model": "example/kai",
    }
    backend.endpoint = lambda _mode: ("http://127.0.0.1:1234", {})
    with patch("cli.instance_runtime.bounded_request") as invoke:
        assert (
            backend.native("/systemone", {"deployment": "vela", "request": {}})[0]
            == 503
        )
        assert (
            backend.native(
                "/systemone",
                {
                    "deployment": "kai",
                    "expected_artifact": "example/vela",
                    "request": {},
                },
            )[0]
            == 409
        )
        invoke.assert_not_called()


def test_attach_preserves_actual_accelerator_scope_without_copying_credentials(
    tmp_path,
):
    directory = tmp_path / "controller"
    directory.mkdir()
    active = tmp_path / "active.yaml"
    active.write_text("version: v0.3\n")
    paths = {"effective_config_path": str(active), "models_dir": str(tmp_path)}

    def inspect(command, **_kwargs):
        if command[3] == "{{json .Config.Env}}":
            return SimpleNamespace(
                stdout=json.dumps(
                    [
                        "VLLM_SR_PLATFORM=amd",
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
    assert manifest["env"] == {"VLLM_SR_PLATFORM": "amd", "ROCR_VISIBLE_DEVICES": "7"}
    assert manifest["router_image"] == manifest["dashboard_image"] == "sha256:pinned"


def test_instance_attach_and_models_use_supported_controller_commands(tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text("version: v0.3\n")
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
