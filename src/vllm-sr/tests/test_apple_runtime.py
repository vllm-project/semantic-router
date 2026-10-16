"""Apple bridge lifecycle and HTTP integration without models or a GPU."""

import hashlib
import http.client
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from cli import apple_runtime, container_start_paths
from cli import apple_runtime_environment as environment
from cli.apple_runtime_environment import validate_apple_host
from cli.apple_runtime_server import BridgeServer, Processes
from cli.commands import runtime as runtime_commands
from cli.container_backend import ContainerBackend
from cli.container_gpu_isolation import router_runtime_env
from cli.container_run_command import append_env_vars
from cli.container_start import _sensitive_runtime_env_names
from cli.main import main
from cli.runtime_stack import resolve_runtime_stack
from click.testing import CliRunner


@pytest.mark.parametrize(
    "system,machine", [("Linux", "arm64"), ("Darwin", "x86_64"), ("Windows", "AMD64")]
)
def test_unsupported_apple_host_fails_before_workspace_creation(
    monkeypatch, tmp_path, system, machine
):
    monkeypatch.setattr("cli.apple_runtime_environment.platform.system", lambda: system)
    monkeypatch.setattr(
        "cli.apple_runtime_environment.platform.machine", lambda: machine
    )

    def docker_unavailable():
        raise AssertionError("Docker was checked before rejecting the unsupported host")

    monkeypatch.setattr(
        "cli.commands.runtime.get_container_runtime", docker_unavailable
    )
    path = tmp_path / "missing.yaml"
    result = CliRunner().invoke(
        main, ["serve", "--platform", "apple", "--config", str(path)]
    )
    assert result.exit_code != 0
    assert "Apple silicon" in result.output
    assert not path.exists()


def test_apple_rejects_kubernetes_before_host_detection():
    with pytest.raises(ValueError, match="Docker target"):
        validate_apple_host("k8s", "docker")


def test_private_host_credential_is_inherited_without_docker_argv_leak(monkeypatch):
    monkeypatch.setenv(apple_runtime.ENDPOINT_ENV, "http://host.docker.internal:8000")
    monkeypatch.setenv(apple_runtime.TOKEN_ENV, "private-supervisor-token")
    common = {"VLLM_SR_PLATFORM": "apple"}
    router = router_runtime_env(common, "apple")
    command = []
    append_env_vars(command, router, _sensitive_runtime_env_names(common, {}))
    assert apple_runtime.TOKEN_ENV in command
    assert "private-supervisor-token" not in " ".join(command)
    assert apple_runtime.TOKEN_ENV not in common


def test_docker_context_overrides_docker_host_for_apple_preflight(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "unix:///local.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote-context")
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        return json.dumps([{"Endpoints": {"docker": {"Host": "ssh://remote"}}}])

    monkeypatch.setattr(environment, "_run", invoke)
    with pytest.raises(ValueError, match="remote daemon"):
        environment.validate_local_docker()
    assert calls == [["docker", "context", "inspect", "remote-context"]]


def test_stop_cleans_host_even_when_docker_fails(monkeypatch):
    stopped = []
    monkeypatch.setattr(ContainerBackend, "_lifecycle_lock", lambda self: _noop())

    def fail():
        raise RuntimeError("Docker unavailable")

    monkeypatch.setattr("cli.container_backend.stop_vllm_sr", fail)
    monkeypatch.setattr("cli.container_backend.stop_managed_controller", lambda: None)
    monkeypatch.setattr(apple_runtime, "stop_bridge", lambda: stopped.append(True))
    with pytest.raises(RuntimeError, match="Docker unavailable"):
        ContainerBackend().teardown()
    assert stopped == [True]


@contextmanager
def _noop():
    yield


def test_stale_pid_never_signals_an_unrelated_process(monkeypatch, tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{}")
    monkeypatch.setattr(apple_runtime, "state_path", lambda: state)
    monkeypatch.setattr(
        apple_runtime,
        "read_state",
        lambda: {"pid": 999, "process_identity": "owned-start-time owned-command"},
    )
    monkeypatch.setattr(
        apple_runtime, "process_identity", lambda pid: "different-process"
    )

    def forbidden(*args):
        raise AssertionError("signalled an unrelated process")

    monkeypatch.setattr(apple_runtime.os, "killpg", forbidden)
    apple_runtime.stop_bridge()
    assert not state.exists()


@pytest.mark.parametrize("local_model", [False, True])
def test_host_proxy_keeps_json_contract_and_reaps_process_leases(
    monkeypatch, tmp_path, local_model
):
    # A real subprocess provides the runtime HTTP boundary. No torch, models,
    # native dependencies, or Docker installation are involved in this test.
    fixture = """
import sys
from http.server import BaseHTTPRequestHandler
from socketserver import UnixStreamServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_POST(self):
        data=self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
UnixStreamServer(sys.argv[1],Handler).serve_forever()
"""
    original = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        worker_socket = command[command.index("--uds") + 1]
        assert "--port" not in command and "--host" not in command
        assert len(worker_socket.encode()) < 104
        assert not Path(worker_socket).parent.stat().st_mode & 0o077
        process = original([sys.executable, "-c", fixture, worker_socket], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr("cli.apple_runtime_server.subprocess.Popen", launch)
    monkeypatch.setattr(Processes, "_check_models", lambda self, path: None)
    monkeypatch.chdir(tmp_path)
    local_directory = tmp_path / "my-model"
    local_directory.mkdir()
    server = BridgeServer(
        {
            "token": "private",
            "identity": "fixture",
            "port": 0,
            "directory": str(tmp_path),
            "cache": str(tmp_path),
            "python": sys.executable,
            "models_directory": str(tmp_path),
        }
    )
    # Drive reap explicitly so the expiry/renewal ordering below is deterministic;
    # the HTTP accept loop must not start a second reaper during the barrier.
    server._reaper_stop.set()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def call(method, path, payload=None, token="private"):
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        try:
            connection.request(
                method,
                path,
                json.dumps(payload) if payload is not None else None,
                {"Authorization": "Bearer " + token},
            )
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    key = hashlib.sha256(b"model-plan").hexdigest()
    artifact = "/app/models/my-model" if local_model else "vllm-sr/fixture"
    models = [{"model": artifact, "name": "fixture", "device": "mps"}]
    owner = hashlib.sha256(b"original-group").hexdigest()
    inventory_request = {"models": models, "lease": owner}

    def expire():
        with server.processes.lock:
            child = server.processes.children[key]
            child["leases"] = {lease: time.monotonic() - 1 for lease in child["leases"]}

    try:
        assert call("GET", "/status", token="incorrect")[0] == 401
        assert call("POST", "/processes/" + key, inventory_request)[0] == 200
        assert call("POST", "/processes/" + key, inventory_request)[0] == 200
        assert len(children) == 1
        assert call("POST", "/engine", {})[0] == 404
        inventory = json.loads(
            server.processes.children[key]["models_path"].read_text()
        )
        expected_artifact = str(local_directory.resolve()) if local_model else artifact
        assert inventory["models"][0]["model"] == expected_artifact
        payload = {"tasks": [{"id": "one", "decisions": {"state": "unchanged"}}]}
        deadline = time.monotonic() + 5
        while True:
            status, body = call("POST", f"/processes/{key}/v1/bundle", payload)
            if status == 200 or time.monotonic() > deadline:
                break
            time.sleep(0.02)
        assert status == 200
        assert body == payload
        # Force the renewal/reaper race: pause after the expiry snapshot, let
        # HTTP acknowledge a renewal, then allow the stale reaper to continue.
        selected = threading.Event()
        resume = threading.Event()
        errors = []
        original_stop = server.processes.stop

        def paused_stop(key, **kwargs):
            selected.set()
            if not resume.wait(timeout=5):
                raise TimeoutError("renewal regression did not resume the reaper")
            original_stop(key, **kwargs)

        def reap():
            try:
                server.processes.reap()
            except BaseException as error:
                errors.append(error)

        expire()
        with monkeypatch.context() as patch:
            patch.setattr(server.processes, "stop", paused_stop)
            reaper = threading.Thread(target=reap, daemon=True)
            reaper.start()
            try:
                assert selected.wait(timeout=5)
                assert call("POST", "/processes/" + key, inventory_request)[0] == 200
                assert server.processes.children[key]["expires"] > time.monotonic()
            finally:
                resume.set()
                reaper.join(timeout=5)
            assert not reaper.is_alive()
            assert not errors
        assert children[0].poll() is None
        assert server.processes.children[key]["process"] is children[0]
        assert len(children) == 1

        # Retirement can overlap reacquisition of the same plan. Pause DELETE
        # at the host, acquire a new owner through HTTP, and resume the old release.
        deleting = threading.Event()
        release = threading.Event()
        responses = []

        def paused_release(key, **kwargs):
            deleting.set()
            if not release.wait(timeout=5):
                raise TimeoutError("reacquisition regression did not resume DELETE")
            return original_stop(key, **kwargs)

        def retire():
            try:
                responses.append(call("DELETE", "/processes/" + key, {"lease": owner}))
            except BaseException as error:
                errors.append(error)

        next_owner = hashlib.sha256(b"reacquired-group").hexdigest()
        with monkeypatch.context() as patch:
            patch.setattr(server.processes, "stop", paused_release)
            retiring = threading.Thread(target=retire, daemon=True)
            retiring.start()
            try:
                assert deleting.wait(timeout=5)
                assert (
                    call(
                        "POST",
                        "/processes/" + key,
                        {"models": models, "lease": next_owner},
                    )[0]
                    == 200
                )
            finally:
                release.set()
                retiring.join(timeout=5)
            assert not retiring.is_alive()
            assert not errors
        assert responses == [(200, {"stopped": False})]
        assert children[0].poll() is None
        assert len(children) == 1
        assert set(server.processes.children[key]["leases"]) == {next_owner}
        # A duplicate release from the retired group is harmless.
        assert call("DELETE", "/processes/" + key, {"lease": owner}) == (
            200,
            {"stopped": False},
        )
        expire()
        server.processes.reap()
        assert children[0].poll() is not None
        assert not server.processes.children

        # A slow retirement must not serialize unrelated HTTP traffic or
        # prevent the same plan from starting a new, independently owned child.
        assert call("POST", "/processes/" + key, inventory_request)[0] == 200
        slow = children[-1]
        old_directory = server.processes.children[key]["directory"]
        other = hashlib.sha256(b"other-plan").hexdigest()
        assert call("POST", "/processes/" + other, inventory_request)[0] == 200
        for _ in range(50):
            if (
                call("POST", f"/processes/{other}/v1/classify", {"inputs": ["ready"]})[
                    0
                ]
                == 200
            ):
                break
            time.sleep(0.05)
        waiting, finish = threading.Event(), threading.Event()
        original_wait = slow.wait

        def slow_wait(timeout=None):
            waiting.set()
            if not finish.wait(timeout=10):
                raise TimeoutError("slow-retirement regression was not resumed")
            return original_wait(timeout=timeout)

        replies = []

        def stop_slow():
            try:
                replies.append(call("DELETE", "/processes/" + key, {"lease": owner}))
            except BaseException as error:
                errors.append(error)

        with monkeypatch.context() as patch:
            patch.setattr(slow, "wait", slow_wait)
            retiring = threading.Thread(target=stop_slow, daemon=True)
            retiring.start()
            try:
                assert waiting.wait(timeout=5)
                body = {"inputs": ["unrelated model remains available"]}
                assert call("POST", f"/processes/{other}/v1/classify", body) == (
                    200,
                    body,
                )
                assert call("POST", "/processes/" + other, inventory_request)[0] == 200
                assert (
                    call(
                        "POST",
                        "/processes/" + key,
                        {"models": models, "lease": next_owner},
                    )[0]
                    == 200
                )
                replacement = server.processes.children[key]
                assert replacement["directory"] != old_directory
            finally:
                finish.set()
                retiring.join(timeout=5)
            assert not retiring.is_alive()
        assert not errors
        assert replies == [(200, {"stopped": True})]
        assert replacement["process"].poll() is None
        assert replacement["directory"].is_dir()
        assert not old_directory.exists()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("engine_mode", [False, True])
@pytest.mark.parametrize("model", [None, "vllm-sr/Vela-2.0-0.3B"])
def test_apple_preserves_the_common_instance_mode_contract(
    monkeypatch, tmp_path, engine_mode, model
):
    monkeypatch.chdir(tmp_path)
    received = []
    monkeypatch.setattr(
        runtime_commands,
        "_execute_serve",
        lambda *args, **kwargs: received.append(kwargs),
    )
    args = ["serve", "--platform", "apple"]
    if model:
        args.append(model)
    if engine_mode:
        args.append("--engine")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert len(received) == 1
    assert received[0]["engine"] is engine_mode
    assert (received[0]["model_options"] or {}).get("artifact") == model


@pytest.mark.parametrize("flag", ["--models", "--device", "--host", "--port", "--uds"])
def test_serve_does_not_reintroduce_removed_bare_runtime_options(flag):
    result = CliRunner().invoke(main, ["serve", flag, "value"])
    assert result.exit_code == 2
    assert "No such option" in result.output


@pytest.mark.parametrize("option", [["-dp", "2"], ["--device-ids", "0"]])
def test_apple_rejects_replica_flags_before_docker_and_workspace(
    monkeypatch, tmp_path, option
):
    monkeypatch.setattr(environment.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(environment.platform, "machine", lambda: "arm64")

    def forbidden():
        raise AssertionError(
            "Docker must not be consulted for rejected placement flags"
        )

    monkeypatch.setattr(runtime_commands, "get_container_runtime", forbidden)
    config = tmp_path / "missing.yaml"
    result = CliRunner().invoke(
        main, ["serve", "--platform", "apple", "--config", str(config), *option]
    )
    assert result.exit_code == 2, result.output
    assert "does not support" in result.output
    assert not config.exists()


def test_background_reaping_never_blocks_the_http_accept_loop(monkeypatch, tmp_path):
    server = BridgeServer({"port": 0, "token": "private", "identity": "fixture"})
    entered, resume = threading.Event(), threading.Event()

    def slow_reap():
        entered.set()
        if not resume.wait(timeout=10):
            raise TimeoutError("reaper regression was not resumed")

    monkeypatch.setattr(server.processes, "reap", slow_reap)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    try:
        assert entered.wait(timeout=5)
        connection.request(
            "GET", "/status", headers={"Authorization": "Bearer private"}
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["identity"] == "fixture"
    finally:
        resume.set()
        connection.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("managed", [False, True])
def test_host_model_mapping_uses_the_actual_container_mount_source(
    monkeypatch, tmp_path, managed
):
    source = tmp_path / "config.yaml"
    monkeypatch.setattr(
        container_start_paths,
        "resolve_active_recipe_directory",
        lambda path: object() if managed else None,
    )
    expected = container_start_paths.models_mount_source(str(source), str(tmp_path))
    actual, _, _ = container_start_paths._prepare_runtime_directories(
        str(tmp_path),
        str(tmp_path / ".vllm-sr"),
        resolve_runtime_stack(),
        managed_recipe=managed,
    )
    assert expected == Path(actual)
    assert expected.is_dir()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "absolute"])
def test_local_host_models_cannot_escape_the_stack_mount(monkeypatch, tmp_path, kind):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    artifact = {
        "traversal": "/app/models/../outside",
        "symlink": "/app/models/linked",
        "absolute": str(outside),
    }[kind]
    processes = Processes({"models_directory": str(root)})
    try:
        with pytest.raises(ValueError, match="mounted models directory"):
            processes.launch("a" * 64, [{"model": artifact}], lease="b" * 64)
        assert not processes.children
    finally:
        processes.close()


def test_placement_rejection_never_spawns_a_worker(monkeypatch, tmp_path):
    processes = Processes({"directory": str(tmp_path)})

    def reject(path):
        raise ValueError("decision-kai: no BF16 support; set dtype: float32")

    def forbidden(*args, **kwargs):
        raise AssertionError("a worker started before its placement was accepted")

    monkeypatch.setattr(processes, "_check_models", reject)
    monkeypatch.setattr("cli.apple_runtime_server.subprocess.Popen", forbidden)
    try:
        with pytest.raises(ValueError, match=r"decision-kai.*dtype: float32"):
            processes.launch("a" * 64, [{"model": "vllm-sr/fixture"}], lease="b" * 64)
        assert not processes.children
    finally:
        processes.close()
