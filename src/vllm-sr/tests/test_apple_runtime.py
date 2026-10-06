"""Apple bridge lifecycle and HTTP integration without models or a GPU."""

import hashlib
import http.client
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import pytest
from click.testing import CliRunner

from cli import apple_runtime
from cli.apple_runtime_environment import validate_apple_host
from cli.apple_runtime_server import BridgeServer
from cli.container_backend import ContainerBackend
from cli.container_gpu_isolation import router_runtime_env
from cli.container_run_command import append_env_vars
from cli.container_start import _sensitive_runtime_env_names
from cli.main import main


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
    from cli import apple_runtime_environment as environment

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


def test_host_proxy_keeps_json_contract_and_reaps_process_leases(monkeypatch, tmp_path):
    # A real subprocess provides the runtime HTTP boundary. No torch, models,
    # native dependencies, or Docker installation are involved in this test.
    fixture = """
import json,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_POST(self):
        data=self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
"""
    original = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        port = command[command.index("--port") + 1]
        process = original([sys.executable, "-c", fixture, port], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr("cli.apple_runtime_server.subprocess.Popen", launch)
    server = BridgeServer(
        {
            "token": "private",
            "identity": "fixture",
            "port": 0,
            "directory": str(tmp_path),
            "cache": str(tmp_path),
            "python": sys.executable,
        }
    )
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
    models = [{"model": "vllm-sr/fixture", "name": "fixture", "device": "mps"}]
    try:
        assert call("GET", "/status", token="incorrect")[0] == 401
        assert call("POST", "/processes/" + key, {"models": models})[0] == 200
        assert call("POST", "/processes/" + key, {"models": models})[0] == 200
        assert len(children) == 1
        payload = {"tasks": [{"id": "one", "decisions": {"state": "unchanged"}}]}
        deadline = time.monotonic() + 5
        while True:
            status, body = call("POST", f"/processes/{key}/v1/bundle", payload)
            if status == 200 or time.monotonic() > deadline:
                break
            time.sleep(0.02)
        assert status == 200
        assert body == payload
        server.processes.children[key]["expires"] = time.monotonic() - 1
        server.processes.reap()
        assert children[0].poll() is not None
        assert not server.processes.children
    finally:
        server.shutdown()
        server.processes.close()
        server.server_close()
        worker.join(timeout=5)
