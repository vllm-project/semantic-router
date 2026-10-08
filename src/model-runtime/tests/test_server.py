"""The real server process over a Unix socket and over TCP."""

import http.client
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from .conftest import QUESTIONS, STATE

SOURCE = Path(__file__).resolve().parents[1]


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=10):
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def request(connection, method, path, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    connection.request(
        method, path, body=payload, headers={"content-type": "application/json"}
    )
    response = connection.getresponse()
    return response.status, response.read()


def start(args):
    env = dict(
        os.environ,
        PYTHONPATH=str(SOURCE) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm_srun",
            "serve",
            *args,
            "--device",
            "cpu",
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def wait_ready(connect, process, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(process.stdout.read().decode())
        try:
            status, _ = request(connect(), "GET", "/health")
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.2)
    raise AssertionError("runtime did not become ready")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("transport", ["uds", "tcp"])
def test_server_process(qwen35_package, tmp_path, transport):
    if transport == "uds":
        path = str(tmp_path / "run" / "runtime.sock")
        process = start([str(qwen35_package), "--uds", path])

        def connect():
            return UnixConnection(path)

    else:
        port = free_port()
        process = start([str(qwen35_package), "--port", str(port)])

        def connect():
            return http.client.HTTPConnection("127.0.0.1", port, timeout=10)

    try:
        wait_ready(connect, process)
        if transport == "uds":
            assert oct(os.stat(Path(path).parent).st_mode & 0o777) == "0o700"
        status, body = request(
            connect(), "POST", "/v1/decisions", {"state": STATE, "questions": QUESTIONS}
        )
        assert status == 200
        answers = json.loads(body)["answers"]
        assert set(answers) == set(QUESTIONS) and all(
            "error" not in a for a in answers.values()
        )
        connection = connect()
        connection.request(
            "POST",
            "/v1/bundle",
            body=json.dumps(
                {
                    "tasks": [
                        {
                            "id": "t",
                            "decisions": {"state": STATE, "questions": QUESTIONS},
                        }
                    ]
                }
            ),
            headers={"content-type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200 and response.read()
        timing = response.getheader("server-timing", "")
        assert timing.startswith("parse;dur=") and ", total;dur=" in timing
        status, body = request(connect(), "GET", "/v1/models")
        assert status == 200 and json.loads(body)["data"][0]["ready"]
    finally:
        process.terminate()
        process.wait(timeout=30)
    if transport == "uds":
        assert not Path(path).exists()


def test_a_server_whose_pytorch_has_no_lapack_refuses_a_gated_delta_model_on_cpu(
    qwen35_package,
):
    """``vllm-srun serve`` keeps serving its health while the refused model reports why, at once."""
    port = free_port()
    env = dict(
        os.environ,
        PYTHONPATH=str(SOURCE) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    without_lapack = (
        "import runpy, torch; torch._C.has_lapack = False; "
        "runpy.run_module('vllm_srun', run_name='__main__')"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            without_lapack,
            "serve",
            str(qwen35_package),
            "--device",
            "cpu",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def connect():
        return http.client.HTTPConnection("127.0.0.1", port, timeout=10)

    try:
        deadline = time.monotonic() + 120
        health = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(process.stdout.read().decode())
            try:
                status, body = request(connect(), "GET", "/health")
                health = json.loads(body)
                if health["status"] not in ("starting", "loading", "warming"):
                    break
            except OSError:
                pass
            time.sleep(0.2)
        assert status == 503 and health["status"] == "failed"
        status, body = request(connect(), "GET", "/v1/models")
        (card,) = json.loads(body)["data"]
        assert card["status"] == "failed" and not card["ready"]
        assert card["reason"].startswith("UnsupportedDeviceError: no device can serve")
        assert "built without LAPACK" in card["reason"]
        assert "CPU image" in card["reason"] and "GPU device" in card["reason"]
    finally:
        process.terminate()
        process.wait(timeout=30)
