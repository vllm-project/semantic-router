"""Startup-contract checks for the model-selection service: loopback binding and a live health endpoint."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != "linux", reason="inspects Linux listening sockets")
def test_default_service_binds_loopback_and_serves_health(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {**os.environ, "ML_SERVICE_DATA_DIR": str(tmp_path)}
    env.pop("ML_SERVICE_HOST", None)
    log_path = tmp_path / "service.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(SERVICE_DIR / "server.py"), "--port", str(port)],
            cwd=SERVICE_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text()
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/health", timeout=1
                    ) as response:
                        result = json.load(response)
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise AssertionError(log_path.read_text())
            assert result["status"] == "ok"
            # Inspect this process's actual listening socket; no remote probes.
            listeners = [
                line.split()[1]
                for line in Path(f"/proc/{process.pid}/net/tcp")
                .read_text()
                .splitlines()[1:]
                if line.split()[3] == "0A" and line.split()[1].endswith(f":{port:04X}")
            ]
            assert listeners == [f"0100007F:{port:04X}"]
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
