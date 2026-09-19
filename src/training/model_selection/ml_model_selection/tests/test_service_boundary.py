"""Benign startup and deployment-contract checks for the private sidecar."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[5]
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


def test_shipped_sidecar_probes_stay_inside_the_pod():
    manifests = [
        ROOT / "deploy/kubernetes/observability/dashboard/deployment.yaml",
        ROOT / "deploy/openshift/dashboard/dashboard-deployment.yaml",
    ]
    for manifest in manifests:
        deployment = next(
            item
            for item in yaml.safe_load_all(manifest.read_text())
            if item and item.get("kind") == "Deployment"
        )
        containers = deployment["spec"]["template"]["spec"]["containers"]
        service = next(item for item in containers if item["name"] == "ml-service")
        env = {item["name"]: item.get("value") for item in service["env"]}
        assert env["ML_SERVICE_HOST"] == "127.0.0.1"
        assert not service.get("ports")
        for probe_name in ("readinessProbe", "livenessProbe"):
            probe = service[probe_name]
            assert "httpGet" not in probe
            assert probe["exec"]["command"][-1] == "http://127.0.0.1:8686/api/health"
            command = probe["exec"]["command"]
            request_timeout = int(command[command.index("--max-time") + 1])
            assert probe["timeoutSeconds"] > request_timeout
