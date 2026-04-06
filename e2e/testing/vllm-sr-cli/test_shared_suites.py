#!/usr/bin/env python3
"""Shared docker/k8s integration suites for the default dev regression contract."""

import json
import os
import socket
import subprocess
import time
import unittest
from contextlib import contextmanager, suppress
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from cli_test_base import CLITestBase

DEFAULT_NAMESPACE = "vllm-semantic-router-system"
DEFAULT_RELEASE_NAME = "semantic-router"
DEFAULT_KIND_CLUSTER = os.environ.get("VLLM_SR_KIND_CLUSTER_NAME", "vllm-sr")
DEFAULT_KIND_CONTEXT = f"kind-{DEFAULT_KIND_CLUSTER}"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_READY_TIMEOUT = int(os.environ.get("VLLM_SR_RUNTIME_READY_TIMEOUT", "600"))
K8S_RUNTIME_START_TIMEOUT = int(
    os.environ.get("VLLM_SR_K8S_RUNTIME_START_TIMEOUT", "1800")
)
LLM_KATAN_CONTAINER = "vllm-sr-shared-llm-katan"
LLM_KATAN_IMAGE = os.environ.get(
    "VLLM_SR_LLM_KATAN_IMAGE",
    "ghcr.io/vllm-project/semantic-router/llm-katan:latest",
)
LLM_KATAN_SERVED_MODEL = os.environ.get(
    "VLLM_SR_LLM_KATAN_SERVED_MODEL",
    "test-model",
)
LLM_KATAN_MODEL = os.environ.get("VLLM_SR_LLM_KATAN_MODEL", LLM_KATAN_SERVED_MODEL)
LLM_KATAN_BACKEND = os.environ.get("VLLM_SR_LLM_KATAN_BACKEND", "echo")


class SharedRuntimeIntegrationBase(CLITestBase):
    """Base helpers for the shared docker/k8s integration suites."""

    ROUTER_LOCAL_PORT = 8888
    DASHBOARD_LOCAL_PORT = 8700

    def setUp(self):
        super().setUp()
        self._port_forwards: list[subprocess.Popen] = []
        self._serve_process: subprocess.Popen | None = None
        self._llm_katan_host_port: int | None = None

    def tearDown(self):
        self._stop_runtime()
        self._stop_llm_katan_fixture()
        self._stop_port_forwards()
        super().tearDown()

    def _integration_target(self) -> str:
        return (os.environ.get("VLLM_SR_TEST_TARGET") or "docker").strip().lower()

    def _shared_suite_selected(self, suite_name: str) -> bool:
        requested = (os.environ.get("VLLM_SR_SHARED_SUITE") or "").strip().lower()
        return not requested or requested == suite_name

    def _requires_dashboard(self) -> bool:
        return False

    def _require_integration_suite(self, suite_name: str) -> None:
        if os.environ.get("RUN_INTEGRATION_TESTS", "").lower() != "true":
            self.skipTest("Integration tests disabled. Set RUN_INTEGRATION_TESTS=true.")
        if not self._shared_suite_selected(suite_name):
            self.skipTest(f"Shared suite {suite_name} not selected for this run.")

    def _k8s_context(self) -> str:
        return os.environ.get("VLLM_SR_KIND_CONTEXT", DEFAULT_KIND_CONTEXT)

    def _run_checked(self, command: list[str], *, timeout: int = 180) -> str:
        result = self._run_subprocess(command, timeout=timeout)
        if result.returncode != 0:
            raise AssertionError(
                f"command failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result.stdout

    def _available_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def _wait_for_http_json(
        self,
        url: str,
        *,
        timeout: int = 120,
        expected_status: int = 200,
    ) -> dict:
        start = time.time()
        last_error: Exception | None = None
        while time.time() - start < timeout:
            try:
                with urllib_request.urlopen(url, timeout=5) as response:
                    if response.status != expected_status:
                        raise AssertionError(
                            f"expected {expected_status}, got {response.status}"
                        )
                    return json.loads(response.read().decode("utf-8"))
            except (
                urllib_error.URLError,
                urllib_error.HTTPError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                time.sleep(2)
        raise AssertionError(f"timed out waiting for {url}: {last_error}")

    def _json_request(self, url: str, payload: dict) -> tuple[int, dict]:
        request = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def _start_llm_katan_fixture(self) -> None:
        target = self._integration_target()
        llm_katan_args = [
            "llm-katan",
            "--model",
            LLM_KATAN_MODEL,
            "--backend",
            LLM_KATAN_BACKEND,
            "--served-model-name",
            LLM_KATAN_SERVED_MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        if target == "docker":
            self._llm_katan_host_port = self._available_port()
            self._run_subprocess(
                [self.container_runtime, "rm", "-f", LLM_KATAN_CONTAINER],
                timeout=30,
            )
            self._run_checked(
                [
                    self.container_runtime,
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    LLM_KATAN_CONTAINER,
                    "-p",
                    f"{self._llm_katan_host_port}:8000",
                    LLM_KATAN_IMAGE,
                    *llm_katan_args,
                ],
                timeout=60,
            )
            self._wait_for_http_json(
                f"http://localhost:{self._llm_katan_host_port}/v1/models", timeout=120
            )
            return

        self._run_checked(
            [
                "kind",
                "load",
                "docker-image",
                LLM_KATAN_IMAGE,
                "--name",
                DEFAULT_KIND_CLUSTER,
            ],
            timeout=120,
        )
        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-katan
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-katan
  template:
    metadata:
      labels:
        app: llm-katan
    spec:
      containers:
        - name: llm-katan
          image: {LLM_KATAN_IMAGE}
          imagePullPolicy: Never
          command: ["llm-katan"]
          args:
            - "--model"
            - "{LLM_KATAN_MODEL}"
            - "--backend"
            - "{LLM_KATAN_BACKEND}"
            - "--served-model-name"
            - "{LLM_KATAN_SERVED_MODEL}"
            - "--host"
            - "0.0.0.0"
            - "--port"
            - "8000"
          ports:
            - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: llm-katan
  namespace: default
spec:
  selector:
    app: llm-katan
  ports:
    - port: 8000
      targetPort: 8000
"""
        apply = subprocess.run(
            [
                "kubectl",
                "--context",
                self._k8s_context(),
                "apply",
                "-f",
                "-",
            ],
            input=manifest,
            text=True,
            capture_output=True,
            check=False,
        )
        if apply.returncode != 0:
            raise AssertionError(
                f"failed to apply llm-katan manifest:\nSTDOUT:\n{apply.stdout}\nSTDERR:\n{apply.stderr}"
            )
        self._run_checked(
            [
                "kubectl",
                "--context",
                self._k8s_context(),
                "-n",
                "default",
                "rollout",
                "status",
                "deployment/llm-katan",
                "--timeout=180s",
            ],
            timeout=240,
        )
        llm_katan_local_port = self._available_port()
        self._start_port_forward(
            "service/llm-katan",
            8000,
            llm_katan_local_port,
            namespace="default",
        )
        self._wait_for_http_json(
            f"http://localhost:{llm_katan_local_port}/v1/models",
            timeout=120,
        )

    def _stop_llm_katan_fixture(self) -> None:
        target = self._integration_target()
        if target == "docker":
            self._run_subprocess(
                [self.container_runtime, "rm", "-f", LLM_KATAN_CONTAINER],
                timeout=30,
            )
            return

        with suppress(Exception):
            self._run_subprocess(
                [
                    "kubectl",
                    "--context",
                    self._k8s_context(),
                    "-n",
                    "default",
                    "delete",
                    "deployment/llm-katan",
                    "--ignore-not-found",
                ],
                timeout=60,
            )
        with suppress(Exception):
            self._run_subprocess(
                [
                    "kubectl",
                    "--context",
                    self._k8s_context(),
                    "-n",
                    "default",
                    "delete",
                    "service",
                    "llm-katan",
                    "--ignore-not-found",
                ],
                timeout=60,
            )

    def _backend_endpoint(self) -> str:
        if self._integration_target() == "docker":
            assert self._llm_katan_host_port is not None
            return f"host.docker.internal:{self._llm_katan_host_port}"
        return "llm-katan.default.svc.cluster.local:8000"

    def _router_base_url(self) -> str:
        return f"http://localhost:{self.ROUTER_LOCAL_PORT}"

    def _dashboard_base_url(self) -> str:
        return f"http://localhost:{self.DASHBOARD_LOCAL_PORT}"

    def _wait_for_docker_runtime(self) -> None:
        time.sleep(5)
        if self._serve_process and self._serve_process.poll() is not None:
            stdout, stderr = self._serve_process.communicate(timeout=10)
            if not self.wait_for_container_running(timeout=120):
                raise AssertionError(
                    f"docker runtime exited before becoming healthy:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )
        else:
            if not self.wait_for_container_running(timeout=120):
                if self._serve_process and self._serve_process.poll() is None:
                    self._serve_process.terminate()
                    with suppress(subprocess.TimeoutExpired):
                        self._serve_process.wait(timeout=10)
                stdout = ""
                stderr = ""
                if self._serve_process:
                    stdout, stderr = self._serve_process.communicate(timeout=10)
                raise AssertionError(
                    f"docker runtime container did not start:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )

    def _wait_for_runtime_contracts(self) -> None:
        self._wait_for_http_json(
            f"{self._router_base_url()}/v1/models",
            timeout=RUNTIME_READY_TIMEOUT,
        )
        if self._requires_dashboard():
            self._wait_for_http_json(
                f"{self._dashboard_base_url()}/healthz",
                timeout=RUNTIME_READY_TIMEOUT,
            )

    def _start_port_forward(
        self,
        resource_ref: str,
        remote_port: int,
        local_port: int,
        *,
        namespace: str = DEFAULT_NAMESPACE,
    ):
        process = subprocess.Popen(
            [
                "kubectl",
                "--context",
                self._k8s_context(),
                "-n",
                namespace,
                "port-forward",
                resource_ref,
                f"{local_port}:{remote_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._port_forwards.append(process)
        start = time.time()
        while time.time() - start < 30:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=5)
                raise AssertionError(
                    f"port-forward failed for {resource_ref}:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", local_port)) == 0:
                    return
            time.sleep(1)
        raise AssertionError(f"timed out waiting for port-forward on {resource_ref}")

    def _start_runtime(self) -> None:
        target = self._integration_target()
        config_path = self.write_minimal_canonical_config(
            port=self.ROUTER_LOCAL_PORT,
            endpoint=self._backend_endpoint(),
        )
        if target == "docker":
            serve_args = [
                "vllm-sr",
                "serve",
                "--config",
                config_path,
                "--image-pull-policy",
                "ifnotpresent",
            ]
            if not self._requires_dashboard():
                serve_args.append("--minimal")
            self._serve_process = subprocess.Popen(
                serve_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.test_dir,
                env=os.environ.copy(),
            )
            self._wait_for_docker_runtime()
            self._wait_for_runtime_contracts()
            return

        serve_args = [
            "serve",
            "--config",
            config_path,
            "--target",
            "k8s",
            "--disable-observability",
            "--image-pull-policy",
            "never",
        ]
        if self._requires_dashboard():
            serve_args.extend(["--profile", "dev"])

        return_code, stdout, stderr = self.run_cli(
            serve_args,
            timeout=K8S_RUNTIME_START_TIMEOUT,
            cwd=str(REPO_ROOT),
        )
        if return_code != 0:
            raise AssertionError(
                f"k8s runtime startup failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        self._start_port_forward(
            f"service/{DEFAULT_RELEASE_NAME}-envoy",
            self.ROUTER_LOCAL_PORT,
            self.ROUTER_LOCAL_PORT,
        )
        if self._requires_dashboard():
            self._start_port_forward(
                f"service/{DEFAULT_RELEASE_NAME}-dashboard",
                8700,
                self.DASHBOARD_LOCAL_PORT,
            )
        self._wait_for_runtime_contracts()

    def _stop_runtime(self) -> None:
        if self._integration_target() == "docker":
            self.run_cli(["stop"], timeout=60)
            if self._serve_process and self._serve_process.poll() is None:
                self._serve_process.terminate()
                with suppress(subprocess.TimeoutExpired):
                    self._serve_process.wait(timeout=10)
            self._cleanup_container()
            return

        self.run_cli(["stop", "--target", "k8s"], timeout=240, cwd=str(REPO_ROOT))

    def _stop_port_forwards(self) -> None:
        for process in reversed(self._port_forwards):
            if process.poll() is None:
                process.terminate()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
        self._port_forwards = []

    @contextmanager
    def _running_shared_runtime(self):
        if self._integration_target() == "docker":
            self._start_llm_katan_fixture()
            self._start_runtime()
        else:
            self._start_runtime()
            self._start_llm_katan_fixture()
        try:
            yield
        finally:
            self._stop_runtime()
            self._stop_llm_katan_fixture()
            self._stop_port_forwards()


class TestSharedKubernetesSuite(SharedRuntimeIntegrationBase):
    """Core routing contract shared across docker and k8s."""

    def test_chat_completions_and_models_contract(self):
        self._require_integration_suite("kubernetes")

        with self._running_shared_runtime():
            models = self._wait_for_http_json(
                f"{self._router_base_url()}/v1/models", timeout=60
            )
            self.assertTrue(
                models.get("data"), "router should expose at least one model"
            )

            status_code, body = self._json_request(
                f"{self._router_base_url()}/v1/chat/completions",
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Say hello"}],
                },
            )
            self.assertEqual(status_code, 200)
            self.assertTrue(
                body.get("choices"), "chat completions should return choices"
            )
            self.assertIn("id", body)


class TestSharedDashboardSuite(SharedRuntimeIntegrationBase):
    """Dashboard API contract shared across docker and k8s."""

    def _requires_dashboard(self) -> bool:
        return True

    def test_dashboard_health_status_and_config_contract(self):
        self._require_integration_suite("dashboard")

        with self._running_shared_runtime():
            health = self._wait_for_http_json(
                f"{self._dashboard_base_url()}/healthz", timeout=60
            )
            self.assertEqual(health.get("status"), "healthy")

            setup_state = self._wait_for_http_json(
                f"{self._dashboard_base_url()}/api/setup/state", timeout=60
            )
            self.assertFalse(
                setup_state.get("setupMode"),
                "shared runtime should boot in active mode",
            )
            self.assertEqual(
                setup_state.get("listenerPort"),
                self.ROUTER_LOCAL_PORT,
            )
            self.assertTrue(setup_state.get("hasModels"))
            self.assertTrue(setup_state.get("hasDecisions"))
            self.assertTrue(setup_state.get("canActivate"))
            self.assertGreaterEqual(setup_state.get("models", 0), 1)
            self.assertGreaterEqual(setup_state.get("decisions", 0), 1)
