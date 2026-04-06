"""Base class for vLLM-SR CLI tests.

Provides common utilities for testing CLI commands including:
- Subprocess execution helpers
- Temporary directory management
- Docker container cleanup
- Logging and assertion helpers

Signed-off-by: vLLM-SR Team
"""

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from contextlib import suppress
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml

HTTP_STATUS_OK = 200


def _coerce_timeout_stream(value: str | bytes | None) -> str:
    """Normalize TimeoutExpired output/stderr values into text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


class CLITestBase(unittest.TestCase):
    """Base class for vLLM-SR CLI tests."""

    # Historical single-container runtime name still cleaned up for local test hygiene.
    CONTAINER_NAME = "vllm-sr-container"
    ROUTER_CONTAINER_NAME = "vllm-sr-router-container"
    ENVOY_CONTAINER_NAME = "vllm-sr-envoy-container"
    DASHBOARD_CONTAINER_NAME = "vllm-sr-dashboard-container"
    SIM_CONTAINER_NAME = "vllm-sr-sim-container"
    AUXILIARY_CONTAINER_NAMES = (
        "vllm-sr-grafana",
        "vllm-sr-prometheus",
        "vllm-sr-jaeger",
    )

    # Default timeout for CLI commands
    DEFAULT_TIMEOUT = 60

    # Health check timeout (for serve command)
    HEALTH_CHECK_TIMEOUT = 300

    @classmethod
    def setUpClass(cls):
        """Set up test class - ensure clean state."""
        # Detect container runtime
        cls.container_runtime = cls._detect_container_runtime()
        print(f"\n{'='*60}")
        print(f"Using container runtime: {cls.container_runtime}")
        print(f"{'='*60}")

        # Ensure no leftover container from previous tests
        cls._cleanup_container()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls._cleanup_container()

    def setUp(self):
        """Set up each test - create temp directory."""
        self.test_dir = tempfile.mkdtemp(prefix="vllm-sr-cli-test-")
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        print(f"\nTest directory: {self.test_dir}")

    def tearDown(self):
        """Clean up after each test."""
        os.chdir(self.original_dir)
        # Clean up temp directory
        try:
            shutil.rmtree(self.test_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up {self.test_dir}: {e}")

    @classmethod
    def _detect_container_runtime(cls) -> str:
        """Detect the required Docker runtime for CLI tests."""
        # Check for explicit environment variable
        env_runtime = os.getenv("CONTAINER_RUNTIME")
        normalized_runtime = (env_runtime or "").lower()
        if normalized_runtime:
            if normalized_runtime != "docker":
                raise RuntimeError(
                    f"CONTAINER_RUNTIME={normalized_runtime} is unsupported; "
                    "CLI tests require Docker"
                )
            if shutil.which("docker"):
                return "docker"
            raise RuntimeError(
                "CONTAINER_RUNTIME=docker was requested but docker is not in PATH"
            )

        # Auto-detect
        if shutil.which("docker"):
            return "docker"
        if shutil.which("podman"):
            raise RuntimeError(
                "Podman is installed but unsupported; CLI tests require Docker"
            )
        raise RuntimeError("Docker not found in PATH")

    @staticmethod
    def _run_subprocess(
        command: list[str],
        *,
        timeout: int,
        capture_output: bool = True,
        text: bool = True,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            env=env,
            cwd=cwd,
            check=False,
        )

    @classmethod
    def _cleanup_container(cls):
        """Stop and remove any existing vllm-sr container."""
        runtime = cls.container_runtime
        managed_container_names = (
            cls.CONTAINER_NAME,
            cls.ROUTER_CONTAINER_NAME,
            cls.ENVOY_CONTAINER_NAME,
            cls.DASHBOARD_CONTAINER_NAME,
            cls.SIM_CONTAINER_NAME,
            *cls.AUXILIARY_CONTAINER_NAMES,
        )
        for container_name in managed_container_names:
            for command in (
                [runtime, "stop", container_name],
                [runtime, "rm", "-f", container_name],
            ):
                with suppress(Exception):
                    cls._run_subprocess(command, timeout=30)

    def _explicit_container_status(self, container_name: str) -> str:
        """Get the status of one managed container."""
        try:
            result = self._run_subprocess(
                [
                    self.container_runtime,
                    "ps",
                    "-a",
                    "--filter",
                    f"name={container_name}",
                    "--format",
                    "{{.Status}}",
                ],
                timeout=10,
            )
            status = result.stdout.strip()
            if not status:
                return "not found"
            if "Up" in status:
                return "running"
            if "Exited" in status:
                return "exited"
            if "Paused" in status:
                return "paused"
            return "unknown"
        except Exception as e:
            print(f"Failed to get container status: {e}")
            return "error"

    def _runtime_container_names(self) -> tuple[str, ...]:
        """Return managed runtime containers in inspect/priority order."""
        return (
            self.ROUTER_CONTAINER_NAME,
            self.DASHBOARD_CONTAINER_NAME,
            self.ENVOY_CONTAINER_NAME,
        )

    def resolve_runtime_inspect_container_name(self) -> str:
        """Pick the best runtime container for inspect/log assertions."""
        for container_name in self._runtime_container_names():
            if self._explicit_container_status(container_name) != "not found":
                return container_name
        return self.ROUTER_CONTAINER_NAME

    def run_cli(
        self,
        args: list[str],
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        cwd: str | None = None,
    ) -> tuple[int, str, str]:
        """
        Run a vllm-sr CLI command.

        Args:
            args: CLI arguments (e.g., ["serve", "--config", "config.yaml"])
            timeout: Command timeout in seconds
            env: Additional environment variables
            capture_output: Whether to capture stdout/stderr
            cwd: Working directory for command

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT

        # Build command
        cmd = ["vllm-sr", *args]

        # Merge environment
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        print(f"\nRunning: {' '.join(cmd)}")

        try:
            result = self._run_subprocess(
                cmd,
                capture_output=capture_output,
                timeout=timeout,
                env=full_env,
                cwd=cwd or self.test_dir,
            )
            stdout = result.stdout if capture_output else ""
            stderr = result.stderr if capture_output else ""

            if result.returncode != 0:
                print(f"Command failed with code {result.returncode}")
                if stderr:
                    print(f"STDERR: {stderr[:500]}")
            else:
                print("Command succeeded")

            return result.returncode, stdout, stderr

        except subprocess.TimeoutExpired as exc:
            print(f"Command timed out after {timeout}s")
            stdout = _coerce_timeout_stream(
                getattr(exc, "stdout", None) or getattr(exc, "output", None)
            )
            stderr = _coerce_timeout_stream(getattr(exc, "stderr", None))
            timeout_message = f"Command timed out after {timeout} seconds"
            if stderr:
                stderr = f"{stderr.rstrip()}\n{timeout_message}"
            else:
                stderr = timeout_message
            return -1, stdout, stderr
        except Exception as e:
            print(f"Command failed with exception: {e}")
            return -1, "", str(e)

    def write_minimal_canonical_config(
        self,
        *,
        port: int = 8888,
        model_name: str = "test-model",
        endpoint: str = "host.docker.internal:8000",
    ) -> str:
        """Write a minimal runnable canonical v0.3 config into the temp workspace."""
        config_path = Path(self.test_dir) / "config.yaml"
        config = {
            "version": "v0.3",
            "listeners": [
                {
                    "name": "test-listener",
                    "address": "0.0.0.0",
                    "port": port,
                    "timeout": "60s",
                }
            ],
            "providers": {
                "defaults": {
                    "default_model": model_name,
                    "default_reasoning_effort": "medium",
                },
                "models": [
                    {
                        "name": model_name,
                        "provider_model_id": model_name,
                        "backend_refs": [
                            {
                                "name": "primary",
                                "weight": 100,
                                "endpoint": endpoint,
                                "protocol": "http",
                            }
                        ],
                    }
                ],
            },
            "routing": {
                "modelCards": [{"name": model_name}],
                "decisions": [
                    {
                        "name": "default-route",
                        "description": "Default route for CLI test coverage",
                        "priority": 100,
                        "rules": {"operator": "AND", "conditions": []},
                        "modelRefs": [{"model": model_name, "use_reasoning": False}],
                    }
                ],
            },
            "global": {
                "stores": {
                    "semantic_cache": {
                        "enabled": False,
                    }
                },
                "model_catalog": {
                    "embeddings": {
                        "semantic": {
                            "mmbert_model_path": "",
                            "qwen3_model_path": "",
                            "gemma_model_path": "",
                            "bert_model_path": "",
                            "multimodal_model_path": "",
                        }
                    },
                    "modules": {
                        "prompt_guard": {
                            "enabled": False,
                            "model_ref": "",
                            "model_id": "",
                            "jailbreak_mapping_path": "",
                            "use_mmbert_32k": False,
                        },
                        "classifier": {
                            "domain": {
                                "model_ref": "",
                                "model_id": "",
                                "category_mapping_path": "",
                                "use_mmbert_32k": False,
                            },
                            "pii": {
                                "model_ref": "",
                                "model_id": "",
                                "pii_mapping_path": "",
                                "use_mmbert_32k": False,
                            },
                        },
                        "feedback_detector": {
                            "enabled": False,
                            "model_ref": "",
                            "model_id": "",
                            "use_mmbert_32k": False,
                        },
                    },
                },
            },
        }
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        return str(config_path)

    def container_status(self, container_name: str | None = None) -> str:
        """
        Get the status of a managed container.

        Returns:
            'running', 'exited', 'paused', 'not found', or 'error'
        """
        if container_name is not None:
            return self._explicit_container_status(container_name)

        statuses = {
            name: self._explicit_container_status(name)
            for name in self._runtime_container_names()
        }
        runtime_statuses = [
            statuses[self.ROUTER_CONTAINER_NAME],
            statuses[self.DASHBOARD_CONTAINER_NAME],
            statuses[self.ENVOY_CONTAINER_NAME],
        ]
        if any(status == "running" for status in runtime_statuses):
            return "running"
        if any(status == "exited" for status in runtime_statuses):
            return "exited"
        if any(status == "paused" for status in runtime_statuses):
            return "paused"
        if any(status == "error" for status in runtime_statuses):
            return "error"
        if any(status != "not found" for status in runtime_statuses):
            return "unknown"
        return "not found"

    def wait_for_container_running(
        self, timeout: int = 60, container_name: str | None = None
    ) -> bool:
        """Wait for container to be in running state."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.container_status(container_name=container_name)
            if status == "running":
                return True
            if status == "exited":
                print("Container exited unexpectedly")
                return False
            time.sleep(2)
        return False

    def wait_for_health(self, port: int = 8080, timeout: int | None = None) -> bool:
        """
        Wait for the router health endpoint to respond.

        Args:
            port: Port to check (default: 8080 for router API)
            timeout: Timeout in seconds

        Returns:
            True if healthy, False otherwise
        """
        if timeout is None:
            timeout = self.HEALTH_CHECK_TIMEOUT

        url = f"http://localhost:{port}/health"
        start = time.time()

        while time.time() - start < timeout:
            try:
                with urllib_request.urlopen(url, timeout=5) as response:
                    if response.status == HTTP_STATUS_OK:
                        print(f"✓ Health check passed on port {port}")
                        return True
            except (urllib_error.URLError, urllib_error.HTTPError, OSError):
                pass
            time.sleep(2)

        print(f"✗ Health check failed after {timeout}s")
        return False

    def container_logs(self, tail: int = 50) -> str:
        """Get container logs."""
        try:
            result = self._run_subprocess(
                [
                    self.container_runtime,
                    "logs",
                    "--tail",
                    str(tail),
                    self.resolve_runtime_inspect_container_name(),
                ],
                timeout=10,
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Failed to get logs: {e}"

    def inspect_container(
        self,
        format_string: str,
        timeout: int = 10,
        container_name: str | None = None,
    ) -> tuple[int, str, str]:
        """Inspect a managed container with the active runtime."""
        container_name = container_name or self.resolve_runtime_inspect_container_name()
        result = self._run_subprocess(
            [
                self.container_runtime,
                "inspect",
                "--format",
                format_string,
                container_name,
            ],
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    def image_exists(self, image_name: str) -> bool:
        """Check if a container image exists locally."""
        try:
            result = self._run_subprocess(
                [self.container_runtime, "images", "-q", image_name],
                timeout=10,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def container_runtime_accessible(self) -> bool:
        """Return True when the configured container runtime daemon is reachable."""
        try:
            result = self._run_subprocess(
                [self.container_runtime, "info"],
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def print_test_header(self, name: str, description: str | None = None):
        """Print a formatted test header."""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        if description:
            print(f"Description: {description}")
        print(f"{'='*60}")

    def print_test_result(self, passed: bool, message: str = ""):
        """Print test result with pass/fail indicator."""
        result = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\nResult: {result}")
        if message:
            print(f"Details: {message}")

    def assert_file_exists(self, path: str, msg: str | None = None):
        """Assert that a file exists."""
        if not os.path.exists(path):
            self.fail(msg or f"File does not exist: {path}")

    def assert_file_contains(
        self, path: str, content: str, msg: str | None = None
    ) -> None:
        """Assert that a file contains specific content."""
        with open(path, encoding="utf-8") as f:
            file_content = f.read()
        if content not in file_content:
            self.fail(msg or f"File {path} does not contain: {content}")

    def assert_dir_exists(self, path: str, msg: str | None = None) -> None:
        """Assert that a directory exists."""
        if not os.path.isdir(path):
            self.fail(msg or f"Directory does not exist: {path}")
