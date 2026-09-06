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
from cli.runtime_stack import DEFAULT_STACK_NAME, resolve_runtime_stack

HTTP_STATUS_OK = 200
AGENT_SMOKE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "config.agent-smoke.cpu.yaml"
)


def stack_scoped_test_container_name(stack_name: str, base_name: str) -> str:
    """Keep test-only containers inside the selected runtime stack namespace."""

    if stack_name == DEFAULT_STACK_NAME:
        return base_name
    return f"{stack_name}-{base_name}"


# Services whose store backends the CLI provisions locally. They are written
# out in full rather than left to the canonical defaults: a test that rides on
# a default is silently retargeted the day that default changes, and these two
# exist to name Redis and Postgres specifically.
MANAGED_STORAGE_SERVICES: dict[str, dict[str, object]] = {
    "response_api": {"enabled": True, "store_backend": "redis"},
    "router_replay": {"enabled": True, "store_backend": "postgres"},
}


def _coerce_timeout_stream(value: str | bytes | None) -> str:
    """Normalize TimeoutExpired output/stderr values into text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _api_only_global_config() -> dict[str, object]:
    """Reuse the canonical smoke global block for API-only integration tests."""
    smoke_config = yaml.safe_load(AGENT_SMOKE_CONFIG_PATH.read_text(encoding="utf-8"))
    global_config = smoke_config.get("global")
    if not isinstance(global_config, dict):
        raise AssertionError(f"{AGENT_SMOKE_CONFIG_PATH} must define a global mapping")
    return global_config


class CLITestBase(unittest.TestCase):
    """Base class for vLLM-SR CLI tests."""

    # Historical single-container runtime name still cleaned up for local test hygiene.
    CONTAINER_NAME = "vllm-sr-container"
    ROUTER_CONTAINER_NAME = "vllm-sr-router-container"
    ENVOY_CONTAINER_NAME = "vllm-sr-envoy-container"
    DASHBOARD_CONTAINER_NAME = "vllm-sr-dashboard-container"
    REDIS_CONTAINER_NAME = "vllm-sr-redis"
    POSTGRES_CONTAINER_NAME = "vllm-sr-postgres"
    MILVUS_CONTAINER_NAME = "vllm-sr-milvus"
    NETWORK_NAME = "vllm-sr-network"
    DATA_NETWORK_NAME = "vllm-sr-data-network"
    AUXILIARY_CONTAINER_NAMES = (
        "vllm-sr-grafana",
        "vllm-sr-prometheus",
        "vllm-sr-jaeger",
    )

    # One-shot container a test drives to probe a network from the inside.
    # Named rather than anonymous so a probe that outlives its test is still
    # removed by the class-level cleanup.
    PROBE_CONTAINER_NAME = "vllm-sr-cli-test-probe"

    # Default timeout for CLI commands
    DEFAULT_TIMEOUT = 60

    # Health check timeout (for serve command)
    HEALTH_CHECK_TIMEOUT = 300

    @classmethod
    def setUpClass(cls):
        """Set up test class - ensure clean state."""
        cls.runtime_stack = resolve_runtime_stack()
        stack_name = cls.runtime_stack.stack_name
        cls.CONTAINER_NAME = (
            "vllm-sr-container"
            if stack_name == DEFAULT_STACK_NAME
            else f"{stack_name}-vllm-sr-container"
        )
        cls.ROUTER_CONTAINER_NAME = cls.runtime_stack.router_container_name
        cls.ENVOY_CONTAINER_NAME = cls.runtime_stack.envoy_container_name
        cls.DASHBOARD_CONTAINER_NAME = cls.runtime_stack.dashboard_container_name
        cls.REDIS_CONTAINER_NAME = cls.runtime_stack.redis_container_name
        cls.POSTGRES_CONTAINER_NAME = cls.runtime_stack.postgres_container_name
        cls.MILVUS_CONTAINER_NAME = cls.runtime_stack.milvus_container_name
        cls.NETWORK_NAME = cls.runtime_stack.network_name
        cls.DATA_NETWORK_NAME = cls.runtime_stack.data_network_name
        cls.PROBE_CONTAINER_NAME = stack_scoped_test_container_name(
            stack_name, "vllm-sr-cli-test-probe"
        )
        cls.AUXILIARY_CONTAINER_NAMES = (
            cls.runtime_stack.grafana_container_name,
            cls.runtime_stack.prometheus_container_name,
            cls.runtime_stack.jaeger_container_name,
            cls.runtime_stack.redis_container_name,
            cls.runtime_stack.postgres_container_name,
            cls.runtime_stack.milvus_container_name,
        )

        # Detect container runtime
        cls.container_runtime = cls._detect_container_runtime()
        print(f"\n{'=' * 60}")
        print(f"Using container runtime: {cls.container_runtime}")
        print(f"{'=' * 60}")

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
        """Detect the container runtime used to drive CLI tests.

        Both Docker and Podman are supported (matching the CLI). The selection
        order is: explicit ``CONTAINER_RUNTIME`` env var → docker on PATH →
        podman on PATH.
        """
        env_runtime = os.getenv("CONTAINER_RUNTIME")
        normalized_runtime = (env_runtime or "").lower()
        if normalized_runtime:
            if normalized_runtime not in ("docker", "podman"):
                raise RuntimeError(
                    f"CONTAINER_RUNTIME={normalized_runtime} is unsupported; "
                    "CLI tests support docker or podman"
                )
            if shutil.which(normalized_runtime):
                return normalized_runtime
            raise RuntimeError(
                f"CONTAINER_RUNTIME={normalized_runtime} was requested but "
                f"{normalized_runtime} is not in PATH"
            )

        if shutil.which("docker"):
            return "docker"
        if shutil.which("podman"):
            return "podman"
        if os.getenv("RUN_INTEGRATION_TESTS", "").lower() != "true":
            # Unit-only suites mock container commands at their behavioral
            # seams. Keep a deterministic command name so those tests can run
            # inside the precommit image, which intentionally has no runtime
            # client or daemon.
            return "docker"
        raise RuntimeError("Neither docker nor podman was found in PATH")

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
            cls.PROBE_CONTAINER_NAME,
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
                    "inspect",
                    "--format",
                    "{{.State.Status}}",
                    container_name,
                ],
                timeout=10,
            )
            if result.returncode != 0:
                return "not found"
            status = result.stdout.strip().lower()
            if status in {"running", "created", "exited", "paused"}:
                return status
            return status or "unknown"
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
        base_url: str | None = None,
        provider: str | None = None,
        api_key_env: str | None = None,
        api_only: bool = False,
        managed_storage: bool = False,
    ) -> str:
        """Write a minimal runnable canonical v0.3 config into the temp workspace.

        *managed_storage* asks for the service backends this CLI provisions
        locally, which is what makes `serve` start Redis and Postgres.
        """
        config_path = Path(self.test_dir) / "config.yaml"
        backend_ref: dict[str, object] = {
            "name": "primary",
            "weight": 100,
        }
        if base_url is not None:
            backend_ref["base_url"] = base_url
        else:
            backend_ref["endpoint"] = endpoint
            backend_ref["protocol"] = "http"
        if provider is not None:
            backend_ref["provider"] = provider
        if api_key_env is not None:
            backend_ref["api_key_env"] = api_key_env

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
                        "backend_refs": [backend_ref],
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
        }
        if api_only:
            config["global"] = _api_only_global_config()
        if managed_storage:
            global_config = config.get("global")
            if not isinstance(global_config, dict):
                global_config = {}
                config["global"] = global_config
            global_config["services"] = {
                service_key: dict(service_config)
                for service_key, service_config in MANAGED_STORAGE_SERVICES.items()
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

    def container_networks(self, container_name: str) -> set[str]:
        """Return the networks *container_name* is currently attached to."""
        result = self._run_subprocess(
            [
                self.container_runtime,
                "inspect",
                "--format",
                "{{range $name, $_ := .NetworkSettings.Networks}}{{$name}} {{end}}",
                container_name,
            ],
            timeout=10,
        )
        if result.returncode != 0:
            self.fail(
                f"Could not inspect the networks of {container_name}: "
                f"{result.stderr.strip() or f'exit code {result.returncode}'}"
            )
        return set(result.stdout.split())

    def run_network_probe(
        self,
        *,
        network_name: str,
        image: str,
        shell_command: str,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        """Run *shell_command* from a throwaway container on *network_name*.

        The probe has to run from inside a network, because that is the only
        vantage point from which "can this workload reach that store" has an
        answer: the host reaches a published port either way.
        """
        with suppress(Exception):
            self._run_subprocess(
                [self.container_runtime, "rm", "-f", self.PROBE_CONTAINER_NAME],
                timeout=30,
            )
        return self._run_subprocess(
            [
                self.container_runtime,
                "run",
                "--rm",
                "--name",
                self.PROBE_CONTAINER_NAME,
                "--network",
                network_name,
                image,
                "sh",
                "-c",
                shell_command,
            ],
            timeout=timeout,
        )

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
        print(f"\n{'=' * 60}")
        print(f"TEST: {name}")
        if description:
            print(f"Description: {description}")
        print(f"{'=' * 60}")

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
