"""Base class for vLLM-SR CLI tests.

Provides common utilities for testing CLI commands including:
- Subprocess execution helpers
- Temporary directory management
- Docker/Podman container cleanup
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
from urllib import error as urllib_error
from urllib import request as urllib_request

HTTP_STATUS_OK = 200
SUPPORTED_CONTAINER_RUNTIMES = ("docker", "podman")


class CLITestBase(unittest.TestCase):
    """Base class for vLLM-SR CLI tests."""

    # Container name used by vllm-sr
    CONTAINER_NAME = "vllm-sr-container"

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
        """Detect available container runtime (docker or podman)."""
        # Check for explicit environment variable
        env_runtime = os.getenv("CONTAINER_RUNTIME")
        normalized_runtime = (env_runtime or "").lower()
        if normalized_runtime in SUPPORTED_CONTAINER_RUNTIMES and shutil.which(
            normalized_runtime
        ):
            return normalized_runtime

        # Auto-detect
        if shutil.which("docker"):
            return "docker"
        if shutil.which("podman"):
            return "podman"
        raise RuntimeError("Neither docker nor podman found in PATH")

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
        for command in (
            [runtime, "stop", cls.CONTAINER_NAME],
            [runtime, "rm", "-f", cls.CONTAINER_NAME],
        ):
            with suppress(Exception):
                cls._run_subprocess(command, timeout=30)

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
            args: CLI arguments (e.g., ["init", "--force"])
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

        except subprocess.TimeoutExpired:
            print(f"Command timed out after {timeout}s")
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            print(f"Command failed with exception: {e}")
            return -1, "", str(e)

    def container_status(self) -> str:
        """
        Get the status of the vllm-sr container.

        Returns:
            'running', 'exited', 'paused', 'not found', or 'error'
        """
        try:
            result = self._run_subprocess(
                [
                    self.container_runtime,
                    "ps",
                    "-a",
                    "--filter",
                    f"name={self.CONTAINER_NAME}",
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

    def wait_for_container_running(self, timeout: int = 60) -> bool:
        """Wait for container to be in running state."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.container_status()
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
                    self.CONTAINER_NAME,
                ],
                timeout=10,
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Failed to get logs: {e}"

    def inspect_container(
        self, format_string: str, timeout: int = 10
    ) -> tuple[int, str, str]:
        """Inspect the vllm-sr container with the active runtime."""
        result = self._run_subprocess(
            [
                self.container_runtime,
                "inspect",
                "--format",
                format_string,
                self.CONTAINER_NAME,
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
