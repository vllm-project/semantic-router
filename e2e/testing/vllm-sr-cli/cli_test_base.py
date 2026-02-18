"""Base class for vLLM-SR CLI tests.

Provides common utilities for testing CLI commands including:
- Subprocess execution helpers
- Temporary directory management
- Docker/Podman container cleanup
- Logging and assertion helpers

Signed-off-by: vLLM-SR Team
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        if env_runtime and env_runtime.lower() in ["docker", "podman"]:
            if shutil.which(env_runtime.lower()):
                return env_runtime.lower()

        # Auto-detect
        if shutil.which("docker"):
            return "docker"
        elif shutil.which("podman"):
            return "podman"
        else:
            raise RuntimeError("Neither docker nor podman found in PATH")

    @classmethod
    def _cleanup_container(cls):
        """Stop and remove any existing vllm-sr container."""
        runtime = cls.container_runtime
        try:
            # Stop container if running
            subprocess.run(
                [runtime, "stop", cls.CONTAINER_NAME],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass

        try:
            # Remove container
            subprocess.run(
                [runtime, "rm", "-f", cls.CONTAINER_NAME],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass

    def run_cli(
        self,
        args: List[str],
        timeout: int = None,
        env: Dict[str, str] = None,
        capture_output: bool = True,
        cwd: str = None,
    ) -> Tuple[int, str, str]:
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
        cmd = ["vllm-sr"] + args

        # Merge environment
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        print(f"\nRunning: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
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
                print(f"Command succeeded")

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
            result = subprocess.run(
                [
                    self.container_runtime,
                    "ps",
                    "-a",
                    "--filter",
                    f"name={self.CONTAINER_NAME}",
                    "--format",
                    "{{.Status}}",
                ],
                capture_output=True,
                text=True,
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

    def wait_for_health(self, port: int = 8080, timeout: int = None) -> bool:
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

        import urllib.request
        import urllib.error

        url = f"http://localhost:{port}/health"
        start = time.time()

        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        print(f"✓ Health check passed on port {port}")
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                pass
            time.sleep(2)

        print(f"✗ Health check failed after {timeout}s")
        return False

    def container_logs(self, tail: int = 50) -> str:
        """Get container logs."""
        try:
            result = subprocess.run(
                [
                    self.container_runtime,
                    "logs",
                    "--tail",
                    str(tail),
                    self.CONTAINER_NAME,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Failed to get logs: {e}"

    def image_exists(self, image_name: str) -> bool:
        """Check if a container image exists locally."""
        try:
            result = subprocess.run(
                [self.container_runtime, "images", "-q", image_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def print_test_header(self, name: str, description: str = None):
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

    def assertFileExists(self, path: str, msg: str = None):
        """Assert that a file exists."""
        if not os.path.exists(path):
            self.fail(msg or f"File does not exist: {path}")

    def assertFileContains(self, path: str, content: str, msg: str = None):
        """Assert that a file contains specific content."""
        with open(path, "r") as f:
            file_content = f.read()
        if content not in file_content:
            self.fail(msg or f"File {path} does not contain: {content}")

    def assertDirExists(self, path: str, msg: str = None):
        """Assert that a directory exists."""
        if not os.path.isdir(path):
            self.fail(msg or f"Directory does not exist: {path}")
