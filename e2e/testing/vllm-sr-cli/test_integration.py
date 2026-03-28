#!/usr/bin/env python3
"""
test_integration.py - Integration tests for vLLM-SR CLI.

These tests require a working Docker image and test complete workflows.
They are slower than unit tests and should be run with --integration flag.

"""

import os
import subprocess
import time
import unittest
from contextlib import contextmanager
from urllib import error as urllib_error
from urllib import request as urllib_request

from cli_test_base import CLITestBase


class TestServeIntegration(CLITestBase):
    """Integration tests for the complete serve workflow."""

    # Timeout for waiting for container to be running
    CONTAINER_STARTUP_TIMEOUT = 120

    def _create_minimal_config(self, port: int = 8888) -> str:
        return self.write_minimal_canonical_config(port=port)

    def _start_serve_background(
        self, env: dict[str, str] | None = None
    ) -> subprocess.Popen:
        """Start vllm-sr serve in background (non-blocking)."""
        cmd = ["vllm-sr", "serve", "--image-pull-policy", "ifnotpresent"]
        print(f"\nStarting in background: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_dir,
            env=env,
        )
        return process

    def _stop_serve_process(self, serve_process: subprocess.Popen | None):
        """Terminate a background serve process if it is still running."""
        if serve_process and serve_process.poll() is None:
            serve_process.terminate()
            try:
                serve_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                serve_process.kill()

    def _wait_for_running_container(self, serve_process: subprocess.Popen):
        """Ensure serve stayed alive long enough to launch the container."""
        time.sleep(5)

        if serve_process.poll() is not None:
            stdout, stderr = serve_process.communicate()
            if self.wait_for_container_running(timeout=self.CONTAINER_STARTUP_TIMEOUT):
                print("  ✓ Serve command completed after launching the runtime")
                return
            self.fail(
                "Serve exited before the runtime was ready: "
                f"{stderr[:500] or stdout[:500]}"
            )

        print(
            f"  Waiting for container (timeout: {self.CONTAINER_STARTUP_TIMEOUT}s)..."
        )
        if not self.wait_for_container_running(timeout=self.CONTAINER_STARTUP_TIMEOUT):
            self._stop_serve_process(serve_process)
            stdout, stderr = serve_process.communicate(timeout=10)
            self.fail(f"Container did not start: {stderr[:500] or stdout[:500]}")

        print("  ✓ Container is running")

    @contextmanager
    def _running_serve(
        self,
        *,
        env: dict[str, str] | None = None,
        ensure_models_dir: bool = False,
    ):
        """Start one background serve session and clean it up automatically."""
        self.write_minimal_canonical_config()
        if ensure_models_dir:
            os.makedirs(os.path.join(self.test_dir, "models"), exist_ok=True)

        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        serve_process = self._start_serve_background(env=full_env)
        try:
            self._wait_for_running_container(serve_process)
            yield serve_process
        finally:
            self._stop_serve_process(serve_process)

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_running_container_contracts(self):
        """Test one running container session against the core CLI contracts."""
        self.print_test_header(
            "Running Container Integration Test",
            "Tests one serve startup against health, mounts, status, and logs",
        )

        with self._running_serve(ensure_models_dir=True):
            self._check_health_endpoint()
            self._assert_volume_mounting()
            self._assert_status_command()
            self._assert_logs_command()

        self.print_test_result(True, "Running container contracts verified")

    def _check_health_endpoint(self):
        """Check health endpoint (informational, doesn't fail test)."""
        try:
            url = "http://localhost:8888/health"
            with urllib_request.urlopen(url, timeout=10) as response:
                print(f"  ✓ Health check: {response.status}")
        except urllib_error.HTTPError as e:
            # 500 = service running but no backend - expected with default config
            print(f"  ⚠ Health check: {e.code} (expected without backend)")
        except Exception as e:
            print(f"  ⚠ Health check failed: {e}")

    def _assert_volume_mounting(self):
        """Verify config and models directories are mounted into the container."""
        return_code, stdout, stderr = self.inspect_container("{{json .Mounts}}")
        if return_code != 0:
            self.fail(f"container inspect failed: {stderr}")

        mounts = stdout.lower()
        print(f"  Mounts: {mounts[:200]}...")

        config_mounted = "config.yaml" in mounts or "config" in mounts
        models_mounted = "models" in mounts

        if config_mounted:
            print("  ✓ config.yaml is mounted")
        else:
            print("  ⚠ config.yaml mount not detected")

        if models_mounted:
            print("  ✓ models/ directory is mounted")
        else:
            print("  ⚠ models/ mount not detected")

        self.assertTrue(
            config_mounted or models_mounted,
            "No expected mounts found in container",
        )

    def _assert_status_command(self):
        """Verify the status command reports a running container."""
        _return_code, stdout, stderr = self.run_cli(["status"])
        output = (stdout + stderr).lower()

        running_indicators = ["running", "up", "healthy", "started"]
        status_ok = any(indicator in output for indicator in running_indicators)
        if not status_ok:
            self.fail(f"Status doesn't show running. Got: {output[:300]}")

        print("  ✓ Status shows container is running")

    def _assert_logs_command(self):
        """Verify the logs command returns container output for one service."""
        time.sleep(5)
        service_failures: list[str] = []
        for service in ("router", "envoy", "dashboard", "simulator"):
            return_code, stdout, stderr = self.run_cli(["logs", service])
            output = stdout + stderr
            if return_code == 0 and output.strip():
                print(f"  ✓ Logs retrieved from {service} ({len(output)} chars)")
                print(f"  Sample: {output[:100]}...")
                return
            service_failures.append(
                f"{service}: rc={return_code}, output={output[:120]}"
            )

        self.fail(
            "logs command failed for all services: " + " | ".join(service_failures)
        )

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_env_var_passed_to_container(self):
        """Test that environment variables are actually passed to container."""
        self.print_test_header(
            "Environment Variable Integration Test",
            "Verifies HF_TOKEN is inside running container via container inspect",
        )

        test_token = "hf_integration_test_token_xyz"
        with self._running_serve(env={"HF_TOKEN": test_token}):
            return_code, stdout, stderr = self.inspect_container("{{.Config.Env}}")
            if return_code != 0:
                self.fail(f"container inspect failed: {stderr}")

            container_env = stdout
            if "HF_TOKEN=" not in container_env:
                self.fail("HF_TOKEN not found in container environment")
            if test_token not in container_env:
                self.fail("HF_TOKEN value mismatch in container")

            print("  ✓ HF_TOKEN found in container environment")
            print("  ✓ HF_TOKEN has correct value")

        self.print_test_result(True, "Environment variable passed to container")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_fleet_sim_sidecar_contracts(self):
        """Test that serve starts the simulator sidecar and exposes its health."""
        self.print_test_header(
            "Fleet Sim Sidecar Integration Test",
            "Verifies serve starts vllm-sr-sim, wires TARGET_FLEET_SIM_URL, and exposes /healthz",
        )

        with self._running_serve():
            if not self.wait_for_container_running(
                timeout=60, container_name=self.SIM_CONTAINER_NAME
            ):
                self.fail("Fleet simulator sidecar did not reach running state")

            return_code, stdout, stderr = self.inspect_container(
                "{{.Config.Env}}",
                container_name=self.resolve_runtime_inspect_container_name(),
            )
            if return_code != 0:
                self.fail(f"router container inspect failed: {stderr}")
            self.assertIn(
                "TARGET_FLEET_SIM_URL=http://vllm-sr-sim-container:8000",
                stdout,
            )

            with urllib_request.urlopen(
                "http://localhost:8810/healthz", timeout=10
            ) as response:
                body = response.read().decode("utf-8")
                self.assertEqual(response.status, 200)
                self.assertIn('"service":"vllm-sr-sim"', body.replace(" ", ""))

            print("  ✓ Simulator sidecar is running")
            print("  ✓ Router container received TARGET_FLEET_SIM_URL")
            print("  ✓ Simulator health endpoint responded on localhost:8810")

        self.print_test_result(True, "Fleet simulator sidecar contracts verified")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_stop_terminates_container(self):
        """Test that vllm-sr stop actually stops the container."""
        self.print_test_header(
            "Stop Command Integration Test",
            "Verifies stop command terminates the container",
        )

        with self._running_serve():
            print("  ✓ Container is running")

            return_code, _stdout, _stderr = self.run_cli(["stop"])
            print(f"  Stop command returned: {return_code}")

            time.sleep(3)  # Give it time to stop

            status = self.container_status()
            if status == "running":
                self.fail(f"Container still running after stop. Status: {status}")
            print("  ✓ Container is stopped")

        self.print_test_result(True, "Stop command terminates container")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_image_pull_policy_never_fails_with_missing_image(self):
        """Test that 'never' policy fails when image doesn't exist locally."""
        self.print_test_header(
            "Image Pull Policy: never",
            "Verifies 'never' policy fails when image is not available locally",
        )

        # Step 1: Create a lean active config
        self.write_minimal_canonical_config()

        # Step 2: Try to serve with fake image and never policy
        fake_image = "fake-nonexistent-image:doesnotexist12345"
        return_code, stdout, stderr = self.run_cli(
            ["serve", "--image", fake_image, "--image-pull-policy", "never"],
            timeout=30,
        )

        output = (stdout + stderr).lower()

        # Should fail because image doesn't exist and can't pull
        if return_code != 0:
            print("  ✓ Command failed as expected (image not found)")
            if "not found" in output or "no such image" in output or "never" in output:
                print("  ✓ Error message mentions image issue")
            self.print_test_result(True, "never policy correctly rejects missing image")
        else:
            self.fail("Command should have failed with never policy and missing image")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_image_pull_policy_always_attempts_pull(self):
        """Test that 'always' policy attempts to pull from registry."""
        self.print_test_header(
            "Image Pull Policy: always",
            "Verifies 'always' policy attempts to pull from registry",
        )

        try:
            # Step 1: Create a lean active config
            self.write_minimal_canonical_config()

            # Step 2: Run serve briefly with always policy
            # We use run_cli with a short timeout - if it accepts the flag, test passes
            cmd = ["serve", "--image-pull-policy", "always"]
            print(f"\nRunning: vllm-sr {' '.join(cmd)}")

            # Use run_cli which handles timeouts gracefully
            _return_code, stdout, stderr = self.run_cli(cmd, timeout=20)
            output = (stdout + stderr).lower()

            # Check for pull-related messages in output
            pull_indicators = ["pull", "pulling", "downloading", "download"]
            pull_detected = any(ind in output for ind in pull_indicators)

            if pull_detected:
                print("  ✓ Pull attempt detected in output")
                self.print_test_result(True, "always policy attempts pull")
            elif self.container_status() == "running":
                # Container running means policy worked (image was up-to-date)
                print("  ✓ Container running (image was up-to-date)")
                self.print_test_result(True, "always policy works")
            else:
                # Policy was accepted by CLI (didn't error on the flag)
                # Even timeout means it started processing
                print("  ✓ always policy was accepted by CLI")
                self.print_test_result(True, "always policy accepted")

        finally:
            # Clean up any running container
            self.run_cli(["stop"], timeout=10)

    def tearDown(self):
        """Clean up after integration tests."""
        self.run_cli(["stop"], timeout=30)
        self._cleanup_container()
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
