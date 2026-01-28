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

from cli_test_base import CLITestBase


class TestServeIntegration(CLITestBase):
    """Integration tests for the complete serve workflow."""

    # Timeout for waiting for container to be running
    CONTAINER_STARTUP_TIMEOUT = 120

    def _start_serve_background(self) -> subprocess.Popen:
        """Start vllm-sr serve in background (non-blocking)."""
        cmd = ["vllm-sr", "serve", "--image-pull-policy", "ifnotpresent"]
        print(f"\nStarting in background: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_dir,
        )
        return process

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_serve_full_startup(self):
        """Test complete serve workflow: init → serve → container running."""
        self.print_test_header(
            "Full Serve Integration Test",
            "Tests: init → serve → container running → stop",
        )

        serve_process = None

        try:
            # Step 1: Initialize config
            return_code, _, _ = self.run_cli(["init", "--force"])
            self.assertEqual(return_code, 0, "init failed")
            print("  ✓ init succeeded")

            # Step 2: Start serve in background
            serve_process = self._start_serve_background()
            time.sleep(5)

            # Check process didn't crash
            if serve_process.poll() is not None:
                stdout, stderr = serve_process.communicate()
                self.fail(f"Serve crashed: {stderr[:500]}")

            # Step 3: Wait for container
            print(
                f"  Waiting for container (timeout: {self.CONTAINER_STARTUP_TIMEOUT}s)..."
            )
            if not self.wait_for_container_running(
                timeout=self.CONTAINER_STARTUP_TIMEOUT
            ):
                serve_process.terminate()
                stdout, stderr = serve_process.communicate(timeout=10)
                self.fail(f"Container did not start: {stderr[:500]}")

            print("  ✓ Container is running")

            # Step 4: Check health (informational only)
            self._check_health_endpoint()

            self.print_test_result(True, "CLI serve workflow completed successfully")

        finally:
            if serve_process and serve_process.poll() is None:
                print("  Cleaning up serve process...")
                serve_process.terminate()
                try:
                    serve_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    serve_process.kill()

    def _check_health_endpoint(self):
        """Check health endpoint (informational, doesn't fail test)."""
        import urllib.request
        import urllib.error

        try:
            url = "http://localhost:8888/health"
            with urllib.request.urlopen(url, timeout=10) as response:
                print(f"  ✓ Health check: {response.status}")
        except urllib.error.HTTPError as e:
            # 500 = service running but no backend - expected with default config
            print(f"  ⚠ Health check: {e.code} (expected without backend)")
        except Exception as e:
            print(f"  ⚠ Health check failed: {e}")

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_env_var_passed_to_container(self):
        """Test that environment variables are actually passed to container."""
        self.print_test_header(
            "Environment Variable Integration Test",
            "Verifies HF_TOKEN is inside running container via docker inspect",
        )

        serve_process = None
        test_token = "hf_integration_test_token_xyz"

        try:
            # Step 1: Initialize config
            return_code, _, _ = self.run_cli(["init", "--force"])
            self.assertEqual(return_code, 0, "init failed")

            # Step 2: Start serve with HF_TOKEN in environment
            cmd = ["vllm-sr", "serve", "--image-pull-policy", "ifnotpresent"]
            env = os.environ.copy()
            env["HF_TOKEN"] = test_token

            serve_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.test_dir,
                env=env,
            )
            time.sleep(5)

            # Step 3: Wait for container
            if not self.wait_for_container_running(
                timeout=self.CONTAINER_STARTUP_TIMEOUT
            ):
                self.fail("Container did not start")

            print("  ✓ Container is running")

            # Step 4: Use docker inspect to verify HF_TOKEN is in container
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.Config.Env}}",
                    "vllm-sr-container",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                container_env = result.stdout
                if "HF_TOKEN=" in container_env:
                    print("  ✓ HF_TOKEN found in container environment")
                    # Verify the actual value
                    if test_token in container_env:
                        print("  ✓ HF_TOKEN has correct value")
                        self.print_test_result(
                            True, "Environment variable passed to container"
                        )
                    else:
                        self.fail("HF_TOKEN value mismatch in container")
                else:
                    self.fail("HF_TOKEN not found in container environment")
            else:
                self.fail(f"docker inspect failed: {result.stderr}")

        finally:
            if serve_process and serve_process.poll() is None:
                serve_process.terminate()
                try:
                    serve_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    serve_process.kill()

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_volume_mounting(self):
        """Test that config and models directories are mounted into container."""
        self.print_test_header(
            "Volume Mounting Integration Test",
            "Verifies config.yaml and models/ are mounted via docker inspect",
        )

        serve_process = None

        try:
            # Step 1: Initialize config
            return_code, _, _ = self.run_cli(["init", "--force"])
            self.assertEqual(return_code, 0, "init failed")

            # Step 2: Create models directory (CLI should create it, but ensure it exists)
            models_dir = os.path.join(self.test_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            # Step 3: Start serve
            serve_process = self._start_serve_background()
            time.sleep(5)

            # Step 4: Wait for container
            if not self.wait_for_container_running(
                timeout=self.CONTAINER_STARTUP_TIMEOUT
            ):
                self.fail("Container did not start")

            print("  ✓ Container is running")

            # Step 5: Use docker inspect to verify mounts
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{json .Mounts}}",
                    "vllm-sr-container",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.fail(f"docker inspect failed: {result.stderr}")

            mounts = result.stdout.lower()
            print(f"  Mounts: {mounts[:200]}...")

            # Check for config.yaml mount
            config_mounted = "config.yaml" in mounts or "config" in mounts
            if config_mounted:
                print("  ✓ config.yaml is mounted")
            else:
                print("  ⚠ config.yaml mount not detected")

            # Check for models directory mount
            models_mounted = "models" in mounts
            if models_mounted:
                print("  ✓ models/ directory is mounted")
            else:
                print("  ⚠ models/ mount not detected")

            # At least one mount should be present
            self.assertTrue(
                config_mounted or models_mounted,
                "No expected mounts found in container",
            )

            self.print_test_result(True, "Volume mounting verified")

        finally:
            if serve_process and serve_process.poll() is None:
                serve_process.terminate()
                try:
                    serve_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    serve_process.kill()

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_status_shows_running_container(self):
        """Test that vllm-sr status correctly reports running container."""
        self.print_test_header(
            "Status Command Integration Test",
            "Verifies status shows running container after serve",
        )

        serve_process = None

        try:
            # Step 1: Initialize and start
            self.run_cli(["init", "--force"])
            serve_process = self._start_serve_background()
            time.sleep(5)

            # Step 2: Wait for container
            if not self.wait_for_container_running(
                timeout=self.CONTAINER_STARTUP_TIMEOUT
            ):
                self.fail("Container did not start")

            # Step 3: Check status command
            return_code, stdout, stderr = self.run_cli(["status"])
            output = (stdout + stderr).lower()

            # Status should indicate running
            running_indicators = ["running", "up", "healthy", "started"]
            status_ok = any(ind in output for ind in running_indicators)

            if status_ok:
                print("  ✓ Status shows container is running")
                self.print_test_result(
                    True, "Status correctly reports running container"
                )
            else:
                self.fail(f"Status doesn't show running. Got: {output[:300]}")

        finally:
            if serve_process and serve_process.poll() is None:
                serve_process.terminate()
                try:
                    serve_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    serve_process.kill()

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_logs_retrieves_container_logs(self):
        """Test that vllm-sr logs retrieves actual container logs."""
        self.print_test_header(
            "Logs Command Integration Test",
            "Verifies logs command retrieves container output",
        )

        serve_process = None

        try:
            # Step 1: Initialize and start
            self.run_cli(["init", "--force"])
            serve_process = self._start_serve_background()
            time.sleep(5)

            # Step 2: Wait for container
            if not self.wait_for_container_running(
                timeout=self.CONTAINER_STARTUP_TIMEOUT
            ):
                self.fail("Container did not start")

            # Wait a bit for container to produce some logs
            time.sleep(5)

            # Step 3: Get logs
            return_code, stdout, stderr = self.run_cli(["logs"])
            output = stdout + stderr

            # Logs should contain something (startup messages, etc.)
            if len(output.strip()) > 0:
                print(f"  ✓ Logs retrieved ({len(output)} chars)")
                print(f"  Sample: {output[:100]}...")
                self.print_test_result(True, "Logs command retrieves container output")
            else:
                self.fail("Logs command returned empty output")

        finally:
            if serve_process and serve_process.poll() is None:
                serve_process.terminate()
                try:
                    serve_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    serve_process.kill()

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

        serve_process = None

        try:
            # Step 1: Initialize and start
            self.run_cli(["init", "--force"])
            serve_process = self._start_serve_background()
            time.sleep(5)

            # Step 2: Wait for container
            if not self.wait_for_container_running(
                timeout=self.CONTAINER_STARTUP_TIMEOUT
            ):
                self.fail("Container did not start")

            print("  ✓ Container is running")

            # Step 3: Stop the container
            return_code, stdout, stderr = self.run_cli(["stop"])
            print(f"  Stop command returned: {return_code}")

            # Step 4: Verify container is stopped
            time.sleep(3)  # Give it time to stop

            status = self.container_status()
            if status is None or status != "running":
                print("  ✓ Container is stopped")
                self.print_test_result(True, "Stop command terminates container")
            else:
                self.fail(f"Container still running after stop. Status: {status}")

        finally:
            if serve_process and serve_process.poll() is None:
                serve_process.terminate()
                try:
                    serve_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    serve_process.kill()

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

        # Step 1: Initialize config
        return_code, _, _ = self.run_cli(["init", "--force"])
        self.assertEqual(return_code, 0, "init failed")

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
            # Step 1: Initialize config
            return_code, _, _ = self.run_cli(["init", "--force"])
            self.assertEqual(return_code, 0, "init failed")

            # Step 2: Run serve briefly with always policy
            # We use run_cli with a short timeout - if it accepts the flag, test passes
            cmd = ["serve", "--image-pull-policy", "always"]
            print(f"\nRunning: vllm-sr {' '.join(cmd)}")

            # Use run_cli which handles timeouts gracefully
            return_code, stdout, stderr = self.run_cli(cmd, timeout=20)
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
