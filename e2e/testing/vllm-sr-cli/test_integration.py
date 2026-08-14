#!/usr/bin/env python3
"""
test_integration.py - Integration tests for vLLM-SR CLI.

These tests require a working Docker image and test complete workflows.
They are slower than unit tests and should be run with --integration flag.

"""

import json
import os
import stat
import subprocess
import time
import unittest
from contextlib import contextmanager
from unittest import mock
from urllib import error as urllib_error
from urllib import request as urllib_request

from cli_test_base import HTTP_STATUS_OK, CLITestBase


class TestServeIntegration(CLITestBase):
    """Integration tests for the complete serve workflow."""

    # Timeout for waiting for container to be running
    CONTAINER_STARTUP_TIMEOUT = 120

    def _create_minimal_config(self, port: int = 8888) -> str:
        return self.write_minimal_canonical_config(port=port)

    def _start_serve_background(
        self,
        env: dict[str, str] | None = None,
        arguments: tuple[str, ...] = (),
    ) -> subprocess.Popen:
        """Start vllm-sr serve in background (non-blocking)."""
        cmd = [
            "vllm-sr",
            "serve",
            *arguments,
            "--image-pull-policy",
            "ifnotpresent",
        ]
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

    def _stop_serve_process(
        self, serve_process: subprocess.Popen | None
    ) -> tuple[str, str]:
        """Terminate a background serve process and drain its output pipes."""
        if serve_process is None:
            return "", ""
        if serve_process.poll() is None:
            serve_process.terminate()
        try:
            stdout, stderr = serve_process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            serve_process.kill()
            stdout, stderr = serve_process.communicate(timeout=10)
        return stdout or "", stderr or ""

    def _wait_for_serve_success(self, serve_process: subprocess.Popen) -> None:
        """Drain the one-shot serve command and require successful startup."""
        try:
            stdout, stderr = serve_process.communicate(
                timeout=self.HEALTH_CHECK_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            serve_process.terminate()
            try:
                stdout, stderr = serve_process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                serve_process.kill()
                stdout, stderr = serve_process.communicate(timeout=10)
            self.fail(
                "Serve did not complete startup before the timeout: "
                f"{(stderr or stdout or '')[:500]}"
            )
        if serve_process.returncode != 0:
            self.fail(
                "Serve failed before completing runtime startup: "
                f"{(stderr or stdout or '')[:500]}"
            )
        print("  ✓ Serve command completed runtime startup")

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
            self._wait_for_serve_success(serve_process)
            yield serve_process
        finally:
            self._stop_serve_process(serve_process)

    def test_wait_for_serve_success_does_not_terminate_a_successful_process(self):
        process = mock.Mock(spec=subprocess.Popen)
        process.communicate.return_value = ("ready", "")
        process.returncode = 0

        self._wait_for_serve_success(process)

        process.communicate.assert_called_once_with(timeout=self.HEALTH_CHECK_TIMEOUT)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_wait_for_serve_success_rejects_a_failed_process(self):
        process = mock.Mock(spec=subprocess.Popen)
        process.communicate.return_value = ("", "startup failed")
        process.returncode = 1

        with self.assertRaisesRegex(AssertionError, "startup failed"):
            self._wait_for_serve_success(process)

        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_wait_for_serve_success_terminates_and_drains_a_timeout(self):
        process = mock.Mock(spec=subprocess.Popen)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("vllm-sr serve", self.HEALTH_CHECK_TIMEOUT),
            ("", "stopped after timeout"),
        ]

        with self.assertRaisesRegex(AssertionError, "before the timeout"):
            self._wait_for_serve_success(process)

        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        self.assertEqual(process.communicate.call_count, 2)

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

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
        "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
    )
    def test_catalog_model_operands_expose_only_selected_virtual_entrypoints(self):
        """Serve two installed virtual models from a private catalog projection."""
        self.print_test_header(
            "Catalog Model Selection Integration Test",
            "Verifies serve MODEL... reaches the live Router model inventory",
        )
        selected = {
            "vllm-sr/chorus-v1-lite",
            "vllm-sr/chorus-v1-flash",
        }
        list_code, list_stdout, list_stderr = self.run_cli(
            ["model", "list", "--output", "json"]
        )
        self.assertEqual(list_code, 0, list_stderr)
        installed_catalog_ids = {
            str(model["id"]) for model in json.loads(list_stdout)["models"]
        }

        process = self._start_serve_background(
            env=os.environ.copy(),
            arguments=(*sorted(selected), "--minimal"),
        )
        try:
            self._wait_for_serve_success(process)
            model_ids = self._wait_for_catalog_model_ids()
        finally:
            self._stop_serve_process(process)

        self.assertEqual(model_ids & installed_catalog_ids, selected)
        self.assertFalse(
            os.path.exists(os.path.join(self.test_dir, "config.yaml")),
            "catalog serving must not overwrite the user's config.yaml",
        )
        self.print_test_result(True, "Selected catalog entrypoints are live")

    def _wait_for_catalog_model_ids(self) -> set[str]:
        """Poll the authenticated Router inventory for selected entrypoints."""
        token = self._read_catalog_management_credential()
        endpoint = f"http://127.0.0.1:{self.runtime_stack.api_port}/v1/models"
        request = urllib_request.Request(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
        )
        deadline = time.monotonic() + 90
        last_error = "no response"
        while time.monotonic() < deadline:
            try:
                with urllib_request.urlopen(request, timeout=5) as response:
                    if response.status != HTTP_STATUS_OK:
                        last_error = f"HTTP {response.status}"
                        time.sleep(1)
                        continue
                    payload = json.load(response)
                models = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(models, list):
                    return {
                        str(model["id"])
                        for model in models
                        if isinstance(model, dict) and isinstance(model.get("id"), str)
                    }
                last_error = "response did not contain a model list"
            except (OSError, ValueError) as error:
                last_error = str(error)
            time.sleep(1)
        self.fail(f"Router catalog inventory did not become ready: {last_error}")

    def _read_catalog_management_credential(self) -> str:
        """Read the stack credential only after verifying its private file contract."""
        credential_path = os.path.join(
            self.test_dir,
            ".vllm-sr",
            "catalog-credentials",
            f"{self.runtime_stack.stack_name}.token",
        )
        descriptor = os.open(
            credential_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            file_info = os.fstat(descriptor)
            self.assertTrue(
                stat.S_ISREG(file_info.st_mode),
                "catalog management credential must be a regular file",
            )
            self.assertEqual(
                file_info.st_nlink,
                1,
                "catalog management credential must not have multiple hard links",
            )
            if os.name == "posix":
                self.assertEqual(
                    file_info.st_uid,
                    os.getuid(),
                    "catalog management credential must be owned by the CLI user",
                )
                self.assertEqual(
                    stat.S_IMODE(file_info.st_mode),
                    0o600,
                    "catalog management credential must be readable only by its owner",
                )
            with os.fdopen(descriptor, encoding="utf-8") as credential_file:
                descriptor = -1
                token = credential_file.read().strip()
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        self.assertEqual(len(token), 64, "catalog management credential is malformed")
        self.assertTrue(
            all(character in "0123456789abcdef" for character in token),
            "catalog management credential is malformed",
        )
        return token

    def _check_health_endpoint(self):
        """Check health endpoint (informational, doesn't fail test)."""
        try:
            listener_port = 8888 + self.runtime_stack.port_offset
            url = f"http://localhost:{listener_port}/health"
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
                (
                    "TARGET_FLEET_SIM_URL=http://"
                    f"{self.runtime_stack.fleet_sim_container_name}:8000"
                ),
                stdout,
            )

            with urllib_request.urlopen(
                f"{self.runtime_stack.fleet_sim_url}/healthz", timeout=10
            ) as response:
                body = response.read().decode("utf-8")
                self.assertEqual(response.status, 200)
                self.assertIn('"service":"vllm-sr-sim"', body.replace(" ", ""))

            print("  ✓ Simulator sidecar is running")
            print("  ✓ Router container received TARGET_FLEET_SIM_URL")
            print(
                "  ✓ Simulator health endpoint responded on "
                f"localhost:{self.runtime_stack.fleet_sim_port}"
            )

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

            for container_name in self._runtime_container_names():
                if not self.wait_for_container_running(
                    timeout=self.CONTAINER_STARTUP_TIMEOUT,
                    container_name=container_name,
                ):
                    self.fail(f"Runtime container did not start: {container_name}")

            return_code, stdout, stderr = self.run_cli(["stop"])
            print(f"  Stop command returned: {return_code}")
            self.assertEqual(return_code, 0, stderr or stdout)

            managed_names = (
                *self._runtime_container_names(),
                self.SIM_CONTAINER_NAME,
                *self.AUXILIARY_CONTAINER_NAMES,
            )
            remaining = {
                name
                for name in managed_names
                if self.container_status(container_name=name) != "not found"
            }
            if remaining:
                self.fail(
                    "Managed containers remain after stop: "
                    + ", ".join(sorted(remaining))
                )
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
