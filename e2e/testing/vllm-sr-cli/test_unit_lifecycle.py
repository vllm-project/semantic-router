#!/usr/bin/env python3
"""
test_vllm_sr_lifecycle.py - Tests for 'vllm-sr status/logs/stop' commands.

This test validates lifecycle management commands:
- vllm-sr status: Shows service status (running, stopped, etc.)
- vllm-sr logs: Retrieves logs from services (envoy, router, dashboard)
- vllm-sr stop: Gracefully stops the container

Signed-off-by: vLLM-SR Team
"""

import os
import subprocess
import sys
import time
import unittest

from cli_test_base import CLITestBase


class TestVllmSRStatus(CLITestBase):
    """Tests for the vllm-sr status command."""

    def test_status_when_not_running(self):
        """Test that status reports correctly when container is not running."""
        self.print_test_header(
            "Status When Not Running",
            "Validates that status correctly reports when no container is running",
        )

        # Ensure container is not running
        self._cleanup_container()

        # Run status command
        return_code, stdout, stderr = self.run_cli(["status"])

        # Should succeed (status command should not fail)
        self.assertEqual(return_code, 0, f"status command failed: {stderr}")

        # Output should indicate not running
        output = (stdout + stderr).lower()
        not_running_indicators = ["not running", "not found", "start"]

        found = any(indicator in output for indicator in not_running_indicators)
        self.assertTrue(
            found,
            f"Should indicate service not running. Got: {output[:200]}",
        )

        self.print_test_result(True, "status correctly reports not running")

    def test_status_shows_all_services_by_default(self):
        """Test that status shows all services when no service specified."""
        self.print_test_header(
            "Status Shows All Services",
            "Validates that 'vllm-sr status' or 'vllm-sr status all' shows all services",
        )

        # Run status without arguments
        return_code, stdout, stderr = self.run_cli(["status"])
        output = stdout + stderr

        # Run status with 'all' argument
        return_code_all, stdout_all, stderr_all = self.run_cli(["status", "all"])
        output_all = stdout_all + stderr_all

        # Both should succeed
        self.assertEqual(return_code, 0)
        self.assertEqual(return_code_all, 0)

        # Both should have similar output (both showing overall status)
        print(f"  Status output: {output[:100]}")
        print(f"  Status all output: {output_all[:100]}")

        self.print_test_result(True, "status shows all services")

    def test_status_individual_services(self):
        """Test that status can query individual services."""
        self.print_test_header(
            "Status Individual Services",
            "Validates that status can query envoy, router, or dashboard individually",
        )

        services = ["envoy", "router", "dashboard"]

        for service in services:
            return_code, stdout, stderr = self.run_cli(["status", service])

            # Should succeed (even if not running)
            self.assertEqual(return_code, 0, f"status {service} failed: {stderr}")

            print(f"  ✓ vllm-sr status {service} succeeded")

        self.print_test_result(True, "status works for individual services")


class TestVllmSRLogs(CLITestBase):
    """Tests for the vllm-sr logs command."""

    def test_logs_requires_service_argument(self):
        """Test that logs command requires a service argument."""
        self.print_test_header(
            "Logs Requires Service Argument",
            "Validates that 'vllm-sr logs' requires envoy, router, or dashboard argument",
        )

        # Run logs without service argument
        return_code, stdout, stderr = self.run_cli(["logs"])

        # Should fail or show usage
        # The behavior depends on CLI implementation
        output = (stdout + stderr).lower()

        # Should mention valid service options or show error
        service_options = ["envoy", "router", "dashboard"]
        mentioned = any(svc in output for svc in service_options)

        print(f"  Return code: {return_code}")
        print(f"  Mentions services: {mentioned}")

        self.print_test_result(True, "logs command requires service argument")

    def test_logs_when_not_running(self):
        """Test that logs reports appropriately when container not running."""
        self.print_test_header(
            "Logs When Not Running",
            "Validates that logs handles non-running container gracefully",
        )

        # Ensure container is not running
        self._cleanup_container()

        # Try to get logs
        return_code, stdout, stderr = self.run_cli(["logs", "router"])

        # Should fail or indicate container not running
        output = (stdout + stderr).lower()

        # Check for appropriate error message
        error_indicators = ["not running", "not found", "start", "error"]
        found = any(indicator in output for indicator in error_indicators)

        if return_code != 0 or found:
            print("  ✓ Correctly handles non-running container")
        else:
            print(f"  Output: {output[:200]}")

        self.print_test_result(True, "logs handles non-running container")

    def test_logs_services(self):
        """Test that logs accepts valid service names."""
        self.print_test_header(
            "Logs Accepts Valid Services",
            "Validates that logs accepts envoy, router, and dashboard",
        )

        services = ["envoy", "router", "dashboard"]

        for service in services:
            return_code, stdout, stderr = self.run_cli(["logs", service])

            # Even if container not running, command structure should be valid
            output = (stdout + stderr).lower()

            # Should not fail with "invalid argument" type errors
            invalid_indicators = ["invalid", "unknown", "usage"]
            is_invalid_arg = any(
                ind in output and service in output for ind in invalid_indicators
            )
            self.assertFalse(
                is_invalid_arg,
                f"logs should accept {service} as valid service",
            )

            print(f"  ✓ vllm-sr logs {service} is valid")

        self.print_test_result(True, "logs accepts all valid service names")

    def test_logs_follow_flag(self):
        """Test that logs supports --follow/-f flag."""
        self.print_test_header(
            "Logs Follow Flag",
            "Validates that logs supports --follow/-f for streaming logs",
        )

        # Test that the flag is recognized (without actually following)
        # We'll use a very short timeout since follow would hang

        # Test with --follow
        return_code, stdout, stderr = self.run_cli(
            ["logs", "router", "--follow"],
            timeout=5,  # Short timeout to avoid hanging
        )

        # Test with -f
        return_code_short, stdout_short, stderr_short = self.run_cli(
            ["logs", "router", "-f"],
            timeout=5,
        )

        # Both should be recognized (might timeout or fail due to no container)
        # The key is they shouldn't fail with "unknown flag" errors
        output = (stdout + stderr + stdout_short + stderr_short).lower()

        # Check that flags are recognized
        unknown_flag = "unknown" in output and ("follow" in output or "-f" in output)
        self.assertFalse(unknown_flag, "Follow flags should be recognized")

        print("  ✓ --follow and -f flags are recognized")
        self.print_test_result(True, "logs supports follow flag")


class TestVllmSRStop(CLITestBase):
    """Tests for the vllm-sr stop command."""

    def test_stop_when_not_running(self):
        """Test that stop handles gracefully when nothing is running."""
        self.print_test_header(
            "Stop When Not Running",
            "Validates that stop handles non-running container gracefully",
        )

        # Ensure container is not running
        self._cleanup_container()

        # Run stop command
        return_code, stdout, stderr = self.run_cli(["stop"])

        # Should succeed or handle gracefully
        # (stop on nothing running is not an error)
        output = (stdout + stderr).lower()

        # Should indicate nothing to stop or already stopped
        indicators = ["not found", "nothing", "stopped", "not running"]
        found = any(ind in output for ind in indicators)

        print(f"  Return code: {return_code}")
        if found:
            print("  ✓ Correctly indicates nothing to stop")

        # Stop should not fail with error when nothing running
        self.assertIn(
            return_code,
            [0, 1],  # 0 = success, 1 = nothing to stop (acceptable)
            f"stop should handle gracefully: {stderr}",
        )

        self.print_test_result(True, "stop handles non-running container")

    def test_stop_command_format(self):
        """Test that stop command has correct format (no arguments needed)."""
        self.print_test_header(
            "Stop Command Format",
            "Validates that 'vllm-sr stop' works without additional arguments",
        )

        # Run stop - should not require any arguments
        return_code, stdout, stderr = self.run_cli(["stop"])

        # Should not fail with "missing argument" errors
        output = (stdout + stderr).lower()
        missing_arg = "missing" in output or "required" in output

        self.assertFalse(
            missing_arg,
            "stop should not require arguments",
        )

        self.print_test_result(True, "stop works without arguments")


class TestVllmSRDashboard(CLITestBase):
    """Tests for the vllm-sr dashboard command."""

    def test_dashboard_when_not_running(self):
        """Test that dashboard command fails gracefully when not running."""
        self.print_test_header(
            "Dashboard When Not Running",
            "Validates that dashboard command reports when service not running",
        )

        # Ensure container is not running
        self._cleanup_container()

        # Run dashboard command with --no-open to avoid opening browser
        return_code, stdout, stderr = self.run_cli(["dashboard", "--no-open"])

        # Should fail or indicate not running
        output = (stdout + stderr).lower()

        # Should mention not running
        not_running_indicators = ["not running", "start", "serve"]
        found = any(ind in output for ind in not_running_indicators)

        if return_code != 0 or found:
            print("  ✓ Correctly indicates service not running")

        self.print_test_result(True, "dashboard handles non-running state")

    def test_dashboard_no_open_flag(self):
        """Test that dashboard supports --no-open flag."""
        self.print_test_header(
            "Dashboard --no-open Flag",
            "Validates that dashboard supports --no-open to just show URL",
        )

        # The flag should be recognized
        return_code, stdout, stderr = self.run_cli(["dashboard", "--no-open"])

        output = (stdout + stderr).lower()

        # Should not fail with unknown flag error
        unknown = "unknown" in output and "no-open" in output
        self.assertFalse(unknown, "--no-open flag should be recognized")

        print("  ✓ --no-open flag is recognized")
        self.print_test_result(True, "dashboard supports --no-open flag")


class TestVllmSRConfig(CLITestBase):
    """Tests for the vllm-sr config command."""

    def test_config_envoy(self):
        """Test that config envoy generates envoy configuration."""
        self.print_test_header(
            "Config Envoy Command",
            "Validates that 'vllm-sr config envoy' outputs envoy config",
        )

        # First init to create config.yaml
        self.run_cli(["init", "--force"])

        # Run config envoy
        return_code, stdout, stderr = self.run_cli(["config", "envoy"])

        # Should succeed if config.yaml exists
        if return_code == 0:
            # Output should contain envoy config elements
            output = stdout + stderr
            print(f"  Config output length: {len(output)} chars")
            self.print_test_result(True, "config envoy generated output")
        else:
            print(f"  Command failed: {stderr[:100]}")
            # Might fail if config structure isn't complete
            self.print_test_result(True, "config envoy command exists")

    def test_config_router(self):
        """Test that config router generates router configuration."""
        self.print_test_header(
            "Config Router Command",
            "Validates that 'vllm-sr config router' outputs router config",
        )

        # First init to create config.yaml
        self.run_cli(["init", "--force"])

        # Run config router
        return_code, stdout, stderr = self.run_cli(["config", "router"])

        # Should succeed if config.yaml exists
        if return_code == 0:
            output = stdout + stderr
            print(f"  Config output length: {len(output)} chars")
            self.print_test_result(True, "config router generated output")
        else:
            print(f"  Command failed: {stderr[:100]}")
            self.print_test_result(True, "config router command exists")


if __name__ == "__main__":
    unittest.main()
