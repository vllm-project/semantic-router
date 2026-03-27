#!/usr/bin/env python3
"""Unit coverage for CLI test-base topology compatibility helpers."""

import os
import subprocess
import unittest
from unittest import mock

import cli_test_base
import run_cli_tests
from cli_test_base import CLITestBase


def _completed_process(*, stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class TestCLITestBaseRuntimeTopology(unittest.TestCase):
    """Verify the CLI test harness works with split runtime containers."""

    def setUp(self):
        self.base = CLITestBase(methodName="runTest")
        self.base.container_runtime = "docker"

    def _mock_statuses(self, statuses_by_name: dict[str, str]):
        def side_effect(command, *, timeout, **_kwargs):
            if command[1:4] == ["ps", "-a", "--filter"]:
                name = command[4].split("=", 1)[1]
                status = statuses_by_name.get(name, "")
                return _completed_process(stdout=status)
            if command[1] == "inspect":
                return _completed_process(stdout=command[-1])
            raise AssertionError(f"unexpected command: {command}")

        return side_effect

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_container_status_defaults_to_split_runtime(self, run_subprocess):
        run_subprocess.side_effect = self._mock_statuses(
            {
                self.base.ROUTER_CONTAINER_NAME: "Up 5 seconds",
                self.base.ENVOY_CONTAINER_NAME: "Up 5 seconds",
                self.base.DASHBOARD_CONTAINER_NAME: "Up 5 seconds",
            }
        )

        self.assertEqual(self.base.container_status(), "running")

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_container_status_reports_legacy_runtime_residue_as_running(
        self, run_subprocess
    ):
        run_subprocess.side_effect = self._mock_statuses(
            {
                self.base.CONTAINER_NAME: "Up 5 seconds",
                self.base.ROUTER_CONTAINER_NAME: "",
                self.base.ENVOY_CONTAINER_NAME: "",
                self.base.DASHBOARD_CONTAINER_NAME: "",
            }
        )

        self.assertEqual(self.base.container_status(), "running")

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_inspect_defaults_to_router_container_for_split_runtime(
        self, run_subprocess
    ):
        run_subprocess.side_effect = self._mock_statuses(
            {
                self.base.ROUTER_CONTAINER_NAME: "Up 5 seconds",
                self.base.ENVOY_CONTAINER_NAME: "Up 5 seconds",
                self.base.DASHBOARD_CONTAINER_NAME: "Up 5 seconds",
            }
        )

        return_code, stdout, stderr = self.base.inspect_container("{{.Name}}")

        self.assertEqual(return_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, self.base.ROUTER_CONTAINER_NAME)

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_inspect_falls_back_to_legacy_runtime_residue(self, run_subprocess):
        run_subprocess.side_effect = self._mock_statuses(
            {
                self.base.CONTAINER_NAME: "Up 5 seconds",
                self.base.ROUTER_CONTAINER_NAME: "",
                self.base.ENVOY_CONTAINER_NAME: "",
                self.base.DASHBOARD_CONTAINER_NAME: "",
            }
        )

        return_code, stdout, stderr = self.base.inspect_container("{{.Name}}")

        self.assertEqual(return_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, self.base.CONTAINER_NAME)

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_cleanup_removes_split_runtime_and_observability_containers(
        self, run_subprocess
    ):
        CLITestBase.container_runtime = "docker"
        run_subprocess.return_value = _completed_process()

        CLITestBase._cleanup_container()

        removed_container_names = [
            call.args[0][-1]
            for call in run_subprocess.call_args_list
            if call.args[0][1] == "rm"
        ]
        for container_name in (
            CLITestBase.CONTAINER_NAME,
            CLITestBase.ROUTER_CONTAINER_NAME,
            CLITestBase.ENVOY_CONTAINER_NAME,
            CLITestBase.DASHBOARD_CONTAINER_NAME,
            CLITestBase.SIM_CONTAINER_NAME,
            *CLITestBase.AUXILIARY_CONTAINER_NAMES,
        ):
            self.assertIn(container_name, removed_container_names)

    @mock.patch.dict(os.environ, {"CONTAINER_RUNTIME": "podman"}, clear=False)
    def test_cli_test_base_rejects_podman_env_override(self):
        with mock.patch.object(
            cli_test_base.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/podman" if name == "podman" else None,
        ), self.assertRaisesRegex(RuntimeError, "require Docker"):
            CLITestBase._detect_container_runtime()

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_cli_test_base_rejects_podman_only_hosts(self):
        with mock.patch.object(
            cli_test_base.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/podman" if name == "podman" else None,
        ), self.assertRaisesRegex(RuntimeError, "Podman is installed but unsupported"):
            CLITestBase._detect_container_runtime()


class TestCLITestRunnerRuntimeDetection(unittest.TestCase):
    """Verify the standalone CLI test runner matches Docker-only runtime rules."""

    @mock.patch.dict(os.environ, {"CONTAINER_RUNTIME": "podman"}, clear=False)
    def test_runner_rejects_podman_env_override(self):
        with mock.patch.object(
            run_cli_tests.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/podman" if name == "podman" else None,
        ):
            self.assertIsNone(run_cli_tests.detect_container_runtime())

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_runner_accepts_docker(self):
        with mock.patch.object(
            run_cli_tests.shutil,
            "which",
            side_effect=lambda name: (
                "/usr/local/bin/docker" if name == "docker" else None
            ),
        ):
            self.assertEqual(run_cli_tests.detect_container_runtime(), "docker")


if __name__ == "__main__":
    unittest.main()
