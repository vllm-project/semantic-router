#!/usr/bin/env python3
"""Unit coverage for CLI test-base topology compatibility helpers."""

import os
import subprocess
import unittest
from unittest import mock

import cli_test_base
import run_cli_tests
from cli_test_base import CLITestBase, stack_scoped_test_container_name


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
        self.base.test_dir = os.getcwd()

    def _mock_statuses(self, statuses_by_name: dict[str, str]):
        def side_effect(command, *, timeout, **_kwargs):
            if command[1] == "inspect":
                if command[2:4] == ["--format", "{{.State.Status}}"]:
                    name = command[-1]
                    status = statuses_by_name.get(name)
                    if status is None:
                        return _completed_process(returncode=1)
                    return _completed_process(stdout=status)
                return _completed_process(stdout=command[-1])
            raise AssertionError(f"unexpected command: {command}")

        return side_effect

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_container_status_defaults_to_split_runtime(self, run_subprocess):
        run_subprocess.side_effect = self._mock_statuses(
            {
                self.base.ROUTER_CONTAINER_NAME: "running",
                self.base.ENVOY_CONTAINER_NAME: "running",
                self.base.DASHBOARD_CONTAINER_NAME: "running",
            }
        )

        self.assertEqual(self.base.container_status(), "running")

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_container_status_reports_split_runtime_exited(self, run_subprocess):
        run_subprocess.side_effect = self._mock_statuses(
            {
                self.base.ROUTER_CONTAINER_NAME: "exited",
            }
        )

        self.assertEqual(self.base.container_status(), "exited")

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_inspect_defaults_to_router_container_for_split_runtime(
        self, run_subprocess
    ):
        run_subprocess.side_effect = self._mock_statuses(
            {
                self.base.ROUTER_CONTAINER_NAME: "running",
                self.base.ENVOY_CONTAINER_NAME: "running",
                self.base.DASHBOARD_CONTAINER_NAME: "running",
            }
        )

        return_code, stdout, stderr = self.base.inspect_container("{{.Name}}")

        self.assertEqual(return_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, self.base.ROUTER_CONTAINER_NAME)

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_inspect_defaults_to_router_container_when_runtime_absent(
        self, run_subprocess
    ):
        run_subprocess.side_effect = self._mock_statuses({})

        return_code, stdout, stderr = self.base.inspect_container("{{.Name}}")

        self.assertEqual(return_code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, self.base.ROUTER_CONTAINER_NAME)

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
            CLITestBase.PROBE_CONTAINER_NAME,
            *CLITestBase.AUXILIARY_CONTAINER_NAMES,
        ):
            self.assertIn(container_name, removed_container_names)

    @mock.patch.dict(
        os.environ,
        {
            "VLLM_SR_STACK_NAME": "isolated-test",
            "VLLM_SR_PORT_OFFSET": "4200",
        },
        clear=False,
    )
    @mock.patch.object(CLITestBase, "_cleanup_container")
    @mock.patch.object(CLITestBase, "_detect_container_runtime", return_value="docker")
    def test_set_up_class_scopes_container_names(
        self,
        _detect_runtime,
        _cleanup_container,
    ):
        class IsolatedCLITestBase(CLITestBase):
            pass

        IsolatedCLITestBase.setUpClass()

        self.assertEqual(
            IsolatedCLITestBase.ROUTER_CONTAINER_NAME,
            "isolated-test-vllm-sr-router-container",
        )
        self.assertEqual(
            IsolatedCLITestBase.PROBE_CONTAINER_NAME,
            "isolated-test-vllm-sr-cli-test-probe",
        )
        self.assertEqual(IsolatedCLITestBase.runtime_stack.port_offset, 4200)

    def test_test_only_container_names_follow_the_runtime_stack(self):
        base_name = "vllm-sr-cli-test-control-redis"

        self.assertEqual(
            stack_scoped_test_container_name("vllm-sr", base_name), base_name
        )
        self.assertEqual(
            stack_scoped_test_container_name("first-run", base_name),
            "first-run-vllm-sr-cli-test-control-redis",
        )
        self.assertNotEqual(
            stack_scoped_test_container_name("first-run", base_name),
            stack_scoped_test_container_name("second-run", base_name),
        )

    @mock.patch.dict(os.environ, {"CONTAINER_RUNTIME": "podman"}, clear=False)
    def test_cli_test_base_accepts_podman_env_override(self):
        with mock.patch.object(
            cli_test_base.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/podman" if name == "podman" else None,
        ):
            self.assertEqual(CLITestBase._detect_container_runtime(), "podman")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_cli_test_base_falls_back_to_podman(self):
        with mock.patch.object(
            cli_test_base.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/podman" if name == "podman" else None,
        ):
            self.assertEqual(CLITestBase._detect_container_runtime(), "podman")

    @mock.patch.dict(os.environ, {"RUN_INTEGRATION_TESTS": "false"}, clear=True)
    def test_cli_test_base_uses_stub_runtime_for_unit_only_suite(self):
        with mock.patch.object(cli_test_base.shutil, "which", return_value=None):
            self.assertEqual(CLITestBase._detect_container_runtime(), "docker")

    @mock.patch.dict(os.environ, {"RUN_INTEGRATION_TESTS": "true"}, clear=True)
    def test_cli_test_base_requires_runtime_for_integration_suite(self):
        with (
            mock.patch.object(cli_test_base.shutil, "which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "Neither docker nor podman"),
        ):
            CLITestBase._detect_container_runtime()

    @mock.patch.object(CLITestBase, "_run_subprocess")
    def test_run_cli_preserves_partial_output_on_timeout(self, run_subprocess):
        run_subprocess.side_effect = subprocess.TimeoutExpired(
            cmd=["vllm-sr", "serve"],
            timeout=30,
            output="Created bootstrap setup config\n",
            stderr="Waiting for Dashboard to become healthy...\n",
        )

        return_code, stdout, stderr = self.base.run_cli(["serve"], timeout=30)

        self.assertEqual(return_code, -1)
        self.assertIn("bootstrap setup config", stdout)
        self.assertIn("dashboard to become healthy", stderr.lower())
        self.assertIn("command timed out after 30 seconds", stderr.lower())


class TestCLITestRunnerRuntimeDetection(unittest.TestCase):
    """Verify the standalone CLI test runner matches Docker-only runtime rules."""

    @mock.patch.dict(os.environ, {"CONTAINER_RUNTIME": "podman"}, clear=False)
    def test_runner_accepts_podman_env_override(self):
        with mock.patch.object(
            run_cli_tests.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/podman" if name == "podman" else None,
        ):
            self.assertEqual(run_cli_tests.detect_container_runtime(), "podman")

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
