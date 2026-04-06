#!/usr/bin/env python3
"""Unit coverage for CLI test-base topology compatibility helpers."""

import os
import subprocess
import unittest
from unittest import mock

import cli_test_base
import run_cli_tests
import test_shared_suites
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
        self.base.test_dir = os.getcwd()

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
    def test_container_status_reports_split_runtime_exited(self, run_subprocess):
        run_subprocess.side_effect = self._mock_statuses(
            {
                self.base.ROUTER_CONTAINER_NAME: "Exited (1) 5 seconds ago",
                self.base.ENVOY_CONTAINER_NAME: "",
                self.base.DASHBOARD_CONTAINER_NAME: "",
            }
        )

        self.assertEqual(self.base.container_status(), "exited")

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
    def test_inspect_defaults_to_router_container_when_runtime_absent(
        self, run_subprocess
    ):
        run_subprocess.side_effect = self._mock_statuses(
            {
                self.base.ROUTER_CONTAINER_NAME: "",
                self.base.ENVOY_CONTAINER_NAME: "",
                self.base.DASHBOARD_CONTAINER_NAME: "",
            }
        )

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

    @mock.patch.object(run_cli_tests.subprocess, "run")
    @mock.patch.object(run_cli_tests.shutil, "which")
    def test_prepare_k8s_chart_dependencies_uses_repo_native_target(
        self, which, run_subprocess
    ):
        which.side_effect = lambda name: "/usr/bin/make" if name == "make" else None
        run_subprocess.return_value = _completed_process(returncode=0)

        self.assertTrue(run_cli_tests.prepare_k8s_chart_dependencies())
        run_subprocess.assert_called_once_with(
            ["make", "helm-ci-setup"],
            cwd=run_cli_tests.REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    @mock.patch.object(run_cli_tests.subprocess, "run")
    @mock.patch.object(run_cli_tests.shutil, "which")
    def test_prepare_k8s_chart_dependencies_reports_failure(
        self, which, run_subprocess
    ):
        which.side_effect = lambda name: "/usr/bin/make" if name == "make" else None
        run_subprocess.return_value = _completed_process(
            returncode=1,
            stdout="nope",
            stderr="broken",
        )

        self.assertFalse(run_cli_tests.prepare_k8s_chart_dependencies())


class TestSharedSuiteK8sPortForwardContract(unittest.TestCase):
    def setUp(self):
        self.base = test_shared_suites.SharedRuntimeIntegrationBase(
            methodName="runTest"
        )
        self.base.test_dir = os.getcwd()
        self.base._port_forwards = []

    @mock.patch.object(test_shared_suites.subprocess, "Popen")
    @mock.patch.object(test_shared_suites.socket, "socket")
    def test_router_port_forward_targets_envoy_service_listener(
        self, socket_ctor, popen_mock
    ):
        fake_process = mock.Mock()
        fake_process.poll.return_value = None
        popen_mock.return_value = fake_process

        fake_socket = mock.MagicMock()
        fake_socket.__enter__.return_value.connect_ex.return_value = 0
        socket_ctor.return_value = fake_socket

        self.base._start_port_forward("service/semantic-router-envoy", 8888, 8888)

        popen_mock.assert_called_once()
        command = popen_mock.call_args.args[0]
        self.assertIn("service/semantic-router-envoy", command)
        self.assertIn("8888:8888", command)

    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "_wait_for_http_json"
    )
    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "_start_port_forward"
    )
    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase,
        "_available_port",
        return_value=18080,
    )
    @mock.patch.object(test_shared_suites.SharedRuntimeIntegrationBase, "_run_checked")
    @mock.patch.object(test_shared_suites.subprocess, "run")
    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase,
        "_integration_target",
        return_value="k8s",
    )
    def test_k8s_llm_katan_fixture_waits_for_service_models(
        self,
        _target,
        subprocess_run,
        run_checked,
        available_port,
        start_port_forward,
        wait_http,
    ):
        subprocess_run.return_value = _completed_process(returncode=0)

        self.base._start_llm_katan_fixture()

        subprocess_run.assert_called_once()
        self.assertEqual(run_checked.call_count, 2)
        available_port.assert_called_once()
        start_port_forward.assert_called_once_with(
            "service/llm-katan",
            8000,
            18080,
            namespace="default",
        )
        wait_http.assert_called_once_with(
            "http://localhost:18080/v1/models",
            timeout=120,
        )


class TestSharedSuiteRuntimeShape(unittest.TestCase):
    def setUp(self):
        self.routing_base = test_shared_suites.SharedRuntimeIntegrationBase(
            methodName="runTest"
        )
        self.routing_base.test_dir = os.getcwd()
        self.routing_base._port_forwards = []
        self.routing_base._llm_katan_host_port = 8000

        self.dashboard_base = test_shared_suites.TestSharedDashboardSuite(
            methodName="runTest"
        )
        self.dashboard_base.test_dir = os.getcwd()
        self.dashboard_base._port_forwards = []
        self.dashboard_base._llm_katan_host_port = 8000

    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "_wait_for_docker_runtime"
    )
    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "_wait_for_runtime_contracts"
    )
    @mock.patch.object(test_shared_suites.subprocess, "Popen")
    def test_routing_suite_uses_minimal_docker_runtime(
        self, popen_mock, _wait_contracts, _wait_runtime
    ):
        popen_mock.return_value = mock.Mock(poll=lambda: None)
        self.routing_base._start_runtime()

        command = popen_mock.call_args.args[0]
        self.assertIn("--minimal", command)
        self.assertNotIn("--profile", command)

    @mock.patch.object(test_shared_suites.SharedRuntimeIntegrationBase, "run_cli")
    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase,
        "_integration_target",
        return_value="k8s",
    )
    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "_wait_for_runtime_contracts"
    )
    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "_start_port_forward"
    )
    def test_dashboard_suite_keeps_k8s_dev_profile(
        self, start_port_forward, _wait_contracts, _target, run_cli
    ):
        run_cli.return_value = (0, "", "")
        self.dashboard_base._start_runtime()

        command = run_cli.call_args.args[0]
        self.assertIn("--profile", command)
        self.assertIn("dev", command)
        start_port_forward.assert_any_call("service/semantic-router-envoy", 8888, 8888)
        start_port_forward.assert_any_call(
            "service/semantic-router-dashboard", 8700, 8700
        )

    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "wait_for_health"
    )
    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "_wait_for_http_json"
    )
    def test_runtime_contract_waits_on_public_models_surface(
        self, wait_http, wait_health
    ):
        self.routing_base._wait_for_runtime_contracts()

        wait_health.assert_not_called()
        wait_http.assert_called_once_with(
            "http://localhost:8888/v1/models",
            timeout=test_shared_suites.RUNTIME_READY_TIMEOUT,
        )

    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "wait_for_health"
    )
    @mock.patch.object(
        test_shared_suites.SharedRuntimeIntegrationBase, "_wait_for_http_json"
    )
    def test_dashboard_runtime_contract_adds_dashboard_health(
        self, wait_http, wait_health
    ):
        self.dashboard_base._wait_for_runtime_contracts()

        wait_health.assert_not_called()
        self.assertEqual(wait_http.call_count, 2)
        wait_http.assert_any_call(
            "http://localhost:8888/v1/models",
            timeout=test_shared_suites.RUNTIME_READY_TIMEOUT,
        )
        wait_http.assert_any_call(
            "http://localhost:8700/healthz",
            timeout=test_shared_suites.RUNTIME_READY_TIMEOUT,
        )


if __name__ == "__main__":
    unittest.main()
