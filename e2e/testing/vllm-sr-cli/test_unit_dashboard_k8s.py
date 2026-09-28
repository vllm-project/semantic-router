#!/usr/bin/env python3
"""Unit coverage for the Dashboard address the CLI reports for Kubernetes."""

import subprocess
import unittest
from unittest import mock

from cli.commands import runtime
from cli.k8s_backend import K8sBackend
from click.testing import CliRunner


class _StubBackend:
    """Backend double that reports a Dashboard the caller cannot open."""

    def __init__(self, url, port_forward):
        self.url = url
        self.port_forward = port_forward

    def is_running(self):
        return True

    def get_dashboard_url(self):
        return self.url

    def get_dashboard_port_forward(self):
        return self.port_forward


def _backend(*, stdout, returncode=0):
    backend = K8sBackend.__new__(K8sBackend)
    backend.namespace = "test-ns"
    backend.context = "kind-test"
    backend.release_name = "router-b"
    completed = subprocess.CompletedProcess([], returncode, stdout=stdout)
    return backend, mock.patch.object(subprocess, "run", return_value=completed)


class TestDashboardPortForward(unittest.TestCase):
    """The Kubernetes Dashboard address is a ClusterIP, so a command is needed."""

    def test_port_forward_names_the_service_the_cluster_reports(self):
        backend, patched = _backend(stdout="router-b-semantic-router-dashboard:8700")
        with patched:
            self.assertEqual(
                backend.get_dashboard_port_forward(),
                "kubectl --context kind-test port-forward --namespace test-ns "
                "svc/router-b-semantic-router-dashboard 8700:8700",
            )

    def test_port_forward_is_absent_without_a_dashboard_service(self):
        backend, patched = _backend(stdout="")
        with patched:
            self.assertIsNone(backend.get_dashboard_port_forward())

    def test_k8s_target_prints_the_command_and_opens_no_browser(self):
        stub = _StubBackend(
            "http://10.96.0.5:8700",
            "kubectl port-forward --namespace ns svc/sr-dashboard 8700:8700",
        )
        opened = []
        with (
            mock.patch.object(runtime, "_build_backend", return_value=stub),
            mock.patch.object(runtime.webbrowser, "open", opened.append),
        ):
            result = CliRunner().invoke(runtime.dashboard, ["--target", "k8s"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(opened, [])
        self.assertIn("http://10.96.0.5:8700", result.stdout)
        self.assertIn("svc/sr-dashboard 8700:8700", result.stdout)


if __name__ == "__main__":
    unittest.main()
