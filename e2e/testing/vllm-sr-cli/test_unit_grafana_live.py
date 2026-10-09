#!/usr/bin/env python3
"""Unit coverage for Dashboard login synchronization in the Live regression."""

import http.client
import unittest
from types import SimpleNamespace
from unittest import mock

import test_integration_grafana_live as grafana_live


class TestDashboardLoginReadiness(unittest.TestCase):
    def setUp(self):
        self.case = grafana_live.TestGrafanaLiveIntegration(methodName="runTest")
        self.case.runtime_stack = SimpleNamespace(dashboard_port=12900)
        self.now = 0.0
        self.clock = mock.patch.object(
            grafana_live.time, "monotonic", side_effect=lambda: self.now
        )
        self.sleeper = mock.patch.object(
            grafana_live.time, "sleep", side_effect=self._advance
        )
        self.clock.start()
        self.sleep = self.sleeper.start()
        self.addCleanup(self.clock.stop)
        self.addCleanup(self.sleeper.stop)

    def _advance(self, seconds):
        self.now += seconds

    def test_transient_connection_failures_retry_then_return_token(self):
        for error in (
            ConnectionRefusedError("starting"),
            ConnectionResetError("starting"),
            TimeoutError("starting"),
            http.client.RemoteDisconnected("starting"),
        ):
            with self.subTest(error=type(error).__name__):
                failed, ready = mock.Mock(), mock.Mock()
                failed.getresponse.side_effect = error
                ready.getresponse.return_value.status = 200
                ready.getresponse.return_value.read.return_value = b'{"token":"ready"}'
                with mock.patch.object(
                    grafana_live.http.client,
                    "HTTPConnection",
                    side_effect=[failed, ready],
                ) as connect:
                    self.assertEqual(self.case._dashboard_session_token(), "ready")
                self.assertEqual(connect.call_count, 2)
                failed.close.assert_called_once()
                ready.close.assert_called_once()
                ready.request.assert_called_once_with(
                    "POST",
                    "/api/auth/login",
                    body=mock.ANY,
                    headers={"Content-Type": "application/json"},
                )

    def test_persistent_connection_failure_exhausts_bounded_budget(self):
        connection = mock.Mock()
        error = ConnectionResetError("still starting")

        def reset_connection():
            self._advance(1)
            raise error

        connection.getresponse.side_effect = reset_connection
        with mock.patch.object(
            grafana_live.http.client, "HTTPConnection", return_value=connection
        ) as connect, self.assertRaisesRegex(TimeoutError, "Dashboard login") as caught:
            self.case._dashboard_session_token(timeout=5.5)
        self.assertIs(caught.exception.__cause__, error)
        self.assertEqual(self.now, 5.5)
        self.assertEqual(
            connect.call_args_list,
            [
                mock.call("127.0.0.1", 12900, timeout=5.5),
                mock.call("127.0.0.1", 12900, timeout=2.5),
            ],
        )
        self.assertEqual(self.sleep.call_args_list, [mock.call(2), mock.call(1.5)])
        self.assertEqual(connection.close.call_count, 2)

    def test_authentication_failures_are_not_retried(self):
        for status in (401, 403):
            with self.subTest(status=status):
                connection = mock.Mock()
                connection.getresponse.return_value.status = status
                connection.getresponse.return_value.read.return_value = b"unauthorized"
                with mock.patch.object(
                    grafana_live.http.client, "HTTPConnection", return_value=connection
                ) as connect, self.assertRaisesRegex(AssertionError, "unauthorized"):
                    self.case._dashboard_session_token()
                connect.assert_called_once()
                connection.close.assert_called_once()
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
