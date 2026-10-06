"""Process completion helpers do not require the live local stack."""

import subprocess
import unittest
from unittest import mock

from serve_session import ServeSessionMixin, startup_diagnostics


class TestServeSession(ServeSessionMixin, unittest.TestCase):
    HEALTH_CHECK_TIMEOUT = 1

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

    def test_startup_failure_keeps_both_stream_tails_and_exit_code(self):
        process = mock.Mock(spec=subprocess.Popen)
        process.communicate.return_value = (
            "starting\n" * 1000 + "docker failed to bind port",
            "initializing\n" * 1000 + "runtime startup aborted",
        )
        process.returncode = 17

        with self.assertRaises(AssertionError) as failure:
            self._wait_for_serve_success(process)

        message = str(failure.exception)
        self.assertIn("Serve exit code: 17", message)
        self.assertIn("stdout:\n[earlier output omitted]", message)
        self.assertIn("docker failed to bind port", message)
        self.assertIn("stderr:\n[earlier output omitted]", message)
        self.assertIn("runtime startup aborted", message)
        self.assertLess(len(message), 8500)

    def test_startup_diagnostics_redacts_before_truncation(self):
        token = "secret-canary-" + "x" * 5000
        message = startup_diagnostics(
            "startup " + token,
            "failed " + token,
            1,
            (token, "secret-canary-", ""),
        )

        self.assertNotIn("secret-canary", message)
        self.assertNotIn("x" * 100, message)
        self.assertIn("stdout:\nstartup [redacted]", message)
        self.assertIn("stderr:\nfailed [redacted]", message)

    def test_wait_for_serve_success_terminates_and_drains_a_timeout(self):
        process = mock.Mock(spec=subprocess.Popen)
        process.returncode = -15
        process.communicate.side_effect = [
            subprocess.TimeoutExpired("vllm-sr serve", self.HEALTH_CHECK_TIMEOUT),
            ("", "stopped after timeout"),
        ]

        with self.assertRaisesRegex(AssertionError, "before the timeout"):
            self._wait_for_serve_success(process)

        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        self.assertEqual(process.communicate.call_count, 2)
