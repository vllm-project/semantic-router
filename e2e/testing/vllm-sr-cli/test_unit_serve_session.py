"""Process completion helpers do not require the live local stack."""

import subprocess
import unittest
from unittest import mock

from serve_session import ServeSessionMixin


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
