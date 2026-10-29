"""Tests for how the lifecycle GitHub client waits out a transient read failure."""

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import community_lifecycle_github as lifecycle


def failed(stderr: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


def succeeded(stdout: str = "{}") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TransientReadFailureTests(unittest.TestCase):
    def call(
        self,
        results: list[subprocess.CompletedProcess[str]],
        endpoint: str = "repos/o/r/pulls",
        **options: object,
    ) -> tuple[object, int, list[int]]:
        client = lifecycle.GitHubClient()
        output = io.StringIO()
        with (
            patch.object(lifecycle.subprocess, "run", side_effect=results) as run,
            patch("time.sleep") as sleep,
            patch("sys.stderr"),
            contextlib.redirect_stdout(output),
        ):
            try:
                value = client.request(endpoint, **options)
            except SystemExit as exit_request:
                value = exit_request.code
        self.output = output.getvalue()
        return value, run.call_count, [call.args[0] for call in sleep.call_args_list]

    def test_a_read_waits_out_both_forms_of_a_gateway_error(self) -> None:
        for stderr in ("gh: HTTP 503\n", "gh: Bad Gateway (HTTP 502)\n"):
            with self.subTest(stderr=stderr):
                value, attempts, delays = self.call(
                    [failed(stderr), failed(stderr), succeeded('{"ok": true}')]
                )
                self.assertEqual(value, {"ok": True})
                self.assertEqual((attempts, delays), (3, [5, 15]))
                self.assertIn(
                    "::warning::GitHub returned HTTP 50", self.output.splitlines()[0]
                )

    def test_a_read_gives_up_after_the_last_delay_and_keeps_the_exit_code(self) -> None:
        value, attempts, delays = self.call([failed("gh: HTTP 504\n")] * 4)
        self.assertEqual((value, attempts, delays), (1, 4, [5, 15, 45]))

    def test_the_budget_caps_the_total_wait_of_one_client(self) -> None:
        client = lifecycle.GitHubClient()
        client.retry_budget_seconds = 10
        with (
            patch.object(
                lifecycle.subprocess, "run", side_effect=[failed("gh: HTTP 503\n")] * 4
            ) as run,
            patch("time.sleep") as sleep,
            patch("sys.stderr"),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            client.request("repos/o/r/pulls")
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(5)

    def test_only_a_5xx_on_a_read_is_retried(self) -> None:
        for method, stderr in (
            ("POST", "gh: HTTP 503\n"),
            ("DELETE", "gh: Bad Gateway (HTTP 502)\n"),
            ("GET", "gh: API rate limit exceeded for installation. (HTTP 403)\n"),
            ("GET", "gh: Not Found (HTTP 404)\n"),
            ("GET", "gh: SECRET HTTP 503\n"),
            ("GET", "gh: HTTP 5030\n"),
            ("GET", "gh: Server Error (HTTP 503)\ngh: trailing text\n"),
        ):
            with self.subTest(method=method, stderr=stderr):
                value, attempts, delays = self.call([failed(stderr)], method=method)
                self.assertEqual((value, attempts, delays), (1, 1, []))

    def test_a_graphql_query_is_a_read_and_a_mutation_is_not(self) -> None:
        query = {"query": "query ($n: Int!) { repository { id } }"}
        mutation = {"query": "mutation { addLabelsToLabelable(input: {}) { id } }"}
        for payload, expected in ((query, (3, [5, 15])), (mutation, (1, []))):
            with self.subTest(payload=payload):
                _, attempts, delays = self.call(
                    [failed("gh: HTTP 502\n"), failed("gh: HTTP 502\n"), succeeded()],
                    endpoint="graphql",
                    method="POST",
                    payload=payload,
                )
                self.assertEqual((attempts, delays), expected)

    def test_a_missing_resource_is_still_ignored_on_request(self) -> None:
        value, attempts, delays = self.call(
            [failed("gh: Not Found (HTTP 404)\n")], ignore_not_found=True
        )
        self.assertEqual((value, attempts, delays), (None, 1, []))


if __name__ == "__main__":
    unittest.main()
