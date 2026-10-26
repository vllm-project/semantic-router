from __future__ import annotations

import re
import subprocess
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRY_SH = REPO_ROOT / "tools" / "ci" / "retry.sh"


def run_snippet(body: str) -> subprocess.CompletedProcess[str]:
    """Run a bash snippet with tools/ci/retry.sh sourced, as a workflow step would."""
    script = "\n".join(
        [
            "set -euo pipefail",
            f'source "{RETRY_SH}"',
            body,
        ]
    )
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def fields(stdout: str, name: str) -> str:
    match = re.search(rf"^{name}=(.+)$", stdout, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{name} not reported in output:\n{stdout}")
    return match.group(1)


class RetryRunTests(unittest.TestCase):
    def test_success_on_the_first_attempt_does_not_retry(self) -> None:
        result = run_snippet(
            """
calls=0
succeeds() { calls=$((calls + 1)); return 0; }
retry_run 3 0 succeeds
echo "calls=${calls}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fields(result.stdout, "calls"), "1")
        self.assertNotIn("retrying", result.stderr)

    def test_retries_until_the_command_succeeds(self) -> None:
        result = run_snippet(
            """
calls=0
flaky() { calls=$((calls + 1)); [[ "${calls}" -ge 3 ]]; }
retry_run 5 0 flaky
echo "calls=${calls}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fields(result.stdout, "calls"), "3")
        self.assertIn("succeeded on attempt 3/5", result.stdout)
        self.assertEqual(result.stderr.count("::warning::"), 2)

    def test_exhausting_attempts_fails_and_reports_every_try(self) -> None:
        result = run_snippet(
            """
calls=0
always_fails() { calls=$((calls + 1)); return 7; }
retry_run 3 0 always_fails || echo "status=$?"
echo "calls=${calls}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fields(result.stdout, "status"), "1")
        self.assertEqual(fields(result.stdout, "calls"), "3")
        # The last attempt is reported as an error, the earlier ones as warnings.
        self.assertEqual(result.stderr.count("::warning::"), 2)
        self.assertEqual(result.stderr.count("::error::"), 1)

    def test_cleanup_runs_before_every_retry_but_not_the_first_attempt(self) -> None:
        result = run_snippet(
            """
calls=0
cleanups=0
always_fails() { calls=$((calls + 1)); return 1; }
cleanup() { cleanups=$((cleanups + 1)); return 0; }
RETRY_CLEANUP=cleanup retry_run 3 0 always_fails || true
echo "calls=${calls}"
echo "cleanups=${cleanups}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fields(result.stdout, "calls"), "3")
        self.assertEqual(fields(result.stdout, "cleanups"), "2")

    def test_a_failing_cleanup_does_not_abort_the_retry_loop(self) -> None:
        result = run_snippet(
            """
calls=0
always_fails() { calls=$((calls + 1)); return 1; }
bad_cleanup() { return 1; }
RETRY_CLEANUP=bad_cleanup retry_run 3 0 always_fails || true
echo "calls=${calls}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fields(result.stdout, "calls"), "3")
        self.assertIn("cleanup", result.stderr.lower())

    def test_the_delay_is_observed_between_attempts(self) -> None:
        started = time.monotonic()
        result = run_snippet(
            """
always_fails() { return 1; }
retry_run 2 1 always_fails || true
"""
        )
        elapsed = time.monotonic() - started
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(elapsed, 0.9)

    def test_a_chained_function_retries_when_an_early_command_fails(self) -> None:
        result = run_snippet(
            """
calls=0
install() {
  calls=$((calls + 1))
  false && echo "the rest must not run"
}
retry_run 2 0 install || true
echo "calls=${calls}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fields(result.stdout, "calls"), "2")
        self.assertIn("retrying", result.stderr)

    def test_an_unchained_function_reports_only_its_last_command(self) -> None:
        # bash suspends errexit inside the `if` that retry_run uses, so a
        # function without `&&` reports the status of its last command and the
        # retry never fires. This is why the workflow functions chain.
        result = run_snippet(
            """
calls=0
unchained() {
  calls=$((calls + 1))
  false
  true
}
retry_run 3 0 unchained
echo "calls=${calls}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fields(result.stdout, "calls"), "1")
        self.assertNotIn("retrying", result.stderr)

    def test_a_caller_variable_cannot_rewrite_the_retry_loop(self) -> None:
        # bash resolves a variable to the nearest binding in the dynamic scope,
        # so a command assigning a name retry_run also uses would otherwise
        # change the attempt limit while the loop is running.
        result = run_snippet(
            """
attempt=0
attempts=0
delay=0
tracks() {
  attempt=$((attempt + 1))
  attempts=99
  delay=99
  return 1
}
retry_run 2 0 tracks || true
echo "calls=${attempt}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fields(result.stdout, "calls"), "2")
        self.assertEqual(result.stderr.count("::warning::"), 1)

    def test_invalid_arguments_are_rejected_before_running_anything(self) -> None:
        result = run_snippet(
            """
calls=0
tracks() { calls=$((calls + 1)); return 0; }
retry_run 0 5 tracks || echo "zero=$?"
retry_run 2 -1 tracks || echo "negative=$?"
retry_run 1 x tracks || echo "nonnumeric=$?"
retry_run 1 5 || echo "missing=$?"
echo "calls=${calls}"
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(fields(result.stdout, "zero"), "2")
        self.assertEqual(fields(result.stdout, "negative"), "2")
        self.assertEqual(fields(result.stdout, "nonnumeric"), "2")
        self.assertEqual(fields(result.stdout, "missing"), "2")
        self.assertEqual(fields(result.stdout, "calls"), "0")


if __name__ == "__main__":
    unittest.main()
