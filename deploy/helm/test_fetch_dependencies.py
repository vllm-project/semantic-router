"""The chart's dependency download retries with backoff; a cache hit skips it (offline)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/helm/fetch-dependencies.sh"
CHART = "deploy/helm/semantic-router"

# Records every call. The first $BUILD_FAILURES dependency builds and the first
# $REPO_ADD_FAILURES repository adds fail, as a failed download would.
FAKE_HELM = """#!/usr/bin/env bash
echo "helm $*" >> "$CALLS"
fail_first() {
  echo x >> "$1"
  [ "$(wc -l < "$1")" -le "$2" ]
}
case "$1 $2" in
  "repo add") if fail_first "$STATE/adds" "${REPO_ADD_FAILURES:-0}"; then exit 1; fi ;;
  "dependency build") if fail_first "$STATE/builds" "${BUILD_FAILURES:-0}"; then exit 1; fi ;;
esac
"""
FAKE_SLEEP = """#!/usr/bin/env bash
echo "sleep $*" >> "$CALLS"
"""


class FetchDependenciesTests(unittest.TestCase):
    def fetch(self, **env: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, body in (("helm", FAKE_HELM), ("sleep", FAKE_SLEEP)):
                (root / name).write_text(body, encoding="utf-8")
                (root / name).chmod(0o755)
            calls = root / "calls"
            calls.touch()
            result = subprocess.run(
                [str(SCRIPT), CHART],
                cwd=ROOT,
                env={
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "CALLS": str(calls),
                    "STATE": str(root),
                    "GITHUB_ACTIONS": "",
                    **env,
                },
                text=True,
                capture_output=True,
                check=False,
            )
            return result, calls.read_text(encoding="utf-8").splitlines()

    def test_retries_with_doubling_backoff_until_a_download_succeeds(self) -> None:
        result, calls = self.fetch(BUILD_FAILURES="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [call for call in calls if call.startswith("sleep")],
            ["sleep 10", "sleep 20"],
        )
        builds = [call for call in calls if call.startswith("helm dependency build")]
        self.assertEqual(builds, [f"helm dependency build {CHART} --skip-refresh"] * 3)
        self.assertEqual(sum(call.startswith("helm repo add") for call in calls), 15)
        self.assertIn("attempt 2 of 4", result.stderr)

    def test_fails_with_a_clear_error_after_the_last_attempt(self) -> None:
        result, calls = self.fetch(BUILD_FAILURES="9", GITHUB_ACTIONS="true")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            [call for call in calls if call.startswith("sleep")],
            ["sleep 10", "sleep 20", "sleep 40"],
        )
        self.assertEqual(
            sum(call.startswith("helm dependency build") for call in calls), 4
        )
        last = result.stderr.strip().splitlines()[-1]
        self.assertTrue(last.startswith("::error::"), last)
        self.assertIn("after 4 attempts", last)

    def test_a_failed_repository_add_is_retried(self) -> None:
        result, calls = self.fetch(REPO_ADD_FAILURES="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [call for call in calls if call.startswith("sleep")], ["sleep 10"]
        )
        self.assertEqual(
            sum(call.startswith("helm dependency build") for call in calls), 1
        )

    def test_a_cache_hit_downloads_nothing(self) -> None:
        result, calls = self.fetch(HELM_DEPENDENCIES_CACHED="true", BUILD_FAILURES="9")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_configured_repositories_are_not_added_again(self) -> None:
        result, calls = self.fetch(HELM_REPO_UPDATE="false")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [f"helm dependency build {CHART} --skip-refresh"])


if __name__ == "__main__":
    unittest.main()
