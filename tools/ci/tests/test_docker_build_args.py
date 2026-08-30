from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "ci" / "docker-build-args.sh"


def run_resolver(**overrides: str) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "github-output"
        env = os.environ | {
            "GITHUB_OUTPUT": str(output),
            "MATRIX_IMAGE": "dashboard",
            "CARGO_BUILD_JOBS": "8",
        }
        env.update(overrides)
        subprocess.run(["bash", str(SCRIPT)], cwd=REPO_ROOT, env=env, check=True)
        return output.read_text(encoding="utf-8")


class DockerBuildArgumentTests(unittest.TestCase):
    def test_release_dashboard_uses_stable_tag(self) -> None:
        output = run_resolver(
            DASHBOARD_VERSION_MODE="release",
            RELEASE_TAG="v0.3.0",
        )
        self.assertIn("DASHBOARD_VERSION=v0.3.0", output)
        self.assertRegex(output, r"VLLM_SR_SOURCE_REVISION=[0-9a-f]{40}\n")

    def test_nightly_dashboard_uses_explicit_date(self) -> None:
        output = run_resolver(
            DASHBOARD_VERSION_MODE="publish",
            IS_NIGHTLY="true",
            NIGHTLY_DATE="20260806",
        )
        self.assertIn("DASHBOARD_VERSION=v0.3.0-nightly.20260806.", output)

    def test_dashboard_source_revision_is_the_full_git_commit(self) -> None:
        output = run_resolver(
            DASHBOARD_VERSION_MODE="release",
            RELEASE_TAG="v0.3.0",
        )
        expected = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        match = re.search(r"^VLLM_SR_SOURCE_REVISION=(.+)$", output, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), expected)


if __name__ == "__main__":
    unittest.main()
