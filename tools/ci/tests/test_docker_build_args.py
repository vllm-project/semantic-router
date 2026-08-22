from __future__ import annotations

import os
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

    def test_nightly_dashboard_uses_explicit_date(self) -> None:
        output = run_resolver(
            DASHBOARD_VERSION_MODE="publish",
            IS_NIGHTLY="true",
            NIGHTLY_DATE="20260806",
        )
        self.assertIn("DASHBOARD_VERSION=v0.3.0-nightly.20260806.", output)


if __name__ == "__main__":
    unittest.main()
