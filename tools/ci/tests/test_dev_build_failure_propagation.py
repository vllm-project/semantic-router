import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


class DevBuildFailurePropagationTest(unittest.TestCase):
    def test_agent_dev_propagates_router_image_failure(self) -> None:
        result = self._run_dev_target(
            """
case "$1" in
  rm) exit 0 ;;
  build) exit 37 ;;
  *) exit 0 ;;
esac
""",
            topology="integrated",
            target="agent-dev",
        )

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Router image built:", result.stdout)
        self.assertNotIn("Development Setup Complete", result.stdout)

    def test_router_image_failure_stops_development_setup(self) -> None:
        result = self._run_dev_target(
            """
case "$1" in
  rm) exit 0 ;;
  build) exit 37 ;;
  *) exit 0 ;;
esac
""",
            topology="integrated",
        )

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Router image built:", result.stdout)
        self.assertNotIn("Development Setup Complete", result.stdout)

    def test_dashboard_image_failure_stops_split_development_setup(self) -> None:
        result = self._run_dev_target(
            """
case "$1" in
  rm) exit 0 ;;
  image) exit 0 ;;
  build)
    case "$*" in
      *dashboard/backend/Dockerfile*) exit 43 ;;
      *) exit 0 ;;
    esac
    ;;
  *) exit 0 ;;
esac
""",
            topology="split",
        )

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Router image built:", result.stdout)
        self.assertNotIn("Dashboard image built:", result.stdout)
        self.assertNotIn("Development Setup Complete", result.stdout)

    def test_envoy_pull_failure_stops_split_development_setup(self) -> None:
        result = self._run_dev_target(
            """
case "$1" in
  rm) exit 0 ;;
  build) exit 0 ;;
  image) exit 1 ;;
  pull) exit 47 ;;
  *) exit 0 ;;
esac
""",
            topology="split",
        )

        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Router image built:", result.stdout)
        self.assertNotIn("Envoy image available:", result.stdout)
        self.assertNotIn("Dashboard image built:", result.stdout)
        self.assertNotIn("Development Setup Complete", result.stdout)

    def _run_dev_target(
        self,
        runtime_body: str,
        *,
        topology: str,
        target: str = "vllm-sr-dev",
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "fake-container-runtime"
            runtime.write_text("#!/bin/sh\n" + runtime_body)
            runtime.chmod(runtime.stat().st_mode | stat.S_IXUSR)
            environment = dict(os.environ)
            environment["PATH"] = os.environ.get("PATH", "")
            return subprocess.run(
                [
                    "make",
                    "--no-print-directory",
                    target,
                    f"CONTAINER_RUNTIME={runtime}",
                    f"VLLM_SR_TOPOLOGY={topology}",
                    "VLLM_SR_PLATFORM=amd",
                    "ENV=amd",
                    "VLLM_SR_SOURCE_REVISION=test-source-revision",
                    "VLLM_SR_DASHBOARD_VERSION=test-dashboard-version",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )


if __name__ == "__main__":
    unittest.main()
