#!/usr/bin/env python3
"""
test_integration_docker_smoke.py - Exercise the llm-katan Docker smoke target.

The container lifecycle itself (detached start, readiness, cleanup) belongs to
`make docker-test-llm-katan`. This test only proves that target is green on a
real container host, which a static Makefile assertion cannot show.
"""

import os
import unittest
from pathlib import Path

from cli_test_base import CLITestBase

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_SMOKE_TARGET = "docker-test-llm-katan"
# Off the image's conventional 8000 so the smoke container cannot share a host
# port with a stack member that happens to use it.
DOCKER_SMOKE_HOST_PORT = os.environ.get("LLM_KATAN_HOST_PORT", "18081")
# The target builds the image first, and its Dockerfile installs torch.
DOCKER_SMOKE_TIMEOUT = 900


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true to enable.",
)
class TestDockerSmokeIntegration(CLITestBase):
    """Integration coverage for the repository's Docker smoke target.

    The gate sits on the class so the unit path never reaches CLITestBase
    .setUpClass, which resolves the runtime stack and probes for a container
    runtime.
    """

    def test_docker_test_llm_katan_target_passes(self):
        """`make docker-test-llm-katan` succeeds on the active container host."""
        self.print_test_header(
            "llm-katan Docker Smoke Test",
            "Runs the Make target that owns startup, readiness, and cleanup",
        )

        result = self._run_subprocess(
            [
                "make",
                DOCKER_SMOKE_TARGET,
                f"LLM_KATAN_HOST_PORT={DOCKER_SMOKE_HOST_PORT}",
            ],
            timeout=DOCKER_SMOKE_TIMEOUT,
            cwd=str(REPO_ROOT),
        )

        self.assertEqual(
            result.returncode,
            0,
            f"{DOCKER_SMOKE_TARGET} failed: rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
