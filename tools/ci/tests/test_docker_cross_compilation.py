from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CPU_DOCKERFILES = (
    "tools/docker/Dockerfile.extproc",
    "src/vllm-sr/Dockerfile",
)


sys.path.insert(0, str(REPO_ROOT / "tools/ci"))
from image_artifacts import DEFINITIONS  # noqa: E402


class DockerCrossCompilationTests(unittest.TestCase):
    def test_arm64_openssl_override_does_not_reach_host_build_dependencies(
        self,
    ) -> None:
        for dockerfile in CPU_DOCKERFILES:
            text = (REPO_ROOT / dockerfile).read_text().replace("\\\n", " ")
            builds = list(
                re.finditer(r"((?:[A-Z0-9_]+=[^\s;]+\s+)*)cargo build([^;&]+)", text)
            )
            arm64_builds = [
                build
                for build in builds
                if "--target aarch64-unknown-linux-gnu" in build.group(2)
                and "OPENSSL" in build.group(1)
            ]
            with self.subTest(dockerfile=dockerfile):
                self.assertTrue(arm64_builds)
                self.assertNotRegex(
                    text, r"(?m)^ENV OPENSSL_(DIR|LIB_DIR|INCLUDE_DIR)="
                )
            for build in arm64_builds:
                with self.subTest(dockerfile=dockerfile, command=build.group(0)):
                    probe = "import json, os; print(json.dumps({k: v for k, v in os.environ.items() if 'OPENSSL' in k}))"
                    environment = {
                        key: value
                        for key, value in os.environ.items()
                        if "OPENSSL" not in key
                    }
                    result = subprocess.run(
                        [
                            "bash",
                            "-e",
                            "-c",
                            f"{build.group(1)}{shlex.quote(sys.executable)} -c {shlex.quote(probe)}",
                        ],
                        env=environment,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    overrides = json.loads(result.stdout)
                    self.assertNotIn("OPENSSL_LIB_DIR", overrides)
                    self.assertEqual(
                        overrides["AARCH64_UNKNOWN_LINUX_GNU_OPENSSL_LIB_DIR"],
                        "/usr/lib/aarch64-linux-gnu",
                    )

    def test_cpu_router_validation_covers_published_architectures(self) -> None:
        for image in ("extproc", "vllm-sr"):
            with self.subTest(image=image):
                published = DEFINITIONS[image][2]
                self.assertEqual(published, ["linux/amd64", "linux/arm64"])
        builder = (REPO_ROOT / ".github/workflows/build-artifacts.yml").read_text()
        publisher = (REPO_ROOT / ".github/workflows/docker-publish.yml").read_text()
        self.assertIn("image_artifacts.py", builder)
        self.assertIn("--multiarch", builder)
        self.assertIn("image_artifacts.py promote", publisher)
        self.assertNotIn("docker/build-push-action", publisher)


if __name__ == "__main__":
    unittest.main()
