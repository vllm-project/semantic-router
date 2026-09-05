from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "ci" / "verify_published_dashboard_image.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_published_dashboard_image", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def completed(
    command: list[str], stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, "")


class Response:
    status = 200

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"status":"healthy","service":"semantic-router-dashboard"}'


class PublishedDashboardImageTests(unittest.TestCase):
    def test_manifest_platforms_extracts_oci_index_platforms(self) -> None:
        raw = """{
          "manifests": [
            {"platform": {"os": "linux", "architecture": "amd64"}},
            {"platform": {"os": "linux", "architecture": "arm64"}},
            {"platform": {"os": "unknown", "architecture": "unknown"}}
          ]
        }"""

        self.assertEqual(
            verifier.manifest_platforms(raw),
            {"linux/amd64", "linux/arm64", "unknown/unknown"},
        )

    def test_verify_manifest_rejects_a_missing_arm64_variant(self) -> None:
        raw = '{"manifests":[{"platform":{"os":"linux","architecture":"amd64"}}]}'
        with (
            mock.patch.object(verifier, "run_command", return_value=completed([], raw)),
            self.assertRaisesRegex(verifier.VerificationError, "linux/arm64"),
        ):
            verifier.verify_manifest("example.invalid/dashboard@sha256:" + "a" * 64)

    def test_health_response_requires_dashboard_identity(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "service"):
            verifier.validate_health_response(200, b'{"status":"healthy"}')

    def test_arm64_runtime_uses_platform_and_always_removes_container(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
            del check
            calls.append(command)
            if command[1] == "port":
                return completed(command, "127.0.0.1:49152\n")
            if command[1] == "inspect":
                return completed(command, "true\n")
            return completed(command)

        with (
            mock.patch.object(verifier, "run_command", side_effect=fake_run),
            mock.patch.object(
                verifier.urllib.request, "urlopen", return_value=Response()
            ),
        ):
            verifier.verify_runtime(
                "example.invalid/dashboard@sha256:" + "a" * 64, "linux/arm64"
            )

        run = next(command for command in calls if command[1] == "run")
        self.assertIn("linux/arm64", run)
        self.assertIn("127.0.0.1::8700", run)
        self.assertEqual(calls[-1][1:3], ["rm", "--force"])

    def test_publish_workflow_verifies_each_dashboard_publication_by_digest(
        self,
    ) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "docker-publish.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = workflow["jobs"]["publish"]["steps"]
        build = next(
            step for step in steps if step.get("name") == "Build and publish image"
        )
        verify = next(
            step
            for step in steps
            if step.get("name") == "Verify published Dashboard image"
        )

        self.assertEqual(build["id"], "publish")
        self.assertEqual(verify["if"], "matrix.image == 'dashboard'")
        self.assertEqual(
            verify["env"]["IMAGE_REF"],
            "${{ steps.tags.outputs.image }}@${{ steps.publish.outputs.digest }}",
        )
        self.assertEqual(
            verify["run"],
            'python3 tools/ci/verify_published_dashboard_image.py "$IMAGE_REF"',
        )


if __name__ == "__main__":
    unittest.main()
