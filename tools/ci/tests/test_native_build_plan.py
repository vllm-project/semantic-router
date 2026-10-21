"""Router builds use explicit, public RISC-V SDK inputs."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_image_artifacts import archive_image

SCRIPT = Path(__file__).resolve().parents[2] / "docker/riscv64/build.py"
sys.path.insert(0, str(SCRIPT.parents[2] / "ci"))
import image_artifacts as images  # noqa: E402
from classify_pr_changes import classify  # noqa: E402


class NativeBuildPlanTests(unittest.TestCase):
    def test_native_build_sources_select_harness_contracts(self):
        for source in (
            "tools/docker/riscv64/build.py",
            "tools/docker/riscv64/Dockerfile",
            "tools/docker/Dockerfile.attached",
            "tools/test/smoke/riscv64-standalone.py",
        ):
            with self.subTest(source=source):
                self.assertIn("harness-tools", classify([source]).selected_jobs)

    def test_artifact_source_requires_clean_matching_checkout(self):
        module = self.native_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            tracked = root / "input"
            tracked.write_text("original\n")
            subprocess.run(["git", "-C", str(root), "add", "input"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Build Test",
                    "-c",
                    "user.email=build@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--quiet",
                    "-m",
                    "fixture",
                ],
                check=True,
            )
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            with patch.object(module, "ROOT", root):
                guard = getattr(module, "require_clean_source", None)
                self.assertIsNotNone(
                    guard, "native artifact export needs a source guard"
                )
                self.assertEqual(guard(), revision)
                with self.assertRaisesRegex(ValueError, "revision changed"):
                    guard("0" * 40)
                tracked.write_text("modified\n")
                with self.assertRaisesRegex(ValueError, "clean"):
                    guard()
                with self.assertRaisesRegex(ValueError, "clean"):
                    module.export_artifact(
                        "router:test", root / "artifact", "pr", "test", []
                    )
                self.assertFalse((root / "artifact").exists())
                subprocess.run(["git", "-C", str(root), "add", "input"], check=True)
                with self.assertRaisesRegex(ValueError, "clean"):
                    guard()
                subprocess.run(
                    ["git", "-C", str(root), "restore", "--staged", "input"], check=True
                )
                tracked.write_text("original\n")
                (root / "extra").write_text("untracked input\n")
                with self.assertRaisesRegex(ValueError, "clean"):
                    guard()

    def native_module(self):
        spec = importlib.util.spec_from_file_location("native_build", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(
            callable(getattr(module, "export_artifact", None)),
            "native builds must produce the registered OCI artifact",
        )
        return module

    def test_native_export_is_accepted_by_shared_artifact_verification(self):
        module = self.native_module()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "artifact"

            def external(command, **kwargs):
                if command[0] == "skopeo":
                    self.assertEqual(
                        command[1:3], ["copy", "docker-daemon:router:test"]
                    )
                    self.assertEqual(
                        command[3], "oci-archive:" + str(directory / "image.tar")
                    )
                    archive_image(directory / "image.tar", ("riscv64",))
                else:
                    self.assertEqual(
                        Path(command[1]), SCRIPT.parents[2] / "ci/image_artifacts.py"
                    )
                    with patch.object(sys, "argv", command[1:]), patch.dict(
                        os.environ, kwargs["env"]
                    ):
                        images.main()

            with patch.object(
                module.subprocess, "run", side_effect=external
            ), patch.object(images, "source_sha", return_value="a" * 40), patch.object(
                module, "require_clean_source", return_value="a" * 40, create=True
            ):
                module.export_artifact(
                    "router:test", directory, "pr", "test", ["BASE_IMAGE=pinned"]
                )
                manifest = images.verify(directory, "vllm-sr-riscv64")
            self.assertEqual(manifest["images"][0]["platform"], "linux/riscv64")
            self.assertEqual(manifest["build_args"], ["BASE_IMAGE=pinned"])
            self.assertRegex(manifest["date"], r"^\d{8}$")

    def test_native_export_does_not_replace_existing_artifacts(self):
        module = self.native_module()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            original = directory / "image.tar"
            original.write_bytes(b"existing")
            with patch.object(
                module, "require_clean_source", return_value="a" * 40, create=True
            ), self.assertRaisesRegex(ValueError, "already contains"):
                module.export_artifact("router:test", directory, "pr", "test", [])
            self.assertEqual(original.read_bytes(), b"existing")

    def test_failed_native_export_does_not_seal_a_manifest(self):
        module = self.native_module()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch.object(
                module.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "skopeo"),
            ), patch.object(
                module, "require_clean_source", return_value="a" * 40, create=True
            ), self.assertRaises(
                subprocess.CalledProcessError
            ):
                module.export_artifact("router:test", directory, "pr", "test", [])
            self.assertFalse((directory / "manifest.json").exists())

    def plan(self, *args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--plan", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)["commands"]

    def test_router_constructs_sdk_then_attached_image_with_layer_cache(self):
        commands = self.plan()
        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertNotIn("--no-cache", command)
            self.assertNotIn("--cache-from", command)
            self.assertIn("TARGETARCH=riscv64", command)
            self.assertIn("TARGETOS=linux", command)
            self.assertIn("BUILDPLATFORM=linux/riscv64", command)
        self.assertIn("tools/docker/riscv64/Dockerfile.toolchain", commands[0])
        self.assertIn("tools/docker/riscv64/Dockerfile", commands[1])
        self.assertIn("router-binary", commands[1])
        self.assertIn("tools/docker/Dockerfile.attached", commands[2])
        self.assertIn("vllm-sr-attached", commands[2])
        self.assertIn("ROUTER_BUILD=semantic-router-native-router:public", commands[2])
        self.assertIn("GO_IMAGE=semantic-router-native-sdk:public", commands[1])
        self.assertIn("RUST_IMAGE=semantic-router-native-sdk:public", commands[1])

    def test_cold_build_is_explicit(self):
        for command in self.plan("--no-cache"):
            self.assertIn("--no-cache", command)

    def test_existing_sdk_skips_sdk_build(self):
        commands = self.plan("--sdk-image", "native-sdk:verified")
        self.assertEqual(len(commands), 2)
        self.assertIn("GO_IMAGE=native-sdk:verified", commands[0])
        self.assertIn("RUST_IMAGE=native-sdk:verified", commands[0])

    def test_sdk_concurrency_is_configurable(self):
        commands = self.plan("--component", "sdk", "--jobs", "4")
        self.assertIn("BUILD_JOBS=4", commands[0])

    def test_invalid_jobs_are_rejected(self):
        for jobs in ("0", "-1"):
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--plan", "--jobs", jobs],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)

    def test_sdk_only_build(self):
        self.assertEqual(len(self.plan("--component", "sdk")), 1)

    def test_pinned_base_override_applies_to_every_runtime_input(self):
        base = "mirror.example/debian@sha256:" + "a" * 64
        commands = self.plan("--base-image", base)
        self.assertIn("BASE_IMAGE=" + base, commands[0])
        self.assertIn("ATTACHED_BASE_IMAGE=" + base, commands[2])
        self.assertIn("ASSETS_IMAGE=" + base, commands[2])

    def test_mutable_base_is_rejected(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--plan", "--base-image", "debian:latest"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("digest", result.stderr)

    def test_engine_local_sdk_requires_the_docker_driver(self):
        self.assertTrue(SCRIPT.is_file(), "Router build runner is missing")
        spec = importlib.util.spec_from_file_location("native_build", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_builder("Name: default\nDriver: docker\n")
        with self.assertRaisesRegex(ValueError, "docker driver"):
            module.validate_builder("Name: isolated\nDriver: docker-container\n")


if __name__ == "__main__":
    unittest.main()
