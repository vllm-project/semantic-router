"""Router builds use explicit, public RISC-V SDK inputs."""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "docker/riscv64/build.py"


class NativeBuildPlanTests(unittest.TestCase):
    def plan(self, *args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--plan", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)["commands"]

    def test_router_constructs_sdk_then_attached_image_without_cache_imports(self):
        commands = self.plan()
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertIn("--no-cache", command)
            self.assertNotIn("--cache-from", command)
            self.assertIn("TARGETARCH=riscv64", command)
            self.assertIn("TARGETOS=linux", command)
            self.assertIn("BUILDPLATFORM=linux/riscv64", command)
        self.assertIn("tools/docker/riscv64/Dockerfile.toolchain", commands[0])
        self.assertIn("vllm-sr-attached", commands[1])
        self.assertIn("GO_IMAGE=semantic-router-native-sdk:public", commands[1])
        self.assertIn("RUST_IMAGE=semantic-router-native-sdk:public", commands[1])

    def test_sdk_only_build(self):
        self.assertEqual(len(self.plan("--component", "sdk")), 1)

    def test_pinned_base_override_applies_to_every_runtime_input(self):
        base = "mirror.example/debian@sha256:" + "a" * 64
        commands = self.plan("--base-image", base)
        self.assertIn("BASE_IMAGE=" + base, commands[0])
        self.assertIn("ATTACHED_BASE_IMAGE=" + base, commands[1])
        self.assertIn("ASSETS_IMAGE=" + base, commands[1])

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
