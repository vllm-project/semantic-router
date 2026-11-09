from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT / "tools/ci"))
from image_artifacts import DEFINITIONS, ROUTER_BUILDS, ROUTER_DOCKERFILE  # noqa: E402


def router_build_command(dockerfile: str = ROUTER_DOCKERFILE) -> str:
    """The router stage's final RUN command, with line continuations joined."""
    text = (REPO_ROOT / dockerfile).read_text().replace("\\\n", " ")
    stage = text.split(" AS router-build", 1)[1].split("\nFROM ", 1)[0]
    runs = re.findall(r"(?m)^RUN (.+)$", stage)
    return next(run for run in reversed(runs) if "go build" in run)


class DockerCrossCompilationTests(unittest.TestCase):
    def test_isolated_router_builds_do_not_use_local_vendor_dependencies(self):
        for dockerfile in (
            "tools/docker/Dockerfile.attached",
            "tools/docker/riscv64/Dockerfile",
        ):
            with self.subTest(
                dockerfile=dockerfile
            ), tempfile.TemporaryDirectory() as tmp:
                stubs = Path(tmp)
                (stubs / "dpkg").write_text("#!/bin/sh\necho amd64\n")
                (stubs / "go").write_text(
                    '#!/bin/sh\ncase " $* " in *" -mod=mod "*) exit 0;; *) exit 23;; esac\n'
                )
                for stub in stubs.iterdir():
                    stub.chmod(0o755)
                result = subprocess.run(
                    ["bash", "-e", "-c", router_build_command(dockerfile)],
                    env=os.environ
                    | {"PATH": f"{stubs}:{os.environ['PATH']}", "TARGETARCH": "amd64"},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    "build must explicitly use module dependencies",
                )

    def test_attached_rejects_amd64_target_on_arm64_builder(self):
        text = (
            (REPO_ROOT / "tools/docker/Dockerfile.attached")
            .read_text()
            .replace("\\\n", " ")
        )
        stage = text.split(" AS router-build", 1)[1].split("\nFROM ", 1)[0]
        command = re.findall(r"(?m)^RUN (.+)$", stage)[0]
        with tempfile.TemporaryDirectory() as tmp:
            stubs = Path(tmp)
            (stubs / "dpkg").write_text("#!/bin/sh\necho arm64\n")
            (stubs / "dpkg").chmod(0o755)
            result = subprocess.run(
                ["bash", "-e", "-c", command],
                env=os.environ
                | {"PATH": f"{stubs}:{os.environ['PATH']}", "TARGETARCH": "amd64"},
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("native amd64 builder", result.stderr)

    def test_separate_native_builder_selects_the_riscv_linker_and_ffi(self):
        command = router_build_command("tools/docker/riscv64/Dockerfile")
        for host, linker in (("amd64", "riscv64-linux-gnu-gcc"), ("riscv64", "")):
            with tempfile.TemporaryDirectory() as tmp:
                stubs = Path(tmp)
                (stubs / "dpkg").write_text(f"#!/bin/sh\necho {host}\n")
                (stubs / "go").write_text(
                    '#!/bin/sh\nprintf "%s\\n" "CC=${CC:-} GOARCH=$GOARCH CGO_ENABLED=$CGO_ENABLED" "$*"\n'
                )
                for stub in stubs.iterdir():
                    stub.chmod(0o755)
                environment = {
                    key: value for key, value in os.environ.items() if key != "CC"
                } | {"PATH": f"{stubs}:{os.environ['PATH']}"}
                result = subprocess.run(
                    ["bash", "-e", "-c", command],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn(
                    f"CC={linker} GOARCH=riscv64 CGO_ENABLED=1", result.stdout
                )
                self.assertIn("-extldflags=-L/out/glide", result.stdout)

    def test_router_cross_compiles_arm64_with_only_the_target_c_linker(self) -> None:
        command = router_build_command()
        with tempfile.TemporaryDirectory() as tmp:
            stubs = Path(tmp)
            (stubs / "dpkg").write_text("#!/bin/sh\necho amd64\n")
            (stubs / "go").write_text(
                '#!/bin/sh\necho "CC=${CC:-} GOARCH=$GOARCH CGO_ENABLED=$CGO_ENABLED"\n'
            )
            for stub in stubs.iterdir():
                stub.chmod(0o755)
            for arch, linker in (("arm64", "aarch64-linux-gnu-gcc"), ("amd64", "")):
                with self.subTest(arch=arch):
                    environment = {
                        key: value for key, value in os.environ.items() if key != "CC"
                    } | {"PATH": f"{stubs}:{os.environ['PATH']}", "TARGETARCH": arch}
                    result = subprocess.run(
                        ["bash", "-e", "-c", command],
                        env=environment,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.stdout.strip(),
                        f"CC={linker} GOARCH={arch} CGO_ENABLED=1",
                    )

    def test_router_images_share_one_dockerfile_and_publish_their_platforms(
        self,
    ) -> None:
        for image, (target, accelerator) in ROUTER_BUILDS.items():
            with self.subTest(image=image):
                _, dockerfile, platforms = DEFINITIONS[image]
                attached = image == "vllm-sr-riscv64"
                self.assertEqual(
                    dockerfile,
                    (
                        "tools/docker/Dockerfile.attached"
                        if attached
                        else ROUTER_DOCKERFILE
                    ),
                )
                self.assertEqual(target, "vllm-sr-attached" if attached else "vllm-sr")
                expected = (
                    ["linux/riscv64"]
                    if attached
                    else (
                        ["linux/amd64", "linux/arm64"]
                        if accelerator == "cpu"
                        else ["linux/amd64"]
                    )
                )
                self.assertEqual(platforms, expected)
        builder = (REPO_ROOT / ".github/workflows/build-artifacts.yml").read_text()
        publisher = (REPO_ROOT / ".github/workflows/docker-publish.yml").read_text()
        self.assertIn("image_artifacts.py", builder)
        self.assertIn("--multiarch", builder)
        self.assertIn("steps.definition.outputs.target", builder)
        self.assertIn("steps.definition.outputs.accelerator", builder)
        self.assertIn("image_artifacts.py promote", publisher)
        self.assertNotIn("docker/build-push-action", publisher)


if __name__ == "__main__":
    unittest.main()
