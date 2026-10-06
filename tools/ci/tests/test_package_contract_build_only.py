from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

import package_contract  # noqa: E402


class PackageBuildOnlyTests(unittest.TestCase):
    SOURCE_SHA = "a" * 40
    BASE_SHA = "b" * 40

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        project = self.root / "src/vllm-sr"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text(
            '[project]\nname = "vllm-sr"\nversion = "9.8.7"\n'
            '[project.optional-dependencies]\nruntime = ["vllm-srun==9.8.7"]\n',
            encoding="utf-8",
        )
        runtime = self.root / "src/model-runtime"
        runtime.mkdir(parents=True)
        (runtime / "pyproject.toml").write_text(
            '[project]\nname = "vllm-srun"\nversion = "9.8.7"\n', encoding="utf-8"
        )
        self.output = self.root / "output"
        self.commands: list[tuple[str, ...]] = []
        self.git_commands: list[tuple[str, ...]] = []

    def fake_run(self, *command: str) -> None:
        self.commands.append(command)
        names = {"src/vllm-sr": "vllm_sr", "src/model-runtime": "vllm_srun"}
        if command[1:3] == ("-m", "build") and command[3] in names:
            dist = Path(command[-1])
            dist.mkdir(parents=True, exist_ok=True)
            name = names[command[3]]
            (dist / f"{name}-9.8.7-py3-none-any.whl").write_bytes(
                b"wheel " + name.encode()
            )
            (dist / f"{name}-9.8.7.tar.gz").write_bytes(b"sdist " + name.encode())

    def fake_check_output(self, command: list[str], **_kwargs: object) -> str:
        self.git_commands.append(tuple(command))
        if command == ["git", "rev-parse", "HEAD"]:
            return self.SOURCE_SHA + "\n"
        if command[:3] == ["git", "rev-parse", "--verify"]:
            return self.BASE_SHA + "\n"
        raise AssertionError(f"Unexpected Git command: {command}")

    def invoke(self, *flags: str) -> tuple[mock.MagicMock, ...]:
        argv = [
            "package_contract.py",
            "--mode",
            "release",
            "--tag",
            "v9.8.7",
            "--base-ref",
            "missing-base-ref",
            "--output",
            str(self.output),
            *flags,
        ]
        with (
            mock.patch.object(package_contract, "ROOT", self.root),
            mock.patch.object(package_contract, "run", side_effect=self.fake_run),
            mock.patch.object(
                package_contract.subprocess,
                "check_output",
                side_effect=self.fake_check_output,
            ),
            mock.patch.object(package_contract, "verify_resources") as resources,
            mock.patch.object(
                package_contract, "verify_runtime_resources"
            ) as runtime_resources,
            mock.patch.object(package_contract, "check_wheel") as installed,
            mock.patch.object(package_contract, "check_runtime_wheel") as runtime,
            mock.patch.object(
                package_contract, "host_platform", return_value="linux/amd64"
            ),
            mock.patch.object(sys, "argv", argv),
        ):
            package_contract.main()
        return resources, installed, runtime_resources, runtime

    def test_build_only_keeps_distribution_integrity_without_test_qualification(
        self,
    ) -> None:
        resources, installed, runtime_resources, runtime = self.invoke("--build-only")

        self.assertEqual(self.git_commands, [("git", "rev-parse", "HEAD")])
        self.assertEqual(
            json.loads((self.output / "evidence.json").read_text())["expected_checks"],
            package_contract.BUILD_ONLY_CHECKS,
        )
        self.assertEqual(
            [
                check["id"]
                for check in json.loads((self.output / "evidence.json").read_text())[
                    "checks"
                ]
            ],
            package_contract.BUILD_ONLY_CHECKS,
        )
        self.assertTrue(
            all(
                check["status"] == "passed"
                for check in json.loads((self.output / "evidence.json").read_text())[
                    "checks"
                ]
            )
        )
        self.assertTrue(
            any(
                any(part.endswith("stage_model_catalog_package.py") for part in command)
                for command in self.commands
            )
        )
        self.assertTrue(any("twine" in command for command in self.commands))
        self.assertTrue(
            any(
                "--check" in command and "--version" in command
                for command in self.commands
            )
        )
        resources.assert_called_once()
        installed.assert_not_called()
        runtime_resources.assert_called_once()
        runtime.assert_not_called()
        built = [
            command[3] for command in self.commands if command[1:3] == ("-m", "build")
        ]
        self.assertEqual(built, ["src/vllm-sr", "src/model-runtime"])

        dist = self.output / "dist"
        manifest = json.loads((dist / "manifest.json").read_text())
        self.assertEqual(
            {key: manifest[key] for key in ("source_sha", "mode", "tag", "version")},
            {
                "source_sha": self.SOURCE_SHA,
                "mode": "release",
                "tag": "v9.8.7",
                "version": "9.8.7",
            },
        )
        self.assertEqual(
            manifest["files"],
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in dist.iterdir()
                if path.name != "manifest.json"
            },
        )

        with (
            mock.patch.object(package_contract, "ROOT", self.root),
            mock.patch.object(package_contract, "run", side_effect=AssertionError),
            mock.patch.object(
                package_contract.subprocess,
                "check_output",
                side_effect=self.fake_check_output,
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    "package_contract.py",
                    "--verify-dist",
                    "--mode",
                    "release",
                    "--tag",
                    "v9.8.7",
                    "--output",
                    str(dist),
                ],
            ),
        ):
            package_contract.main()

    def test_normal_qualification_still_runs_all_checks_and_resolves_base(self) -> None:
        resources, installed, runtime_resources, runtime = self.invoke()
        self.assertEqual(
            self.git_commands,
            [
                ("git", "rev-parse", "HEAD"),
                (
                    "git",
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    "missing-base-ref^{commit}",
                ),
            ],
        )
        report = json.loads((self.output / "evidence.json").read_text())
        self.assertEqual(report["expected_checks"], package_contract.CHECKS)
        self.assertEqual(
            [check["id"] for check in report["checks"]], package_contract.CHECKS
        )
        self.assertTrue(
            any("tools/catalog/tests" in command for command in self.commands)
        )
        self.assertTrue(
            any("--check-published" in command for command in self.commands)
        )
        resources.assert_called_once()
        installed.assert_called_once()
        runtime_resources.assert_called_once()
        (runtime_wheel, cli_wheel), _ = runtime.call_args
        self.assertEqual(runtime_wheel.name, "vllm_srun-9.8.7-py3-none-any.whl")
        self.assertEqual(cli_wheel.name, "vllm_sr-9.8.7-py3-none-any.whl")

    def test_build_only_requires_release_mode(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "package_contract.py",
                    "--mode",
                    "main",
                    "--build-only",
                    "--output",
                    str(self.output),
                ],
            ),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            package_contract.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
