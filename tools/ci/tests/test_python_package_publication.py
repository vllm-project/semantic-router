from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from check_cli_wheel import check_wheel  # noqa: E402
from prepare_dev_package import prepare_version  # noqa: E402
from validate_workflows import load_workflows, needs  # noqa: E402


class DevelopmentVersionTests(unittest.TestCase):
    def test_source_timestamp_orders_versions_and_preserves_other_metadata(
        self,
    ) -> None:
        epoch = int(datetime(2026, 9, 14, tzinfo=timezone.utc).timestamp())
        versions = []
        for timestamp in (epoch, epoch, epoch + 1):
            with tempfile.TemporaryDirectory() as temporary:
                project = Path(temporary)
                source = '[project]\nname = "vllm-sr"\nversion = "0.3.0"\n'
                (project / "pyproject.toml").write_text(source, encoding="utf-8")
                version = prepare_version(project, timestamp)
                versions.append(version)
                self.assertEqual(
                    (project / "pyproject.toml").read_text(encoding="utf-8"),
                    source.replace('"0.3.0"', f'"{version}"'),
                )
        self.assertEqual(versions[0], "0.3.0.dev20260914000000")
        self.assertEqual(versions[0], versions[1])
        self.assertLess(versions[1], versions[2])

    def test_rejects_non_release_base_without_editing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            path = project / "pyproject.toml"
            source = 'version = "0.3.0.dev20260101"\n'
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_version(project, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), source)


class PythonPublisherContractTests(unittest.TestCase):
    def setUp(self) -> None:
        errors: list[str] = []
        self.workflows = load_workflows(errors)
        self.assertEqual(errors, [])
        self.publisher = self.workflows["pypi-publish.yml"]

    def test_main_dev_publication_survives_skipped_domains_after_gate(self) -> None:
        main = self.workflows["main.yml"]
        publisher = main.jobs["pypi"]
        self.assertEqual(publisher["uses"], "./.github/workflows/pypi-publish.yml")
        self.assertEqual(publisher["with"]["channel"], "dev")
        for job_name in ("pypi", "images", "helm"):
            job = main.jobs[job_name]
            self.assertIn("ci", needs(job))
            self.assertIn("!cancelled()", job["if"])
            self.assertIn("needs.ci.result == 'success'", job["if"])

    def test_only_trusted_main_and_release_events_can_publish(self) -> None:
        condition = " ".join(self.publisher.jobs["pypi"]["if"].split())
        self.assertIn("github.repository == 'vllm-project/semantic-router'", condition)
        self.assertIn("github.event_name == 'push'", condition)
        self.assertIn(
            "inputs.channel == 'dev' && github.ref == 'refs/heads/main'", condition
        )
        self.assertIn(
            "inputs.channel == 'stable' && startsWith(github.ref, 'refs/tags/v')",
            condition,
        )
        self.assertEqual(
            set(self.publisher.events), {"workflow_call", "workflow_dispatch"}
        )
        self.assertEqual(
            self.publisher.jobs["testpypi"]["if"],
            "github.event_name == 'workflow_dispatch'",
        )
        callers = {
            workflow.path.name
            for workflow in self.workflows.values()
            for job in workflow.jobs.values()
            if job.get("uses") == "./.github/workflows/pypi-publish.yml"
        }
        self.assertEqual(callers, {"main.yml", "release.yml"})
        for step in self.publisher.jobs["build"]["steps"]:
            self.assertNotIn("secrets.", str(step))

    def test_publisher_and_pr_cli_gate_exercise_installed_wheel(self) -> None:
        steps = self.publisher.jobs["build"]["steps"]
        smoke_index = next(
            index
            for index, step in enumerate(steps)
            if "check_cli_wheel.py" in step.get("run", "")
        )
        upload_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses", "").startswith("actions/upload-artifact@")
        )
        self.assertLess(smoke_index, upload_index)
        self.assertEqual(needs(self.publisher.jobs["pypi"]), {"build"})
        package = self.workflows["package-check.yml"]
        self.assertIn("package_contract.py", str(package.jobs))
        implementation = (REPO_ROOT / "tools/ci/package_contract.py").read_text()
        self.assertIn("check_wheel(wheels[0])", implementation)
        for filename in ("main.yml", "release.yml"):
            self.assertTrue(
                self.workflows[filename].jobs["pypi"]["with"]["prebuilt-dist"]
            )

    def test_prebuilt_publication_installs_qualification_dependency_before_verify(
        self,
    ) -> None:
        steps = self.publisher.jobs["pypi"]["steps"]
        install_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Install qualification dependencies"
        )
        verify_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Verify qualified source and package content"
        )
        publish_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("name") == "Publish verified package"
        )
        self.assertLess(install_index, verify_index)
        self.assertLess(verify_index, publish_index)
        install = steps[install_index]
        self.assertEqual(install["if"], "inputs.prebuilt-dist")
        self.assertIn("PyYAML==6.0.3", install["run"])
        self.assertIn("PyYAML==6.0.3", str(self.workflows["package-check.yml"].jobs))

    def test_installer_uses_isolated_home_without_runtime_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "candidate.whl"
            wheel.touch()
            with (
                patch.dict(
                    os.environ,
                    {
                        "VLLM_SR_INSTALL_ROOT": "/outside",
                        "VLLM_SR_PIP_SPEC": "/outside.whl",
                        "BASH_ENV": "/outside",
                        "PYTHONPATH": "/outside",
                    },
                ),
                patch("check_cli_wheel.sys.platform", "linux"),
                patch("check_cli_wheel.venv.EnvBuilder.create"),
                patch(
                    "check_cli_wheel.subprocess.run",
                    side_effect=RuntimeError("stop before installation"),
                ) as run,
                self.assertRaisesRegex(RuntimeError, "stop before installation"),
            ):
                check_wheel(wheel)

            command = run.call_args.kwargs["args"]
            environment = run.call_args.kwargs["env"]
            self.assertEqual(command[:2], ["bash", str(REPO_ROOT / "install.sh")])
            self.assertEqual(command[command.index("--mode") + 1], "cli")
            self.assertEqual(command[command.index("--runtime") + 1], "skip")
            self.assertIn("--no-launch", command)
            self.assertEqual(
                command[command.index("--pip-spec") + 1], str(wheel.resolve())
            )
            for name in (
                "VLLM_SR_INSTALL_ROOT",
                "VLLM_SR_PIP_SPEC",
                "BASH_ENV",
                "PYTHONPATH",
            ):
                self.assertNotIn(name, environment)
            for option in ("--install-root", "--bin-dir"):
                self.assertTrue(
                    Path(command[command.index(option) + 1]).is_relative_to(
                        run.call_args.kwargs["cwd"]
                    )
                )
            self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)
            self.assertNotEqual(Path(environment["HOME"]), Path.home())
            self.assertTrue(
                Path(environment["HOME"]).is_relative_to(run.call_args.kwargs["cwd"])
            )

    def test_windows_keeps_the_direct_wheel_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "candidate.whl"
            wheel.touch()
            with (
                patch("check_cli_wheel.sys.platform", "win32"),
                patch("check_cli_wheel.venv.EnvBuilder.create") as create,
                patch(
                    "check_cli_wheel.subprocess.run",
                    side_effect=RuntimeError("stop before installation"),
                ) as run,
                self.assertRaisesRegex(RuntimeError, "stop before installation"),
            ):
                check_wheel(wheel)

            create.assert_called_once()
            command = run.call_args.kwargs["args"]
            self.assertEqual(
                command[1:],
                [
                    "-I",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    str(wheel.resolve()),
                ],
            )

    def _run_version_contract(
        self, *, channel: str, version: str, tag: str, snapshot: str
    ) -> subprocess.CompletedProcess[str]:
        contract = next(
            step
            for step in self.publisher.jobs["build"]["steps"]
            if step.get("id") == "contract"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "src" / "vllm-sr"
            project.mkdir(parents=True)
            (project / "pyproject.toml").write_text(
                f'version = "{version}"\n', encoding="utf-8"
            )
            (root / "config" / "recipes" / "built-in" / snapshot).mkdir(parents=True)
            result = subprocess.run(
                [
                    "bash",
                    "-e",
                    "-c",
                    contract["run"].replace("${{ github.event_name }}", "push"),
                ],
                cwd=project,
                env={
                    **os.environ,
                    "PACKAGE_CHANNEL": channel,
                    "RELEASE_VERSION": version,
                    "RELEASE_TAG": tag,
                    "INPUT_CATALOG_SNAPSHOT": "",
                    "GITHUB_OUTPUT": str(root / "output"),
                },
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self.assertEqual(
                    (root / "output").read_text().strip(),
                    f"catalog_snapshot={snapshot}",
                )
            return result

    def test_dev_build_uses_latest_without_requiring_future_release_snapshot(
        self,
    ) -> None:
        result = self._run_version_contract(
            channel="dev", version="9.8.0.dev20260914000000", tag="", snapshot="latest"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stable_release_keeps_minor_snapshot_and_tag_contract(self) -> None:
        for tag, expected in (("v0.3.0", 0), ("v0.2.0", 1)):
            result = self._run_version_contract(
                channel="stable", version="0.3.0", tag=tag, snapshot="v0.3"
            )
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_unknown_channel_fails_before_build(self) -> None:
        result = self._run_version_contract(
            channel="preview", version="0.3.0", tag="v0.3.0", snapshot="v0.3"
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
