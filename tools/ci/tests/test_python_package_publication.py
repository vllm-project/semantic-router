from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from check_cli_wheel import check_wheel  # noqa: E402
from ci_plan import make_plan  # noqa: E402
from image_artifacts import DECISION_RUNTIME_BASES  # noqa: E402
from package_contract import (  # noqa: E402
    DECISION_LOCK_PATH,
    ROCM_CHECKS,
    decision_catalog_revisions,
    decision_lock,
    decision_publication,
    rocm_qualification,
    verify_distribution,
)
from prepare_dev_package import prepare_version  # noqa: E402
from validate_workflows import load_workflows, needs  # noqa: E402


def rocm_record(source: str, manifest: bytes) -> dict:
    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    return {
        "schema_version": 1,
        "image": "decision-runtime-rocm",
        "source_sha": source,
        "platform": "linux/amd64",
        "digest": digest,
        "ref": f"ghcr.io/example/semantic-router/decision-runtime-rocm@{digest}",
        "evidence_sha256": "e" * 64,
        "models": {
            model_id: {
                "revision": revision,
                "checks": dict.fromkeys(ROCM_CHECKS, "passed"),
            }
            for model_id, revision in decision_catalog_revisions().items()
        },
    }


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
        self.assertEqual(needs(publisher), {"ci"})
        self.assertFalse(publisher["with"].get("decision_runtime", False))
        self.assertNotIn("decision-runtime-images", main.jobs["ci"]["with"])
        source = "src/vllm-sr/cli/decision_runtime/image_lock.py"
        normal_plan = make_plan([source], source_sha="a" * 40, profile="main")
        decision_plan = make_plan(
            [source], source_sha="a" * 40, profile="main", decision_runtime_images=True
        )
        self.assertTrue(normal_plan["publish_python"])
        self.assertNotIn("decision-runtime-cpu", normal_plan["publish_images"])
        self.assertIn("decision-runtime-cpu", decision_plan["publish_images"])
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
            if step.get("name") == "Verify prebuilt distribution"
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

    def test_python_publication_waits_for_the_published_decision_image(self) -> None:
        publisher = self.workflows["release.yml"].jobs["pypi"]
        self.assertIn("docker", needs(publisher))
        self.assertIn("needs.docker.result == 'success'", publisher["if"])
        self.assertTrue(publisher["with"]["decision_runtime"])
        receipt = self.workflows["docker-publish.yml"].jobs["publish"]["steps"]
        self.assertTrue(
            any(
                step.get("with", {}).get("name") == "ci-published-decision-runtime-cpu"
                for step in receipt
            )
        )
        publish_steps = self.publisher.jobs["pypi"]["steps"]
        self.assertTrue(
            any(
                "--decision-image-receipt" in step.get("run", "")
                and "--verify-registry" in step["run"]
                for step in publish_steps
            )
        )
        release = self.workflows["release.yml"]
        validation = release.jobs["validate"]
        self.assertIn("decision_rocm_qualification", validation["outputs"])
        rocm_step = next(
            step for step in validation["steps"] if step.get("id") == "decision_rocm"
        )
        self.assertEqual(rocm_step["if"], "github.event_name == 'push'")
        self.assertIn("--verify-rocm-qualification", rocm_step["run"])
        self.assertIn("--verify-registry", rocm_step["run"])
        self.assertIn("DECISION_ROCM_QUALIFICATION", rocm_step["env"])
        self.assertEqual(
            release.jobs["pypi"]["with"]["decision_rocm_qualification"],
            "${{ needs.validate.outputs.decision_rocm_qualification }}",
        )

    def test_stable_release_requires_version_bound_decision_images(
        self,
    ) -> None:
        release = self.workflows["release.yml"]
        publisher = release.jobs["pypi"]
        self.assertNotIn("decision_pypi", release.jobs)
        self.assertNotIn(
            "decision_runtime_release", release.jobs["validate"]["outputs"]
        )
        self.assertNotIn(
            "DECISION_RUNTIME_RELEASE", str(release.jobs["validate"]["steps"])
        )
        self.assertEqual(needs(publisher), {"validate", "gate", "docker"})
        self.assertIn("needs.docker.result == 'success'", publisher["if"])
        self.assertTrue(publisher["with"]["decision_runtime"])
        self.assertEqual(
            release.jobs["ci"]["with"]["decision-runtime-images"],
            True,
        )
        notes = release.jobs["release-notes"]
        self.assertIn("needs.pypi.result == 'success'", notes["if"])
        self.assertNotIn("decision_pypi", str(notes))
        python_downloads = [
            step
            for step in notes["steps"]
            if step.get("with", {}).get("path") == "release-assets/python"
        ]
        self.assertEqual(
            [step["with"]["name"] for step in python_downloads],
            ["vllm-sr-bound-dist"],
        )
        decision_plan = make_plan(
            [], source_sha="a" * 40, profile="release", decision_runtime_images=True
        )
        self.assertIn("decision-runtime-cpu", decision_plan["publish_images"])

        stable_guard = next(
            step
            for step in self.publisher.jobs["pypi"]["steps"]
            if step.get("name") == "Require Decision images for stable packages"
        )
        self.assertEqual(stable_guard["if"], "inputs.channel == 'stable'")
        self.assertIn('test "$DECISION_RUNTIME" = true', stable_guard["run"])
        self.assertIn('test "$PREBUILT_DIST" = true', stable_guard["run"])
        verify = next(
            step
            for step in self.publisher.jobs["pypi"]["steps"]
            if step.get("name") == "Verify prebuilt distribution"
        )
        self.assertIn('"$MODE" == release', verify["run"])
        self.assertIn("--require-decision-images", verify["run"])

    def test_decision_receipt_rejects_stale_source_and_registry_content(self) -> None:
        source = "a" * 40
        raw = b'{"schemaVersion":2}'
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        receipt = {
            "schema_version": 1,
            "image": "decision-runtime-cpu",
            "source_sha": source,
            "mode": "release",
            "tag": "v1.2.3",
            "digest": digest,
            "ref": f"ghcr.io/example/semantic-router/decision-runtime-cpu@{digest}",
            "archive_sha256": "b" * 64,
            "platform": "linux/amd64",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "published.json"
            path.write_text(json.dumps(receipt))
            with (
                patch.dict(os.environ, {"GITHUB_REPOSITORY_OWNER": "Example"}),
                patch("package_contract.subprocess.check_output", return_value=raw),
            ):
                decision_publication(
                    path,
                    source_sha=source,
                    mode="release",
                    tag="v1.2.3",
                    verify_registry=True,
                )
                with self.assertRaisesRegex(ValueError, "does not match"):
                    decision_publication(
                        path,
                        source_sha="c" * 40,
                        mode="release",
                        tag="v1.2.3",
                        verify_registry=True,
                    )
                with (
                    patch(
                        "package_contract.subprocess.check_output",
                        return_value=b"other",
                    ),
                    self.assertRaisesRegex(ValueError, "digest differs"),
                ):
                    decision_publication(
                        path,
                        source_sha=source,
                        mode="release",
                        tag="v1.2.3",
                        verify_registry=True,
                    )

    def test_rocm_qualification_requires_six_models_and_exact_registry_image(
        self,
    ) -> None:
        source = "a" * 40
        manifest = b'{"schemaVersion":2}'
        record = rocm_record(source, manifest)
        config = {
            "os": "linux",
            "architecture": "amd64",
            "config": {
                "Labels": {
                    "org.opencontainers.image.base.name": DECISION_RUNTIME_BASES[
                        "decision-runtime-rocm"
                    ][0],
                    "org.opencontainers.image.revision": source,
                    "ai.vllm-sr.decision.source-state": "clean",
                    "ai.vllm-sr.decision.backend": "rocm",
                }
            },
        }

        def inspect(command: list[str]) -> bytes:
            if "--raw" in command:
                return manifest
            if "--config" in command:
                return json.dumps(config).encode()
            raise AssertionError(f"unexpected command: {command}")

        with (
            patch.dict(os.environ, {"GITHUB_REPOSITORY_OWNER": "Example"}),
            patch("package_contract.subprocess.check_output", side_effect=inspect),
        ):
            with self.assertRaisesRegex(ValueError, "missing or oversized"):
                rocm_qualification("", source_sha=source, verify_registry=True)
            self.assertEqual(
                rocm_qualification(
                    json.dumps(record), source_sha=source, verify_registry=True
                ),
                record,
            )
            extra = {**record, "validation_host": "never-public"}
            with self.assertRaisesRegex(ValueError, "fields are invalid"):
                rocm_qualification(
                    json.dumps(extra), source_sha=source, verify_registry=True
                )
            with self.assertRaisesRegex(ValueError, "does not match"):
                rocm_qualification(
                    json.dumps(record), source_sha="b" * 40, verify_registry=True
                )
            missing = json.loads(json.dumps(record))
            missing["models"].pop(next(iter(missing["models"])))
            with self.assertRaisesRegex(ValueError, "all catalog models"):
                rocm_qualification(
                    json.dumps(missing), source_sha=source, verify_registry=True
                )
            failed = json.loads(json.dumps(record))
            next(iter(failed["models"].values()))["checks"]["no_regression"] = "failed"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                rocm_qualification(
                    json.dumps(failed), source_sha=source, verify_registry=True
                )
            config["config"]["Labels"]["ai.vllm-sr.decision.backend"] = "cpu"
            with self.assertRaisesRegex(ValueError, "config differs"):
                rocm_qualification(
                    json.dumps(record), source_sha=source, verify_registry=True
                )
            config["config"]["Labels"]["ai.vllm-sr.decision.backend"] = "rocm"
            with (
                patch(
                    "package_contract.subprocess.check_output",
                    return_value=b"different",
                ),
                self.assertRaisesRegex(ValueError, "digest differs"),
            ):
                rocm_qualification(
                    json.dumps(record), source_sha=source, verify_registry=True
                )

    def test_bound_wheel_and_sdist_must_contain_identical_version_lock(self) -> None:
        source = "a" * 40
        images = {
            backend: f"ghcr.io/example/semantic-router/decision-runtime-{backend}@sha256:{letter * 64}"
            for backend, letter in (("cpu", "b"), ("rocm", "c"))
        }
        lock = decision_lock("1.2.3", source, images)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            wheel = output / "vllm_sr-1.2.3-py3-none-any.whl"
            sdist = output / "vllm_sr-1.2.3.tar.gz"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(DECISION_LOCK_PATH, lock)
            with tarfile.open(sdist, "w:gz") as archive:
                member = tarfile.TarInfo("vllm_sr-1.2.3/" + DECISION_LOCK_PATH)
                member.size = len(lock)
                archive.addfile(member, io.BytesIO(lock))
            manifest = {
                "source_sha": source,
                "mode": "release",
                "tag": "v1.2.3",
                "version": "1.2.3",
                "files": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (wheel, sdist)
                },
                "decision_images": {
                    "refs": images,
                    "lock_sha256": hashlib.sha256(lock).hexdigest(),
                },
            }
            (output / "manifest.json").write_text(json.dumps(manifest))
            with patch("package_contract.subprocess.check_output", return_value=source):
                verify_distribution(
                    output, "release", "v1.2.3", require_decision_images=True
                )
                manifest["decision_images"]["refs"]["rocm"] = images["rocm"].replace(
                    "c" * 64, "d" * 64
                )
                (output / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "lock metadata differs"):
                    verify_distribution(output, "release", "v1.2.3")
                del manifest["decision_images"]["refs"]["rocm"]
                (output / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "backends are incomplete"):
                    verify_distribution(output, "release", "v1.2.3")
                for empty in ({}, None):
                    manifest["decision_images"] = empty
                    (output / "manifest.json").write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "metadata is invalid"):
                        verify_distribution(
                            output,
                            "release",
                            "v1.2.3",
                            require_decision_images=True,
                        )

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
