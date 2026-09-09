from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools/ci"))

from classify_pr_changes import classify  # noqa: E402
from docker_image_catalog import image_definition  # noqa: E402
from domain_registry import domain_records, image_records  # noqa: E402
from gate_results import validate_results  # noqa: E402
from promote_image import CHANNEL, REVISION, should_promote  # noqa: E402
from qualify_release import qualified_run  # noqa: E402


class SelectionTests(unittest.TestCase):
    def test_python_changes_do_not_require_native_builds(self):
        for path in (
            "src/vllm-sr/cli/evaluation/scoring.py",
            "src/fleet-sim/tests/test_api.py",
            "src/training/tests/test_models.py",
        ):
            with self.subTest(path=path):
                result = classify([path])
                self.assertIn("python-unit", result.selected_jobs)
                self.assertNotIn("core-tests", result.selected_jobs)

    def test_workflow_changes_execute_the_full_python_baseline(self):
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/python-unit.yml").read_text()
        )
        scope = next(
            step
            for step in workflow["jobs"]["unit"]["steps"]
            if step.get("id") == "scope"
        )
        baseline = {"vllm-sr-cli", "fleet-sim", "training"}
        for paths, expected in (
            ([".github/workflows/python-unit.yml"], baseline),
            (
                [".github/workflows/python-unit.yml", "src/vllm-sr/cli/core.py"],
                baseline,
            ),
            (["src/fleet-sim/tests/test_api.py"], {"fleet-sim"}),
        ):
            result = classify(paths)
            self.assertIn("python-unit", result.selected_jobs)
            with self.subTest(paths=paths), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "output"
                subprocess.run(
                    ["bash", "-e", "-c", scope["run"]],
                    env={
                        **os.environ,
                        "DOMAINS": json.dumps(result.domains),
                        "GITHUB_OUTPUT": str(output),
                    },
                    check=True,
                )
                selected = json.loads(output.read_text().removeprefix("domains="))
                self.assertEqual(set(selected), expected)

    def test_resource_and_kind_tooling_select_harness_contracts(self):
        for path in (
            "tools/dev/with_test_resources.py",
            "tools/dev/tests/test_test_resources.py",
            "tools/kind/generate-kind-config.sh",
            "tools/kind/kind-config.yaml.template",
        ):
            with self.subTest(path=path):
                result = classify([path])
                self.assertIn("harness", result.domains)
                self.assertIn(
                    "make harness-check", domain_records()["harness"]["checks"]
                )
                self.assertIn("quality", result.selected_jobs)

    def test_live_logs_resolve_only_the_current_cluster_private_config(self):
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/integration-test-k8s.yml").read_text()
        )
        job = workflow["jobs"]["integration-test"]
        prepare = next(
            step
            for step in job["steps"]
            if step.get("name") == "Prepare private CI kubeconfig path"
        )
        self.assertIn("KUBECONFIG=$RUNNER_TEMP/", prepare["run"])
        step = next(step for step in job["steps"] if step.get("id") == "e2e-test")
        function = step["run"].split("start_router_log_stream &", 1)[0]
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            kind = directory / "kind"
            kind.write_text(
                "#!/bin/sh\n"
                'test "$*" = "get kubeconfig --name current-run" || exit 9\n'
                "echo private-current-run-config\n"
            )
            kind.chmod(0o755)
            bash = directory / "bash"
            bash.write_text('#!/bin/sh\ncat "$KUBECONFIG"\n')
            bash.chmod(0o755)
            config = directory / "private-config"
            result = subprocess.run(
                ["/bin/bash", "-eu", "-c", function + "\nstart_router_log_stream"],
                env={
                    **os.environ,
                    "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                    "E2E_CLUSTER_NAME": "current-run",
                    "KUBECONFIG": str(config),
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), "private-current-run-config")
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)

    def test_shared_cli_escalations_are_reachable(self):
        for path in (
            "tools/make/docker.mk",
            "e2e/testing/vllm-sr-cli/test_integration_storage_isolation.py",
        ):
            with self.subTest(path=path):
                self.assertIn("cli", classify([path]).selected_jobs)

    def test_adding_documentation_cannot_remove_a_selected_check(self):
        for path in (
            "src/vllm-sr/cli/core.py",
            "dashboard/frontend/src/App.tsx",
            "src/semantic-router/pkg/config/config.go",
        ):
            before = classify([path])
            after = classify([path, "website/docs/installation/index.md"])
            self.assertTrue(set(before.selected_jobs) <= set(after.selected_jobs))
            self.assertTrue(after.signals["website"])

    def test_e2e_compile_remains_lightweight_and_is_selected_for_unmapped_cases(self):
        result = classify(["e2e/testcases/new_contract.go"])
        self.assertIn("e2e-framework", result.domains)
        self.assertNotIn("core-tests", result.selected_jobs)
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/pre-commit.yml").read_text()
        )
        job = workflow["jobs"]["e2e-compile"]
        self.assertIn("e2e-framework", job["if"])
        self.assertEqual(
            [step["run"] for step in job["steps"] if "run" in step], ["make build-e2e"]
        )


class GateTests(unittest.TestCase):
    def test_aggregate_jobs_keep_failure_evidence_but_stop_after_cancellation(self):
        # A status function prevents GitHub from adding the implicit success()
        # guard, so failed dependencies still reach the gate. Unlike always(),
        # !cancelled() releases a superseded workflow's concurrency slot.
        aggregates = (
            ("pr.yml", "pr-gate"),
            ("main.yml", "gate"),
            ("nightly-build.yml", "qualification"),
            ("recipe-conformance.yml", "report"),
        )
        for filename, job_name in aggregates:
            with self.subTest(workflow=filename, job=job_name):
                workflow = yaml.safe_load(
                    (REPO_ROOT / ".github/workflows" / filename).read_text()
                )
                condition = workflow["jobs"][job_name]["if"]
                self.assertIn("!cancelled()", condition)
                self.assertNotIn("always()", condition)
                self.assertNotIn("success()", condition)
                if job_name == "pr-gate":
                    self.assertIn("github.event.label.name == 'ci/full'", condition)
                if job_name == "report":
                    self.assertIn("needs.inventory.result == 'success'", condition)
                    steps = workflow["jobs"]["live-cpu"]["steps"]
                    cleanup = next(
                        step for step in steps if step.get("name") == "Clean up"
                    )
                    self.assertEqual(cleanup["if"], "always()")

    def test_selected_jobs_must_succeed(self):
        for state in ("failure", "cancelled", "skipped", None):
            result = {
                "changes": {"result": "success"},
                "quality": {"result": "success"},
                "python-unit": {"result": state},
            }
            self.assertTrue(validate_results(result, {"python_unit": "true"}))
            result["python-unit"]["result"] = "success"
            self.assertEqual(validate_results(result, {"python_unit": "true"}), [])

    def test_unselected_jobs_may_skip_but_selected_missing_jobs_fail(self):
        result = {
            "changes": {"result": "success"},
            "quality": {"result": "success"},
            "python-unit": {"result": "skipped"},
        }
        self.assertEqual(validate_results(result, {"python_unit": "false"}), [])
        del result["python-unit"]
        self.assertTrue(validate_results(result, {"python_unit": "true"}))

    def test_main_images_are_downstream_of_validation(self):
        result = {
            "changes": {"result": "success"},
            "quality": {"result": "success"},
            "core": {"result": "success"},
        }
        self.assertEqual(
            validate_results(
                result, {"core_test": "true", "images": "true"}, lifecycle="main"
            ),
            [],
        )
        self.assertTrue(
            validate_results(result, {"core_test": "true", "images": "true"})
        )

    def test_failed_classifier_is_never_treated_as_no_work(self):
        self.assertTrue(validate_results({"changes": {"result": "failure"}}, {}))


class PublicationTests(unittest.TestCase):
    def test_release_qualification_uses_exact_sha_and_latest_attempt(self):
        run = {
            "id": 1,
            "head_sha": "target",
            "event": "push",
            "head_branch": "main",
            "status": "completed",
            "conclusion": "success",
        }
        self.assertEqual(qualified_run([run], "target"), run)
        for extra in (
            {"head_sha": "other"},
            {"head_branch": "topic"},
            {"conclusion": "failure"},
            {"status": "in_progress"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                qualified_run([{**run, **extra}], "target")
        with self.assertRaises(ValueError):
            qualified_run([run, {**run, "id": 2, "conclusion": "failure"}], "target")

    def test_promotion_never_rolls_back_newer_source(self):
        def ancestor(old, new):
            return (old, new) == ("old", "new")

        self.assertTrue(should_promote({REVISION: "old"}, "new", "main", ancestor))
        self.assertFalse(should_promote({REVISION: "new"}, "old", "release", ancestor))
        self.assertFalse(
            should_promote(
                {REVISION: "same", CHANNEL: "release"}, "same", "main", ancestor
            )
        )
        with self.assertRaises(ValueError):
            should_promote({REVISION: "unrelated"}, "new", "main", ancestor)

    def test_every_image_uses_one_definition_with_preserved_pr_platform_scope(self):
        for name in image_records():
            published = image_definition(name)
            validated = image_definition(name, pr=True)
            self.assertEqual(published.context, validated.context)
            self.assertEqual(published.dockerfile, validated.dockerfile)
            self.assertTrue(
                set(validated.platforms.split(","))
                <= set(published.platforms.split(","))
            )
        self.assertEqual(image_definition("extproc", pr=True).platforms, "linux/amd64")
        self.assertEqual(
            image_definition("dashboard", pr=True).platforms, "linux/amd64,linux/arm64"
        )
        with self.assertRaises(ValueError):
            image_definition("unknown")


if __name__ == "__main__":
    unittest.main()
