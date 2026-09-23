"""Decision publication must consume live, source-bound image evidence."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from validate_workflows import load_workflows, needs  # noqa: E402


class DecisionReleaseWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        errors: list[str] = []
        self.workflows = load_workflows(errors)
        self.assertEqual(errors, [])
        self.decision = self.workflows["decision-qualified-package.yml"]

    def test_only_protected_main_can_run_live_rocm_qualification(self) -> None:
        self.assertEqual(set(self.decision.events), {"workflow_call"})
        rocm = self.decision.jobs["rocm"]
        for restriction in (
            "github.repository == 'vllm-project/semantic-router'",
            "github.ref == 'refs/heads/main'",
            "vars.DECISION_RUNTIME_RELEASE_ENABLED == 'true'",
        ):
            self.assertIn(restriction, rocm["if"])
        self.assertEqual(
            rocm["runs-on"], ["self-hosted", "linux", "x64", "decision-rocm"]
        )
        self.assertEqual(rocm["environment"], "decision-runtime-release")
        self.assertEqual(rocm["permissions"]["packages"], "write")
        candidate = next(
            step for step in rocm["steps"] if step.get("id") == "candidate"
        )
        qualify = next(
            step
            for step in rocm["steps"]
            if "decision_rocm_qualify.py" in step.get("run", "")
        )
        self.assertIn("build-image.sh rocm", candidate["run"])
        self.assertIn("docker push", candidate["run"])
        self.assertIn("skopeo inspect --raw", candidate["run"])
        self.assertIn('--candidate-ref "$CANDIDATE_REF"', qualify["run"])
        self.assertIn("decision_rocm_promotion.py", qualify["run"])
        self.assertIn(
            "decision_perf_release_gate.py validate",
            "\n".join(step.get("run", "") for step in rocm["steps"]),
        )
        performance_gate = next(
            step
            for step in rocm["steps"]
            if "decision_perf_release_gate.py validate" in step.get("run", "")
        )
        for argument in ("--qualification-receipt", "--candidate-ref", "--owner"):
            self.assertIn(argument, performance_gate["run"])
        self.assertNotIn(
            "decision_perf_release_producer.py",
            "\n".join(step.get("run", "") for step in rocm["steps"]),
        )
        upload = next(
            step
            for step in rocm["steps"]
            if step.get("with", {}).get("name") == "decision-rocm-qualified-receipt"
        )
        self.assertIn("/raw/**", upload["with"]["path"])
        performance_upload = next(
            step
            for step in rocm["steps"]
            if step.get("with", {}).get("name") == "decision-paired-performance"
        )
        self.assertTrue(performance_upload["with"]["path"].endswith("/**"))
        self.assertNotIn("workflow_dispatch", self.decision.events)

    def test_package_uses_both_published_digests_then_builds_new_dist(self) -> None:
        package = self.decision.jobs["package"]
        self.assertEqual(needs(package), {"rocm"})
        self.assertEqual(package["if"], "needs.rocm.result == 'success'")
        artifacts = [
            step["with"]["name"]
            for step in package["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
        ]
        self.assertEqual(
            artifacts,
            [
                "ci-published-digest-decision-runtime-cpu",
                "decision-rocm-qualified-receipt",
                "decision-paired-performance",
            ],
        )
        commands = "\n".join(step.get("run", "") for step in package["steps"])
        for expected in (
            "decision_rocm_promotion.py",
            "--promote",
            "decision_image_lock_release.py generate",
            "package_contract.py --mode main",
            "decision_image_lock_release.py check-dist",
            "decision_perf_release_gate.py validate",
        ):
            self.assertIn(expected, commands)
        for argument in ("--qualification-receipt", "--candidate-ref", "--owner"):
            self.assertIn(argument, commands)
        self.assertLess(commands.index("--promote"), commands.index("generate"))
        self.assertLess(
            commands.index("generate"), commands.index("package_contract.py")
        )
        self.assertLess(
            commands.index("package_contract.py"), commands.index("check-dist")
        )
        self.assertEqual(package["environment"], "decision-runtime-release")
        release_evidence = next(
            step
            for step in package["steps"]
            if step.get("with", {}).get("name") == "decision-qualified-release-evidence"
        )
        self.assertIn(
            ".agent-harness/decision-rocm/raw/**",
            release_evidence["with"]["path"],
        )
        self.assertIn(
            ".agent-harness/decision-performance/**",
            release_evidence["with"]["path"],
        )

    def test_missing_runner_or_registry_evidence_fails_the_called_workflow(
        self,
    ) -> None:
        gate = self.decision.jobs["gate"]
        self.assertEqual(gate["if"], "always()")
        self.assertEqual(needs(gate), {"rocm", "package"})
        command = gate["steps"][0]["run"]
        for rocm, package, expected in (
            ("success", "success", 0),
            ("skipped", "skipped", 1),
            ("failure", "skipped", 1),
            ("success", "failure", 1),
        ):
            with self.subTest(rocm=rocm, package=package):
                result = subprocess.run(
                    ["bash", "-e", "-c", command],
                    env={**os.environ, "ROCM_RESULT": rocm, "PACKAGE_RESULT": package},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, expected)

    def test_main_pypi_waits_for_qualified_decision_artifact(self) -> None:
        main = self.workflows["main.yml"]
        decision = main.jobs["decision"]
        publisher = main.jobs["pypi"]
        self.assertEqual(needs(decision), {"ci", "images"})
        self.assertIn("vars.DECISION_RUNTIME_RELEASE_ENABLED == 'true'", decision["if"])
        self.assertEqual(
            decision["uses"], "./.github/workflows/decision-qualified-package.yml"
        )
        self.assertEqual(needs(publisher), {"ci", "decision"})
        self.assertIn("needs.decision.result == 'success'", publisher["if"])
        self.assertIn("decision-runtime-cpu", publisher["if"])
        self.assertIn("qualified-decision", publisher["with"])

    def test_pypi_rechecks_lock_and_distribution_before_upload(self) -> None:
        publisher = self.workflows["pypi-publish.yml"]
        self.assertFalse(
            publisher.call_contract["inputs"]["qualified-decision"]["default"]
        )
        steps = publisher.jobs["pypi"]["steps"]
        commands = [step.get("run", "") for step in steps]
        condition = publisher.jobs["pypi"]["if"]
        self.assertIn("always()", condition)
        self.assertIn(
            "needs.build.result == 'success' || inputs.prebuilt-dist", condition
        )
        download = next(
            step
            for step in steps
            if step.get("uses", "").startswith("actions/download-artifact@")
        )
        self.assertEqual(
            download["with"]["name"],
            "${{ inputs.qualified-decision && 'decision-qualified-dist' || 'vllm-sr-dist' }}",
        )
        validate_index = next(
            index
            for index, command in enumerate(commands)
            if "decision_image_lock_release.py validate" in command
        )
        upload_index = next(
            index for index, command in enumerate(commands) if "twine upload" in command
        )
        self.assertLess(validate_index, upload_index)
        self.assertIn(
            "decision_image_lock_release.py check-dist", commands[validate_index]
        )
        self.assertIn(
            "decision_perf_release_gate.py validate", commands[validate_index]
        )
        self.assertIn(
            'receipt="$evidence/.agent-harness/decision-rocm/qualification.json"',
            commands[validate_index],
        )
        self.assertIn('candidate_ref="$(python3 -c ', commands[validate_index])
        self.assertIn('"$receipt")"', commands[validate_index])
        for argument in (
            '--qualification-receipt "$receipt"',
            '--owner "$GITHUB_REPOSITORY_OWNER"',
            '--candidate-ref "$candidate_ref"',
            '--run-id "$GITHUB_RUN_ID"',
            '--run-attempt "$GITHUB_RUN_ATTEMPT"',
        ):
            self.assertIn(argument, commands[validate_index])
        self.assertEqual(steps[validate_index]["if"], "inputs.qualified-decision")
        self.assertIn("twine upload dist/*.whl dist/*.tar.gz\n", commands[upload_index])

    def test_changed_decision_inputs_block_unqualified_stable_release(self) -> None:
        release = self.workflows["release.yml"]
        validate = release.jobs["validate"]
        gate = release.jobs["gate"]
        self.assertEqual(
            validate["outputs"]["decision_changed"],
            "${{ steps.decision-impact.outputs.changed }}",
        )
        impact = next(
            step for step in validate["steps"] if step.get("id") == "decision-impact"
        )
        self.assertIn("git diff --quiet", impact["run"])
        self.assertIn("src/vllm-sr/decision_runtime/", impact["run"])
        self.assertIn("src/vllm-sr/cli/commands/drun.py", impact["run"])
        self.assertIn("DECISION_CHANGED", gate["steps"][0]["run"])


if __name__ == "__main__":
    unittest.main()
