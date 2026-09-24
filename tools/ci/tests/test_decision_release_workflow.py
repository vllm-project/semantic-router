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

    def test_only_protected_main_or_stable_tag_can_run_live_rocm(self) -> None:
        self.assertEqual(set(self.decision.events), {"workflow_call"})
        self.assertTrue(self.decision.call_contract["inputs"]["mode"]["required"])
        rocm = self.decision.jobs["rocm"]
        for restriction in (
            "github.repository == 'vllm-project/semantic-router'",
            "github.event_name == 'push'",
            "inputs.mode == 'main' && github.ref == 'refs/heads/main'",
            "inputs.mode == 'release' && startsWith(github.ref, 'refs/tags/v')",
            "github.ref_name == inputs.tag",
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
        self.assertIn("${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}", candidate["run"])
        self.assertIn('--candidate-ref "$CANDIDATE_REF"', qualify["run"])
        self.assertIn("decision_rocm_promotion.py", qualify["run"])
        commands = "\n".join(step.get("run", "") for step in rocm["steps"])
        self.assertIn("decision_perf_release_gate.py validate", commands)
        self.assertIn("decision_perf_release_producer.py", commands)
        self.assertLess(
            commands.index("decision_perf_release_producer.py"),
            commands.index("decision_perf_release_gate.py validate"),
        )
        producer = next(
            step
            for step in rocm["steps"]
            if "decision_perf_release_producer.py" in step.get("run", "")
        )
        self.assertEqual(
            producer["env"]["BASELINE_CONFIG"],
            "${{ vars.DECISION_PAIRED_BASELINE_CONFIG_PATH }}",
        )
        for argument in (
            "--qualification-receipt",
            "--source-sha",
            "--run-id",
            "--run-attempt",
        ):
            self.assertIn(argument, producer["run"])
        performance_gate = next(
            step
            for step in rocm["steps"]
            if "decision_perf_release_gate.py validate" in step.get("run", "")
        )
        for argument in ("--qualification-receipt", "--candidate-ref", "--owner"):
            self.assertIn(argument, performance_gate["run"])
        upload = next(
            step
            for step in rocm["steps"]
            if step.get("with", {})
            .get("name", "")
            .startswith("decision-rocm-qualified-receipt-")
        )
        self.assertIn("${{ github.run_attempt }}", upload["with"]["name"])
        self.assertIn("/raw/**", upload["with"]["path"])
        cpu_qualify = next(
            step
            for step in rocm["steps"]
            if "decision_cpu_qualify.py" in step.get("run", "")
        )
        cpu_digest_download = next(
            step
            for step in rocm["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
            and step.get("with", {})
            .get("name", "")
            .startswith("ci-published-digest-decision-runtime-cpu-")
        )
        self.assertLess(
            rocm["steps"].index(cpu_digest_download), rocm["steps"].index(cpu_qualify)
        )
        self.assertLess(
            rocm["steps"].index(performance_gate), rocm["steps"].index(cpu_qualify)
        )
        self.assertEqual(cpu_qualify["env"]["HF_TOKEN"], "${{ secrets.HF_TOKEN }}")
        self.assertIn('--published-ref "$published_ref"', cpu_qualify["run"])
        self.assertIn("decision-runtime-cpu@${digest}", cpu_qualify["run"])
        self.assertIn('--port "$port"', cpu_qualify["run"])
        self.assertLess(
            commands.index("decision_perf_release_gate.py validate"),
            commands.index("decision_cpu_qualify.py"),
        )
        cpu_upload = next(
            step
            for step in rocm["steps"]
            if step.get("with", {})
            .get("name", "")
            .startswith("decision-cpu-qualified-receipt-")
        )
        self.assertIn("/qualification.json", cpu_upload["with"]["path"])
        self.assertIn("/raw/**", cpu_upload["with"]["path"])
        performance_upload = next(
            step
            for step in rocm["steps"]
            if step.get("with", {})
            .get("name", "")
            .startswith("decision-paired-performance-")
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
                "ci-published-digest-decision-runtime-cpu-${{ github.run_id }}-${{ github.run_attempt }}",
                "decision-cpu-qualified-receipt-${{ github.run_id }}-${{ github.run_attempt }}",
                "decision-rocm-qualified-receipt-${{ github.run_id }}-${{ github.run_attempt }}",
                "decision-paired-performance-${{ github.run_id }}-${{ github.run_attempt }}",
            ],
        )
        commands = "\n".join(step.get("run", "") for step in package["steps"])
        for expected in (
            "decision_rocm_promotion.py",
            "--promote",
            "decision_image_lock_release.py generate",
            "--cpu-receipt .agent-harness/decision-cpu-qualified/qualification.json",
            'package_contract.py --mode "$MODE" --tag "$TAG"',
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
        build = next(
            step
            for step in package["steps"]
            if "package_contract.py" in step.get("run", "")
        )
        self.assertEqual(build["env"]["MODE"], "${{ inputs.mode }}")
        self.assertEqual(build["env"]["TAG"], "${{ inputs.tag }}")
        release_evidence = next(
            step
            for step in package["steps"]
            if step.get("with", {})
            .get("name", "")
            .startswith("decision-qualified-release-evidence-")
        )
        self.assertIn(
            ".agent-harness/decision-cpu-qualified/raw/**",
            release_evidence["with"]["path"],
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
        self.assertEqual(decision["with"]["mode"], "main")
        self.assertEqual(needs(publisher), {"ci", "decision"})
        self.assertIn("needs.decision.result == 'success'", publisher["if"])
        self.assertIn(
            "contains(fromJSON(needs.ci.outputs.publish_images), 'decision-runtime-cpu')",
            publisher["if"],
        )
        self.assertIs(publisher["with"]["qualified-decision"], True)

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
            "${{ inputs.qualified-decision && format('decision-qualified-dist-{0}-{1}', github.run_id, github.run_attempt) || 'vllm-sr-dist' }}",
        )
        preflight = next(
            step
            for step in steps
            if "decision_release_policy.py" in step.get("run", "")
        )
        self.assertEqual(preflight["if"], "inputs.channel == 'stable'")
        self.assertIn('--expect "$QUALIFIED_DECISION"', preflight["run"])
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
            '--cpu-receipt "$evidence/.agent-harness/decision-cpu-qualified/qualification.json"',
            '--qualification-receipt "$receipt"',
            '--owner "$GITHUB_REPOSITORY_OWNER"',
            '--candidate-ref "$candidate_ref"',
            '--run-id "$GITHUB_RUN_ID"',
            '--run-attempt "$GITHUB_RUN_ATTEMPT"',
        ):
            self.assertIn(argument, commands[validate_index])
        self.assertEqual(steps[validate_index]["if"], "inputs.qualified-decision")
        self.assertIn("twine upload dist/*.whl dist/*.tar.gz\n", commands[upload_index])

    def test_stable_tag_always_qualifies_decision_capable_source(self) -> None:
        release = self.workflows["release.yml"]
        validate = release.jobs["validate"]
        gate = release.jobs["gate"]
        self.assertEqual(
            validate["outputs"]["decision_required"],
            "${{ steps.decision-policy.outputs.decision_required }}",
        )
        policy = next(
            step for step in validate["steps"] if step.get("id") == "decision-policy"
        )
        self.assertIn("decision_release_policy.py", policy["run"])
        self.assertNotIn("git diff --quiet", policy["run"])
        self.assertIn("DECISION_REQUIRED", gate["steps"][0]["run"])
        for required, enabled, expected in (
            ("", "", 1),
            ("false", "", 0),
            ("true", "", 1),
            ("true", "true", 0),
        ):
            with self.subTest(required=required, enabled=enabled):
                result = subprocess.run(
                    ["bash", "-e", "-c", gate["steps"][0]["run"]],
                    env={
                        **os.environ,
                        "VERSION_RESULT": "success",
                        "QUALIFICATION_RESULT": "success",
                        "DECISION_REQUIRED": required,
                        "DECISION_ENABLED": enabled,
                        "RELEASE_EVENT": "push",
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, expected)
        decision = release.jobs["decision"]
        cpu = release.jobs["decision-cpu"]
        self.assertEqual(cpu["with"]["images"], '["decision-runtime-cpu"]')
        self.assertEqual(cpu["with"]["mode"], "release")
        self.assertNotIn("publish_latest", cpu["with"])
        self.assertEqual(needs(decision), {"validate", "gate", "decision-cpu"})
        self.assertIn("decision_required == 'true'", decision["if"])
        self.assertIn("needs.decision-cpu.result == 'success'", decision["if"])
        self.assertEqual(decision["with"]["mode"], "release")
        self.assertEqual(decision["with"]["tag"], "${{ needs.validate.outputs.tag }}")
        for job_name in ("docker", "helm", "pypi", "crate"):
            job = release.jobs[job_name]
            self.assertIn("decision", needs(job))
            self.assertIn("needs.decision.result == 'success'", job["if"])
            self.assertIn("decision_required == 'false'", job["if"])
        self.assertFalse(release.jobs["docker"]["with"]["record_decision_digest"])
        self.assertEqual(
            release.jobs["pypi"]["with"]["qualified-decision"],
            "${{ needs.validate.outputs.decision_required == 'true' }}",
        )
        note_download = next(
            step
            for step in release.jobs["release-notes"]["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
        )
        self.assertIn("decision-qualified-dist-{0}-{1}", note_download["with"]["name"])


if __name__ == "__main__":
    unittest.main()
