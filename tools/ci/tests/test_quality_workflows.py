"""Quality and component checks retain their execution owners without wrapper jobs."""

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def step_with_command(job, command):
    return next(step for step in job["steps"] if command in step.get("run", ""))


class QualityWorkflowTests(unittest.TestCase):
    def test_quality_checks_and_receipt_share_one_successful_job(self):
        data = workflow("pre-commit.yml")
        self.assertEqual(set(data["jobs"]), {"quality"})
        job = data["jobs"]["quality"]
        steps = job["steps"]
        references = step_with_command(job, "make docs-generated-check")
        source = step_with_command(job, "make check CI_STATIC_ONLY=true")
        website = step_with_command(job, "npm run build")
        receipt = step_with_command(job, "workflow_evidence.py")
        self.assertNotIn("if", references)
        self.assertNotIn("if", source)
        self.assertEqual(website["if"], "inputs.website")
        self.assertLess(steps.index(references), steps.index(website))
        self.assertLess(steps.index(source), steps.index(receipt))
        self.assertLess(steps.index(website), steps.index(receipt))
        self.assertNotIn("always()", receipt.get("if", ""))
        self.assertIn("--check website-build", receipt["run"])
        for step in (references, source, website, receipt):
            self.assertFalse(step.get("continue-on-error", False))

    def test_generated_native_contracts_do_not_block_source_quality(self):
        parent = workflow("ci.yml")["jobs"]
        self.assertEqual(parent["quality"]["needs"], "plan")
        self.assertEqual(parent["generated"]["needs"], ["plan", "native-build"])
        self.assertEqual(
            parent["generated"]["uses"], "./.github/workflows/check-generated.yml"
        )
        generated = workflow("check-generated.yml")["jobs"]
        self.assertEqual(set(generated), {"generated"})
        job = generated["generated"]
        loader = next(
            step
            for step in job["steps"]
            if step.get("uses") == "./.github/actions/load-native-artifact"
        )
        self.assertNotIn("if", loader)
        check = step_with_command(job, "make config-schema-check")
        self.assertIn("api-docs-check docs-crd-check", check["run"])
        self.assertLess(job["steps"].index(loader), job["steps"].index(check))
        self.assertEqual(parent["gate"]["if"], "always()")
        self.assertIn("generated", parent["gate"]["needs"])

    def test_security_receipt_requires_both_scans_against_the_planned_source(self):
        jobs = workflow("security-scan.yml")["jobs"]
        self.assertEqual(set(jobs), {"security"})
        job = jobs["security"]
        steps = job["steps"]
        checkouts = {
            step["with"]["path"]: step["with"]
            for step in steps
            if step.get("uses") == "actions/checkout@v4"
        }
        self.assertEqual(checkouts["pr-code"]["ref"], "${{ github.sha }}")
        self.assertIn("github.event.pull_request.base.sha", checkouts["base"]["ref"])
        self.assertFalse(checkouts["base"]["persist-credentials"])
        ast = step_with_command(job, "scan pr-code --fail-on HIGH")
        iac = step_with_command(job, "trivy config")
        receipt = step_with_command(job, "workflow_evidence.py")
        self.assertEqual(iac["working-directory"], "pr-code")
        self.assertEqual(receipt["working-directory"], "pr-code")
        self.assertIn("--severity HIGH,CRITICAL", iac["run"])
        for scan in (ast, iac):
            self.assertFalse(scan.get("continue-on-error", False))
            self.assertLess(steps.index(scan), steps.index(receipt))
        self.assertNotIn("always()", receipt.get("if", ""))
        advisory = [step for step in steps if step.get("continue-on-error")]
        self.assertEqual(len(advisory), 1)
        self.assertIn("scan_malicious_code.py", advisory[0]["run"])
        upload = next(
            step
            for step in steps
            if step.get("uses") == "github/codeql-action/upload-sarif@v4"
        )
        self.assertIn("head.repo.full_name == github.repository", upload["if"])
        self.assertIn("github.actor != 'dependabot[bot]'", upload["if"])


if __name__ == "__main__":
    unittest.main()
