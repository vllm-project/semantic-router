"""Real framework-shaped evidence must retain each selected deployment assertion."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment_test_results import go_unit, kubernetes, operator


class DeploymentResultsTests(unittest.TestCase):
    def test_kubernetes_requires_exact_passed_inventory(self):
        report = {
            "profile": "envoy-ai-gateway",
            "expected_cases": ["route", "cache"],
            "test_results": [
                {"Name": "route", "Passed": True},
                {"Name": "cache", "Passed": True},
            ],
            "status": "PASSED",
            "exit_code": 0,
            "total_tests": 2,
            "passed_tests": 2,
            "failed_tests": 0,
        }
        self.assertEqual(len(kubernetes(report, "envoy-ai-gateway")["cases"]), 2)
        changes = [
            {"expected_cases": []},
            {"expected_cases": ["route", "missing"]},
            {"test_results": report["test_results"] * 2},
            {"profile": "foreign"},
            {
                "test_results": [
                    {"Name": "route", "Passed": True},
                    {"Name": "cache", "Passed": False},
                ]
            },
            {"total_tests": 3},
            {"status": "FAILED"},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                kubernetes({**report, **change}, "envoy-ai-gateway")

    def test_go_discovery_and_subtests_are_both_mandatory(self):
        discovery = [
            {"Action": "output", "Package": "operator/api", "Output": "TestReconcile\n"}
        ]
        events = [
            {"Action": action, "Package": "operator/api", "Test": name}
            for name in ("TestReconcile", "TestReconcile/routing")
            for action in ("run", "pass")
        ]
        self.assertEqual(len(go_unit(discovery, events)["cases"]), 2)
        for mutation in (
            events[:-1],
            [],
            [*events, events[-1]],
            [
                (
                    {**row, "Action": "skip"}
                    if row["Test"].endswith("routing") and row["Action"] == "pass"
                    else row
                )
                for row in events
            ],
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                go_unit(discovery, mutation)

    def test_operator_requires_every_variant_job_and_image_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            unit = {
                "cases": [{"id": "TestReconcile", "status": "passed"}],
                "expected_cases": ["TestReconcile"],
            }
            request = {
                "cases": [{"id": "operator-routed-request", "status": "passed"}],
                "expected_cases": ["operator-routed-request"],
            }
            for name in (
                "operator-unit",
                "operator-bundle",
                "operator-request-memory",
                "operator-request-redis",
            ):
                (directory / name).mkdir()
            (directory / "operator-unit/operator-unit.json").write_text(
                json.dumps(unit)
            )
            (directory / "operator-bundle/artifacts.json").write_text(
                json.dumps([{"id": "image:operator-bundle", "sha256": "a" * 64}])
            )
            for name in ("memory", "redis"):
                base = directory / f"operator-request-{name}"
                (base / "operator-request.json").write_text(json.dumps(request))
                (base / "artifacts.json").write_text(
                    json.dumps([{"id": "image:operator", "sha256": "b" * 64}])
                )
            jobs = {
                name: {"result": "success"}
                for name in (
                    "checks",
                    "bundle-validate",
                    "integration-test",
                )
            }
            variants = [{"cache-backend": name} for name in ("memory", "redis")]
            result = operator(directory, variants, jobs)
            self.assertEqual(len(result["cases"]), 3)
            self.assertEqual(len(result["artifacts"]), 2)
            for job in jobs:
                for state in ("skipped", "cancelled", "failure", "missing"):
                    bad = copy.deepcopy(jobs)
                    if state == "missing":
                        del bad[job]
                    else:
                        bad[job]["result"] = state
                    with self.subTest(job=job, state=state), self.assertRaisesRegex(
                        ValueError, f"prerequisite did not succeed: {job}"
                    ):
                        operator(directory, variants, bad)
            with self.assertRaises(ValueError):
                operator(directory, variants * 2, jobs)
            (directory / "operator-request-redis/operator-request.json").unlink()
            with self.assertRaises(FileNotFoundError):
                operator(directory, variants, jobs)

    def test_operator_checks_share_setup_without_masking_section_failures(self):
        root = Path(__file__).resolve().parents[3]
        jobs = yaml.safe_load((root / ".github/workflows/operator-ci.yml").read_text())[
            "jobs"
        ]
        self.assertFalse({"lint", "test", "manifests"} & jobs.keys())
        self.assertEqual(
            jobs["result"]["needs"], ["checks", "bundle-validate", "integration-test"]
        )
        self.assertEqual(jobs["integration-test"]["strategy"]["max-parallel"], 2)
        steps = jobs["checks"]["steps"]
        setup_steps = [
            step
            for step in steps
            if step.get("uses") == "./.github/actions/setup-go-ci"
        ]
        self.assertEqual(len(setup_steps), 1)
        self.assertEqual(
            setup_steps[0]["with"]["cache-dependency-path"],
            "deploy/operator/go.sum",
        )
        sections = {
            "Check Go Formatting": "dependencies",
            "Vet Operator Code": "dependencies",
            "Lint Operator Code": "golangci_lint",
            "Discover Operator Unit Tests": "dependencies",
            "Run Operator Unit Tests": "discovery",
            "Install controller-gen": "dependencies",
            "Generate manifests and API code": "controller_gen",
            "Verify no changes": "manifests",
        }
        for name, prerequisite in sections.items():
            with self.subTest(section=name):
                step = next(step for step in steps if step.get("name") == name)
                # Explicit status functions keep independent sections running
                # after a sibling failure; failures themselves stay blocking.
                self.assertEqual(
                    step["if"],
                    "${{ !cancelled() && steps."
                    + prerequisite
                    + ".outcome == 'success' }}",
                )
                self.assertFalse(step.get("continue-on-error", False))
        upload = next(
            step
            for step in steps
            if step.get("with", {}).get("name") == "operator-unit"
        )
        self.assertEqual(upload["if"], "always()")
        self.assertTrue(upload["with"]["include-hidden-files"])

    def test_kubernetes_loads_shared_images_after_runner_cleanup(self):
        root = Path(__file__).resolve().parents[3]
        workflow = yaml.safe_load(
            (root / ".github/workflows/integration-test-k8s.yml").read_text()
        )
        steps = workflow["jobs"]["integration-test"]["steps"]
        actions = [step.get("uses", "") for step in steps]
        # setup-kind prunes unused Docker images while reclaiming/relocating disk.
        # Loading first would delete every verified tag before framework use.
        self.assertLess(
            actions.index("./.github/actions/setup-kind"),
            actions.index("./.github/actions/load-ci-images"),
        )
