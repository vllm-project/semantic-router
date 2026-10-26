"""Real framework-shaped evidence must retain each selected deployment assertion."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_plan import github_outputs, make_plan
from deployment_test_results import go_unit, kubernetes, operator, operator_source
from release_guard_waiver import GUARD_WAIVER


class DeploymentResultsTests(unittest.TestCase):
    def test_operator_source_runs_before_deployment_approval_without_images(self):
        plan = make_plan(
            ["deploy/operator/controllers/canonical_config_builder.go"],
            source_sha="a" * 40,
            require_approval=True,
        )
        contracts = {record["id"]: record for record in plan["verifications"]}
        self.assertIn("operator.source", contracts)
        self.assertNotIn("operator", contracts)
        self.assertEqual(contracts["operator.source"]["images"], [])
        self.assertEqual(contracts["operator.source"]["runtime"], "none")
        output = github_outputs(plan)
        self.assertEqual(
            [row["id"] for row in json.loads(output["operator-source"])],
            ["operator.source"],
        )
        root = Path(__file__).resolve().parents[3]
        jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())["jobs"]
        self.assertEqual(jobs["operator-source"]["needs"], "plan")
        self.assertIn("operator-source", jobs["gate"]["needs"])

    def test_operator_full_and_affected_plans_use_one_deployment_contract(self):
        plan = make_plan([], source_sha="a" * 40, full=True)
        contracts = {record["id"]: record for record in plan["verifications"]}
        self.assertIn("operator.source", contracts)
        affected = make_plan(
            ["deploy/operator/controllers/canonical_config_builder.go"],
            source_sha="a" * 40,
        )
        runtime = next(
            row for row in affected["verifications"] if row["id"] == "operator"
        )
        for record in (contracts["operator"], runtime):
            self.assertNotIn("integration_matrix", record)
            self.assertEqual(
                record["inventory"], "operator-memory-reconcile-and-routed-request"
            )
            self.assertEqual(record["executor"], "operator")

    def test_release_guard_case_remains_failed_and_other_cases_are_required(self):
        report = {
            "profile": "production-stack",
            "expected_cases": ["routing", "jailbreak-detection"],
            "test_results": [
                {"Name": "routing", "Passed": True},
                {"Name": "jailbreak-detection", "Passed": False},
            ],
            "status": "FAILED",
            "exit_code": 1,
            "total_tests": 2,
            "passed_tests": 1,
            "failed_tests": 1,
        }
        evidence = kubernetes(report, "production-stack", waiver=GUARD_WAIVER)
        self.assertEqual(
            evidence["cases"][-1],
            {"id": "jailbreak-detection", "status": "failed"},
        )
        with self.assertRaises(ValueError):
            kubernetes(report, "production-stack")
        for mutation in (
            {
                "test_results": [
                    {"Name": "routing", "Passed": False},
                    *report["test_results"][1:],
                ]
            },
            {"expected_cases": ["jailbreak-detection"]},
            {"failed_tests": 0},
            {"status": "PASSED"},
            {"exit_code": 0},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                kubernetes(
                    {**report, **mutation}, "production-stack", waiver=GUARD_WAIVER
                )

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

    def test_a_case_that_passed_only_on_a_retry_does_not_count_as_failed(self):
        # `bin/e2e -flake-attempts=2` marks a case that passed on a later attempt
        # with `Flaked: true` and `Attempts: 2`. It is still a pass, so
        # `failed_tests` stays 0 and the #4120 waiver checks keep working.
        report = {
            "profile": "envoy-ai-gateway",
            "expected_cases": ["route", "cache"],
            "test_results": [
                {"Name": "route", "Passed": True, "Attempts": 1, "Flaked": False},
                {"Name": "cache", "Passed": True, "Attempts": 2, "Flaked": True},
            ],
            "status": "PASSED",
            "exit_code": 0,
            "total_tests": 2,
            "passed_tests": 2,
            "failed_tests": 0,
            "flaky_tests": 1,
        }
        evidence = kubernetes(report, "envoy-ai-gateway")
        self.assertEqual(
            evidence["cases"],
            [
                {"id": "route", "status": "passed"},
                {"id": "cache", "status": "passed"},
            ],
        )
        # Counting the retried case as failed would break the runner counters.
        with self.assertRaises(ValueError):
            kubernetes({**report, "failed_tests": 1}, "envoy-ai-gateway")

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

    def test_operator_requires_the_memory_request_prerequisites_and_images(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            request = {
                "cases": [{"id": "operator-routed-request", "status": "passed"}],
                "expected_cases": ["operator-routed-request"],
            }
            for name in ("operator-bundle", "operator-request"):
                (directory / name).mkdir()
            (directory / "operator-bundle/artifacts.json").write_text(
                json.dumps([{"id": "image:operator-bundle", "sha256": "a" * 64}])
            )
            base = directory / "operator-request"
            report = base / "operator-request.json"
            report.write_text(json.dumps(request))
            (base / "artifacts.json").write_text(
                json.dumps([{"id": "image:operator", "sha256": "b" * 64}])
            )
            jobs = {
                name: {"result": "success"}
                for name in (
                    "bundle-image",
                    "integration-test",
                )
            }
            result = operator(directory, jobs)
            self.assertEqual(
                result["cases"],
                [{"id": "memory/operator-routed-request", "status": "passed"}],
            )
            self.assertEqual(len(result["artifacts"]), 2)
            for job in jobs:
                for state in ("skipped", "cancelled", "failure", "missing"):
                    bad = copy.deepcopy(jobs)
                    if state == "missing":
                        del bad[job]
                    else:
                        bad[job]["result"] = state
                    with (
                        self.subTest(job=job, state=state),
                        self.assertRaisesRegex(
                            ValueError, f"prerequisite did not succeed: {job}"
                        ),
                    ):
                        operator(directory, bad)
            for mutation in (
                {"cases": []},
                {"cases": request["cases"] * 2},
                {"cases": [{"id": "operator-routed-request", "status": "skipped"}]},
                {"expected_cases": []},
                {"expected_cases": ["different-contract"]},
            ):
                report.write_text(json.dumps({**request, **mutation}))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    operator(directory, jobs)
            report.unlink()
            with self.assertRaises(FileNotFoundError):
                operator(directory, jobs)

    def test_operator_source_requires_each_section_and_complete_unit_inventory(self):
        unit = {
            "cases": [{"id": "TestReconcile", "status": "passed"}],
            "expected_cases": ["TestReconcile"],
        }
        steps = {
            name: {"outcome": "success"}
            for name in ("vet", "lint", "tests", "generated", "bundle_drift")
        }
        evidence = operator_source(unit, steps)
        self.assertEqual(len(evidence["cases"]), 5)
        self.assertEqual(evidence["cases"][0]["id"], "unit/TestReconcile")
        for name in steps:
            for outcome in ("failure", "cancelled", "skipped", "missing"):
                bad = copy.deepcopy(steps)
                if outcome == "missing":
                    del bad[name]
                else:
                    bad[name]["outcome"] = outcome
                with self.subTest(name=name, outcome=outcome), self.assertRaises(
                    ValueError
                ):
                    operator_source(unit, bad)
        for bad_unit in (
            {"cases": [], "expected_cases": []},
            {**unit, "cases": []},
            {**unit, "cases": [{"id": "TestReconcile", "status": "skipped"}]},
            {**unit, "cases": unit["cases"] * 2},
            {**unit, "expected_cases": ["TestReconcile", "TestMissing"]},
        ):
            with self.subTest(unit=bad_unit), self.assertRaises(ValueError):
                operator_source(bad_unit, steps)

    def test_operator_checks_share_setup_without_masking_section_failures(self):
        root = Path(__file__).resolve().parents[3]
        source = yaml.safe_load(
            (root / ".github/workflows/operator-source.yml").read_text()
        )["jobs"]
        runtime = yaml.safe_load(
            (root / ".github/workflows/operator-ci.yml").read_text()
        )["jobs"]
        self.assertEqual(set(source), {"checks"})
        self.assertEqual(set(runtime), {"bundle-image", "integration-test", "result"})
        self.assertEqual(
            runtime["result"]["needs"], ["bundle-image", "integration-test"]
        )
        self.assertNotIn("strategy", runtime["integration-test"])
        steps = source["checks"]["steps"]
        self.assertEqual(
            sum(step.get("uses") == "actions/setup-go@v5" for step in steps), 1
        )
        self.assertFalse(
            any(
                step.get("uses") == "./.github/actions/load-ci-images" for step in steps
            )
        )
        sections = {
            "Vet Operator Code": "dependencies",
            "Lint Operator Code": "golangci_lint",
            "Discover Operator Unit Tests": "dependencies",
            "Run Operator Unit Tests": "discovery",
            "Install controller-gen": "dependencies",
            "Generate manifests and API code": "controller_gen",
            "Verify no changes": "manifests",
            "Regenerate bundle": "dependencies",
            "Verify bundle is up to date": "bundle",
        }
        for name, prerequisite in sections.items():
            with self.subTest(section=name):
                step = next(step for step in steps if step.get("name") == name)
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
        record = next(
            step
            for step in steps
            if step.get("name") == "Record complete source contract evidence"
        )
        self.assertEqual(record["if"], "always()")
        self.assertIn("operator-source", record["run"])

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
