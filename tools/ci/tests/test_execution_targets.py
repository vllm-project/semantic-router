"""Placement changes cannot silently upgrade CPU evidence or mix workers."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

from check_ci_gate import evaluate_gate  # noqa: E402
from ci_plan import digest, make_plan  # noqa: E402
from ci_results import make_receipt  # noqa: E402
from execution_batches import (  # noqa: E402
    component_batches,
    e2e_batches,
    validate_component_batch,
    validate_execution_batch,
)
from execution_targets import (  # noqa: E402
    execution_observation,
    load_execution_targets,
    observation_errors,
    resolve_execution_target,
    target_errors,
)
from verification_catalog import load_catalog  # noqa: E402

SHA = "a" * 40
HOSTED = {"GITHUB_ACTIONS": "true", "CI_RUNNER_ENVIRONMENT": "github-hosted"}


def record_for(name="cli-unit"):
    return make_plan([], source_sha=SHA, requested=(name,))["verifications"][0]


def reseal(record):
    record["contract_sha256"] = digest(
        {key: value for key, value in record.items() if key != "contract_sha256"}
    )
    return record


class ExecutionTargetTests(unittest.TestCase):
    def test_only_the_supported_cpu_target_is_enabled(self):
        targets = load_execution_targets()["targets"]
        self.assertEqual(
            [name for name, value in targets.items() if value["enabled"]],
            ["linux-amd64-cpu"],
        )
        record = record_for()
        self.assertEqual(record["runner"], "ubuntu-24.04")
        self.assertEqual(record["execution_adapter"], "cpu-v1")
        self.assertEqual(target_errors(record), [])
        self.assertEqual(
            record["contract_sha256"], reseal(dict(record))["contract_sha256"]
        )

    def test_disabled_unknown_or_unsupported_targets_fail_closed(self):
        record = record_for()
        registry = copy.deepcopy(load_execution_targets())
        target = registry["targets"][record["execution_target"]]
        for field, value in (
            ("enabled", False),
            ("adapter", "label-only-gpu"),
            ("devices", ["cpu", "rocm"]),
            ("runtimes", ["unimplemented-runtime"]),
            ("runner", []),
            ("runner_environment", "unknown"),
        ):
            candidate = copy.deepcopy(registry)
            candidate["targets"][record["execution_target"]] = {**target, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_execution_target(record, registry=candidate)
        for field, value in (
            ("execution_target", "missing"),
            ("device", "cuda"),
            ("runtime", "unimplemented-runtime"),
            ("executor", "arbitrary-command"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_execution_target({**record, field: value})

    def test_resolved_placement_cannot_override_the_registry(self):
        record = record_for()
        for field, value in (
            ("runner", ["self-hosted"]),
            ("platform", "linux/arm64"),
            ("execution_adapter", "other"),
            ("execution_target_sha256", "b" * 64),
        ):
            with self.subTest(field=field):
                self.assertTrue(target_errors({**record, field: value}))

    def test_a_reviewed_cpu_placement_does_not_require_a_new_test_contract(self):
        record = record_for()
        registry = copy.deepcopy(load_execution_targets())
        registry["targets"][record["execution_target"]].update(
            runner=["self-hosted", "linux", "cpu-tests"],
            runner_environment="self-hosted",
        )
        moved = resolve_execution_target(record, registry=registry)
        self.assertEqual(moved["inventory"], record["inventory"])
        self.assertEqual(moved["device"], record["device"])
        self.assertNotEqual(
            moved["execution_target_sha256"], record["execution_target_sha256"]
        )
        self.assertEqual(target_errors(moved, registry=registry), [])

    def test_actions_environment_and_platform_must_be_observed(self):
        record = record_for()
        observed = execution_observation(record, "linux/amd64", environ=HOSTED)
        self.assertEqual(observation_errors(record, observed, require_actions=True), [])
        for environment in (
            {"GITHUB_ACTIONS": "true"},
            {"GITHUB_ACTIONS": "true", "CI_RUNNER_ENVIRONMENT": "self-hosted"},
        ):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                execution_observation(record, "linux/amd64", environ=environment)
        with self.assertRaises(ValueError):
            execution_observation(record, "linux/arm64", environ=HOSTED)
        local = execution_observation(record, "linux/amd64", environ={})
        self.assertEqual(local["runner_environment"], "local")
        self.assertTrue(observation_errors(record, local, require_actions=True))

    def test_target_and_observation_are_required_by_the_gate(self):
        plan = make_plan([], source_sha=SHA, requested=("cli-unit",))
        record = plan["verifications"][0]
        evidence = {
            "runtime": "none",
            "device": "none",
            "platform": "linux/amd64",
            "cases": [{"id": "required", "status": "passed"}],
            "expected_cases": ["required"],
        }
        receipt = make_receipt(
            record,
            evidence,
            source_sha=SHA,
            execution_platform="linux/amd64",
            environ=HOSTED,
        )
        self.assertTrue(evaluate_gate(plan, [receipt], require_actions=True).passed)
        for field, value in (
            ("execution_target", "other"),
            ("execution_target_sha256", "b" * 64),
            ("execution_adapter", "other"),
            ("runner", "another-runner"),
            ("execution", None),
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    evaluate_gate(
                        plan, [{**receipt, field: value}], require_actions=True
                    ).passed
                )
        local = copy.deepcopy(receipt)
        local["execution"]["runner_environment"] = "local"
        self.assertFalse(evaluate_gate(plan, [local], require_actions=True).passed)

    def test_required_gpu_skips_cannot_become_cpu_successes(self):
        record = record_for()
        evidence = {
            "runtime": "none",
            "device": "none",
            "platform": "linux/amd64",
            "cases": [{"id": "gpu-required", "status": "skipped"}],
            "expected_cases": ["gpu-required"],
        }
        with self.assertRaisesRegex(ValueError, "skipped"):
            make_receipt(
                record,
                evidence,
                source_sha=SHA,
                execution_platform="linux/amd64",
                environ=HOSTED,
            )


class BatchPolicyTests(unittest.TestCase):
    def test_components_are_independent_until_explicitly_opted_in(self):
        plan = make_plan([], source_sha=SHA, requested=("onnx-artifacts", "training"))
        self.assertEqual(len(plan["component_batches"]), 2)
        shared = [
            reseal({**row, "batch_policy": "shared-components"})
            for row in plan["verifications"]
        ]
        workers = load_catalog()["component_workers"]
        batches = component_batches(shared, workers)
        self.assertEqual(len(batches), 1)
        validate_component_batch(batches[0], workers)

    def test_policy_and_setup_environment_split_otherwise_compatible_contracts(self):
        record = record_for("e2e.standalone")
        for field, value in (
            ("batch_policy", "exclusive"),
            ("setup_environment", "isolated-fixture"),
        ):
            records = [record, reseal({**record, "id": "e2e.other", field: value})]
            with self.subTest(field=field):
                batches = e2e_batches(records)
                self.assertEqual(len(batches), 2)
                self.assertEqual(len({row["display_name"] for row in batches}), 2)
                for batch in batches:
                    validate_execution_batch(batch, "e2e")

    def test_target_identity_separates_even_identical_runner_labels(self):
        record = record_for("e2e.standalone")
        registry = copy.deepcopy(load_execution_targets())
        registry["targets"]["second-cpu"] = copy.deepcopy(
            registry["targets"][record["execution_target"]]
        )
        other = reseal(
            resolve_execution_target(
                {**record, "id": "e2e.other", "execution_target": "second-cpu"},
                registry=registry,
            )
        )
        with patch("execution_targets.load_execution_targets", return_value=registry):
            self.assertEqual(len(e2e_batches([record, other])), 2)

    def test_worker_deadline_includes_setup_overhead(self):
        record = record_for("e2e.standalone")
        for policy in ("integration", "exclusive"):
            candidate = reseal(
                {**record, "batch_policy": policy, "timeout_minutes": 330}
            )
            batches = e2e_batches([candidate])
            self.assertEqual(batches[0]["job_timeout_minutes"], 360)
            with self.assertRaises(ValueError):
                e2e_batches([{**candidate, "timeout_minutes": 331}])

    def test_batched_workflows_bind_target_placement(self):
        for name in ("test-tools", "test-platform", "integration-test-k8s"):
            workflow = yaml.safe_load(
                (ROOT / f".github/workflows/{name}.yml").read_text()
            )
            for job in workflow["jobs"].values():
                with self.subTest(workflow=name):
                    self.assertEqual(
                        job["runs-on"], "${{ fromJSON(inputs.batch).runner }}"
                    )

    def test_producer_workflows_observe_runner_in_step_context(self):
        records = make_plan([], source_sha=SHA, full=True)["verifications"]
        workflows = {record["workflow"] for record in records}
        for name in workflows:
            workflow = yaml.safe_load((ROOT / name).read_text())
            for job_id, job in workflow["jobs"].items():
                with self.subTest(workflow=name, job=job_id):
                    self.assertNotIn("runner.", str(job.get("env", {})))
                    first_step = job["steps"][0]
                    self.assertEqual(
                        first_step["env"]["OBSERVED_RUNNER_ENVIRONMENT"],
                        "${{ runner.environment }}",
                    )
                    self.assertEqual(
                        first_step["working-directory"], "${{ github.workspace }}"
                    )
                    self.assertIn(
                        "CI_RUNNER_ENVIRONMENT=$OBSERVED_RUNNER_ENVIRONMENT",
                        first_step["run"],
                    )
                    self.assertIn('>> "$GITHUB_ENV"', first_step["run"])


if __name__ == "__main__":
    unittest.main()
