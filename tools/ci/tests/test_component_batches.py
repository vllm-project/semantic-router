"""Worker batching preserves each selected contract's discovery and failure boundary."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
import run_component_batch as runner  # noqa: E402
from check_ci_gate import evaluate_gate  # noqa: E402
from ci_plan import github_outputs, make_plan  # noqa: E402

SHA = "a" * 40


class ComponentBatchTests(unittest.TestCase):
    def test_ten_contracts_use_three_workers_without_changing_selection(self):
        full = make_plan([], source_sha=SHA, full=True)
        batches = full["component_batches"]
        self.assertEqual([row["id"] for row in batches], ["cli", "model", "router"])
        rows = [row for batch in batches for row in batch["verifications"]]
        selected = [row for row in full["verifications"] if row["executor"] == "tools"]
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["id"] for row in rows}, {row["id"] for row in selected})
        self.assertEqual(json.loads(github_outputs(full)["component_batches"]), batches)
        partial = make_plan([], source_sha=SHA, requested=("ck-rewrite",))
        self.assertEqual(len(partial["component_batches"]), 1)
        self.assertEqual(
            [row["id"] for row in partial["component_batches"][0]["verifications"]],
            ["ck-rewrite"],
        )
        self.assertNotIn("generated_contracts", full["quality_context"])
        self.assertNotIn("soak", full["quality_context"])

    def test_each_contract_keeps_its_own_events_and_source_bound_receipt(self):
        passed, plan, receipts, raw, calls = self.run_contracts(
            ("cli-unit", "fleet-sim")
        )
        self.assertTrue(passed)
        self.assertEqual({row["id"] for row in receipts}, {"cli-unit", "fleet-sim"})
        self.assertTrue(evaluate_gate(plan, receipts).passed)
        event_paths = {
            env["CI_PYTHON_TEST_EVENTS"]
            for _, env in calls
            if "CI_PYTHON_TEST_EVENTS" in env
        }
        self.assertEqual(len(event_paths), 2)
        for row in receipts:
            self.assertEqual(row["source_sha"], SHA)
            expected_cases = 2 if row["id"] == "cli-unit" else 1
            self.assertEqual(len(row["evidence"]["cases"]), expected_cases)
            self.assertIn(row["id"] + "/python-events.jsonl", raw)

    def test_real_subprocess_observers_do_not_leak_between_contracts(self):
        plan = make_plan([], source_sha=SHA, requested=("cli-unit", "fleet-sim"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            (root / "tools/ci").symlink_to(ROOT / "tools/ci", target_is_directory=True)
            (root / "bin").mkdir()
            executable = root / "bin/make"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                "import unittest\n"
                "class ObservedContract(unittest.TestCase):\n"
                "    def test_actual_process(self): self.assertEqual(2 + 2, 4)\n"
                "ObservedContract.__qualname__ = 'Observed_' + sys.argv[1].replace('-', '_')\n"
                "result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ObservedContract))\n"
                "raise SystemExit(not result.wasSuccessful())\n"
            )
            executable.chmod(0o755)
            with patch.object(runner, "ROOT", root), patch.object(
                runner, "install_cpu_torch"
            ), patch.object(
                runner.subprocess, "check_output", return_value=SHA
            ), patch.object(
                runner, "actual_platform", return_value="linux/amd64"
            ), patch.dict(
                os.environ,
                {"PATH": str(root / "bin") + os.pathsep + os.environ["PATH"]},
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                output = root / "batch"
                self.assertTrue(runner.run_batch(plan["component_batches"][0], output))
            receipts = [
                json.loads(path.read_text())
                for path in (output / "results").glob("*.json")
            ]
            self.assertTrue(evaluate_gate(plan, receipts).passed)
            self.assertEqual(len(receipts), 2)
            self.assertEqual(
                {row["id"]: len(row["evidence"]["cases"]) for row in receipts},
                {"cli-unit": 2, "fleet-sim": 1},
            )

    def test_failed_command_does_not_hide_successful_sibling_or_write_receipt(self):
        passed, plan, receipts, raw, calls = self.run_contracts(
            ("cli-unit", "fleet-sim"), failure="vllm-sr-test"
        )
        self.assertFalse(passed)
        self.assertEqual([row["id"] for row in receipts], ["fleet-sim"])
        self.assertFalse(evaluate_gate(plan, receipts).passed)
        self.assertIn("cli-unit/failure.txt", raw)
        self.assertEqual(
            [self.stage(command) for command, _ in calls],
            ["vllm-sr-test", "vllm-sr-sim-test"],
        )

    def test_missing_skipped_or_wrong_source_evidence_cannot_qualify(self):
        for outcome in ("missing", "skipped", "failed", "wrong-source"):
            with self.subTest(outcome=outcome):
                passed, _, receipts, raw, _ = self.run_contracts(
                    ("cli-unit",), outcome=outcome
                )
                self.assertFalse(passed)
                self.assertEqual(receipts, [])
                self.assertIn("cli-unit/failure.txt", raw)

    def test_maintained_entrypoints_all_run_and_each_failure_propagates(self):
        stages = {
            "cli-unit": ["vllm-sr-test", "torch", "vllm-sr-decision-runtime-test"],
            "learning-tools": ["test-learning-tools", "test-calibration"],
            "soak-tools": ["soak-test", "proxy-tests"],
            "mock-provider": ["test-provider-mocker"],
            "e2e-unit": ["test-e2e-unit"],
            "training": ["torch", "training-deps", "test-training-contracts"],
        }
        for identity, expected in stages.items():
            passed, _, receipts, _, calls = self.run_contracts((identity,))
            with self.subTest(identity=identity):
                self.assertTrue(passed)
                self.assertEqual(len(receipts), 1)
                self.assertEqual(
                    [self.stage(command) for command, _ in calls], expected
                )
            for index, failure in enumerate(expected):
                passed, _, receipts, _, calls = self.run_contracts(
                    (identity,), failure=failure
                )
                with self.subTest(identity=identity, failure=failure):
                    self.assertFalse(passed)
                    self.assertEqual(receipts, [])
                    self.assertEqual(
                        [self.stage(command) for command, _ in calls],
                        expected[: index + 1],
                    )

    def test_decision_unit_inventory_requires_cpu_torch_without_rocm_model(self):
        makefile = (ROOT / "tools/make/docker.mk").read_text()
        target = makefile.split(
            "vllm-sr-decision-runtime-test: harness-venv-install", 1
        )[1].split("vllm-sr-test-integration:", 1)[0]
        self.assertIn("test_decision_runtime_torch_batch_sync.py", target)
        self.assertNotIn("test_decision_qwen35_rocm_graph_model.py", target)
        passed, _, receipts, _, calls = self.run_contracts(("cli-unit",))
        self.assertTrue(passed)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(
            [self.stage(command) for command, _ in calls],
            ["vllm-sr-test", "torch", "vllm-sr-decision-runtime-test"],
        )
        self.assertEqual(
            calls[1][0][-2:],
            ["--index-url", "https://download.pytorch.org/whl/cpu"],
        )

    def test_go_and_soak_reports_are_isolated_without_python_observer(self):
        passed, _, receipts, raw, calls = self.run_contracts(("e2e-unit", "soak-tools"))
        self.assertTrue(passed)
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all("CI_PYTHON_TEST_EVENTS" not in env for _, env in calls))
        self.assertIn("e2e-unit/evidence.json", raw)
        self.assertIn("soak-tools/proxy.xml", raw)
        self.assertIn("soak-tools/inventory.jsonl", raw)
        self.assertEqual(
            len(
                next(row for row in receipts if row["id"] == "soak-tools")["evidence"][
                    "cases"
                ]
            ),
            2,
        )

    def test_tampered_or_duplicate_worker_contracts_are_rejected(self):
        batch = make_plan([], source_sha=SHA, requested=("cli-unit",))[
            "component_batches"
        ][0]
        for field, value in (("worker", "model"), ("target", "other-command")):
            changed = copy.deepcopy(batch)
            changed["verifications"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.validate_batch(changed)
        batch["verifications"] *= 2
        with self.assertRaises(ValueError):
            runner.validate_batch(batch)
        with self.assertRaises(ValueError):
            runner.commands("arbitrary-target", Path("output"))

    def test_workflow_uploads_individual_results_after_failure_and_gate_is_unchanged(
        self,
    ):
        data = yaml.safe_load((ROOT / ".github/workflows/test-tools.yml").read_text())
        job = data["jobs"]["tests"]
        self.assertEqual(job["name"], "Execute Contracts")
        go = next(
            step for step in job["steps"] if step.get("uses") == "actions/setup-go@v5"
        )
        self.assertEqual(go["if"], "fromJSON(inputs.batch).go")
        uploads = [
            step
            for step in job["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        ]
        self.assertTrue(all(step["if"] == "always()" for step in uploads))
        self.assertTrue(uploads[0]["with"]["path"].endswith("/results/*.json"))
        parent = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
        self.assertEqual(
            parent["tools"]["strategy"]["matrix"]["label"],
            "${{ fromJSON(needs.plan.outputs.worker_labels)['tools'] }}",
        )
        self.assertEqual(parent["gate"]["if"], "always()")
        self.assertIn("tools", parent["gate"]["needs"])

    @staticmethod
    def stage(command):
        if command[0] == "make":
            return command[1]
        if "pip" in command:
            return "torch" if "torch==2.10.0" in command else "training-deps"
        return "proxy-tests"

    def run_contracts(self, identities, *, failure="", outcome="passed"):
        plan = make_plan([], source_sha=SHA, requested=identities)
        batch = plan["component_batches"][0]
        calls = []

        def process(command, *, env, stdout, **_kwargs):
            calls.append((command, env))
            raw = Path(stdout.name).parent
            stage = self.stage(command)
            if stage == failure:
                return subprocess.CompletedProcess(command, 37)
            if stage == "test-e2e-unit":
                (raw / "evidence.json").write_text(
                    json.dumps(
                        {
                            "runtime": "none",
                            "device": "none",
                            "platform": "linux/amd64",
                            "cases": [{"id": "TestFramework", "status": "passed"}],
                            "expected_cases": ["TestFramework"],
                        }
                    )
                )
            elif stage == "soak-test":
                (raw / "inventory.jsonl").write_text(
                    json.dumps({"Package": "soak", "Output": "TestSoak\n"}) + "\n"
                )
                (raw / "tests.jsonl").write_text(
                    json.dumps(
                        {"Package": "soak", "Test": "TestSoak", "Action": "pass"}
                    )
                    + "\n"
                )
            elif stage == "proxy-tests":
                (raw / "proxy.xml").write_text(
                    '<testsuite><testcase classname="proxy" name="test_fault"/></testsuite>'
                )
            elif "CI_PYTHON_TEST_EVENTS" in env:
                events = [{"kind": "expected", "id": stage}]
                if outcome != "missing":
                    events.append(
                        {
                            "kind": "case",
                            "id": stage,
                            "status": (
                                "passed" if outcome == "wrong-source" else outcome
                            ),
                        }
                    )
                with Path(env["CI_PYTHON_TEST_EVENTS"]).open("a") as stream:
                    stream.write("\n".join(map(json.dumps, events)) + "\n")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner.subprocess, "run", side_effect=process
        ), patch.object(
            runner.subprocess,
            "check_output",
            return_value="b" * 40 if outcome == "wrong-source" else SHA,
        ), patch.object(
            runner, "actual_platform", return_value="linux/amd64"
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            output = Path(directory) / "batch"
            passed = runner.run_batch(batch, output)
            receipts = [
                json.loads(path.read_text())
                for path in sorted((output / "results").glob("*.json"))
            ]
            raw = {
                str(path.relative_to(output / "raw")): path.read_text()
                for path in (output / "raw").rglob("*")
                if path.is_file()
            }
            with self.assertRaises(FileExistsError):
                runner.run_batch(batch, output)
            return passed, plan, receipts, raw, calls


if __name__ == "__main__":
    unittest.main()
