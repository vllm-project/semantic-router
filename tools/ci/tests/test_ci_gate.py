from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
from check_ci_gate import evaluate_gate  # noqa: E402
from ci_plan import digest, github_outputs, make_plan  # noqa: E402
from ci_results import make_receipt  # noqa: E402
from execution_batches import ALL_DISPATCH_JOBS  # noqa: E402
from provider_mocker_image import IMAGE, REGISTRY  # noqa: E402
from validate_workflows import load_workflows  # noqa: E402
from workflow_policy_validation import validate_gate_transport  # noqa: E402

SHA = "a" * 40
MAX_GATE_ENV_BYTES = 4096


def completed(paths=None, *, full=False):
    plan = make_plan(paths or ["tools/make/openvino.mk"], source_sha=SHA, full=full)
    builds = [
        {"id": f"image:{name}", "sha256": "b" * 64, "source_sha": SHA}
        for name in plan["images"]
    ]
    if IMAGE in plan["image_sources"]:
        source = plan["image_sources"][IMAGE]
        if source["source"] == "published":
            source.update(
                {
                    "registry_digest": "sha256:" + "d" * 64,
                    "ref": REGISTRY + "@sha256:" + "d" * 64,
                    "image_source_sha": "e" * 40,
                    "images": [
                        {
                            "platform": "linux/amd64",
                            "manifest": "sha256:" + "f" * 64,
                            "config": "sha256:" + "1" * 64,
                        }
                    ],
                }
            )
        build = next(item for item in builds if item["id"] == "image:" + IMAGE)
        build.update(
            {
                key: value
                for key, value in source.items()
                if key not in {"id", "source", "lookup_ref"}
            }
        )
        build["acquisition"] = source["source"]
        plan["plan_sha256"] = digest(
            {k: v for k, v in plan.items() if k != "plan_sha256"}
        )
    if plan["native"]:
        builds.append({"id": "native:cpu", "sha256": "c" * 64, "source_sha": SHA})
    receipts = []
    for record in plan["verifications"]:
        field = "checks" if record["activity"] in {"quality", "package"} else "cases"
        evidence = {
            field: [{"id": "required", "status": "passed"}],
            f"expected_{field}": ["required"],
            "runtime": record["runtime"],
            "device": record["device"],
            "platform": record["platform"],
            **({"execution": record["execution"]} if record.get("execution") else {}),
            "artifacts": [
                {"id": item["id"], "sha256": item["sha256"]}
                for item in builds
                if item["id"]
                in {
                    *(f"image:{image}" for image in record["images"]),
                    *(["native:cpu"] if record["native"] else []),
                }
            ],
        }
        receipts.append(
            make_receipt(
                record,
                evidence,
                source_sha=SHA,
                execution_platform="linux/amd64",
                environ={},
            )
        )
    return plan, receipts, builds


class GateTests(unittest.TestCase):
    def test_complete_plan_passes(self):
        for full in (False, True):
            plan, receipts, builds = completed(full=full)
            self.assertTrue(evaluate_gate(plan, receipts, builds=builds).passed)

    def test_published_fixture_requires_planned_input_and_registry_identity(self):
        plan, receipts, builds = completed(["e2e/testing/run_memory_integration.sh"])
        fixture = next(item for item in builds if item["id"] == "image:" + IMAGE)
        self.assertNotEqual(fixture["image_source_sha"], SHA)
        self.assertTrue(evaluate_gate(plan, receipts, builds=builds).passed)
        for field in (
            "inputs_sha256",
            "registry_digest",
            "image_source_sha",
            "ref",
            "images",
            "acquisition",
            "source_sha",
        ):
            candidate = copy.deepcopy(builds)
            next(item for item in candidate if item["id"] == "image:" + IMAGE)[
                field
            ] = "wrong"
            with self.subTest(field=field):
                self.assertFalse(evaluate_gate(plan, receipts, builds=candidate).passed)

    def test_missing_result_cannot_be_hidden_by_omitting_needs(self):
        plan, receipts, builds = completed()
        removed = receipts.pop()
        verdict = evaluate_gate(plan, receipts, builds=builds)
        self.assertFalse(verdict.passed)
        self.assertIn(f"required verification {removed['id']}: missing", verdict.errors)

    def test_duplicate_result_does_not_fill_a_missing_result(self):
        plan, receipts, builds = completed()
        receipts[-1] = receipts[0]
        verdict = evaluate_gate(plan, receipts, builds=builds)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("duplicate" in error for error in verdict.errors))
        self.assertTrue(any("missing" in error for error in verdict.errors))

    def test_source_runtime_device_and_platform_must_match(self):
        plan, receipts, builds = completed()
        for field in ("source_sha", "runtime", "device", "platform", "contract_sha256"):
            candidate = copy.deepcopy(receipts)
            candidate[-1][field] = "wrong"
            with self.subTest(field=field):
                self.assertFalse(evaluate_gate(plan, candidate, builds=builds).passed)

    def test_required_skip_failure_zero_and_missing_case_fail(self):
        plan, receipts, builds = completed()
        for cases in (
            [],
            [{"id": "required", "status": "skipped"}],
            [{"id": "required", "status": "failed"}],
            [{"id": "other", "status": "passed"}],
            [{"id": "required", "status": "passed"}] * 2,
        ):
            candidate = copy.deepcopy(receipts)
            candidate[-1]["evidence"]["cases"] = cases
            candidate[-1]["evidence_sha256"] = digest(candidate[-1]["evidence"])
            self.assertFalse(
                evaluate_gate(plan, candidate, builds=builds).passed, cases
            )

    def test_supplemental_subtests_do_not_replace_required_parent(self):
        plan, receipts, builds = completed()
        receipts[-1]["evidence"]["cases"].append(
            {"id": "required/subtest", "status": "passed"}
        )
        receipts[-1]["evidence_sha256"] = digest(receipts[-1]["evidence"])
        self.assertTrue(evaluate_gate(plan, receipts, builds=builds).passed)
        receipts[-1]["evidence"]["cases"].pop(0)
        receipts[-1]["evidence_sha256"] = digest(receipts[-1]["evidence"])
        self.assertFalse(evaluate_gate(plan, receipts, builds=builds).passed)

    def test_full_inventory_cannot_shrink_with_the_plan(self):
        plan, receipts, builds = completed(full=True)
        missing = plan["expected_verification_ids"].pop()
        plan["verifications"] = [
            record for record in plan["verifications"] if record["id"] != missing
        ]
        receipts = [record for record in receipts if record["id"] != missing]
        plan["plan_sha256"] = digest(
            {k: v for k, v in plan.items() if k != "plan_sha256"}
        )
        self.assertFalse(evaluate_gate(plan, receipts, builds=builds).passed)

    def test_wrong_missing_or_duplicate_build_identity_fails(self):
        plan, receipts, builds = completed(["e2e/testing/run_memory_integration.sh"])
        self.assertTrue(evaluate_gate(plan, receipts, builds=builds).passed)
        for candidate in (
            builds[:-1],
            [*builds, builds[0]],
            [{**item, "sha256": "d" * 64} for item in builds],
        ):
            self.assertFalse(evaluate_gate(plan, receipts, builds=candidate).passed)

    def test_skipped_failed_cancelled_executor_is_never_success(self):
        plan, receipts, builds = completed()
        jobs = {name: {"result": "success"} for name in plan["expected_dispatch_jobs"]}
        self.assertTrue(evaluate_gate(plan, receipts, builds=builds, jobs=jobs).passed)
        for executor in ("native-shared", "native-build"):
            for status in ("skipped", "failure", "cancelled"):
                jobs[executor]["result"] = status
                self.assertFalse(
                    evaluate_gate(plan, receipts, builds=builds, jobs=jobs).passed
                )
            jobs[executor]["result"] = "success"

    def test_draft_is_explicit_not_inferred_from_missing_results(self):
        plan = make_plan(["README.md"], source_sha=SHA, draft=True)
        self.assertTrue(
            evaluate_gate(plan, [], jobs={"plan": {"result": "success"}}).passed
        )
        self.assertFalse(
            evaluate_gate(plan, [], jobs={"plan": {"result": "failure"}}).passed
        )
        plan = make_plan(["README.md"], source_sha=SHA)
        self.assertFalse(evaluate_gate(plan, []).passed)

    def test_dispatch_cannot_hide_a_selected_worker_or_move_a_contract(self):
        plan, receipts, builds = completed(full=True)
        for mutate in (
            lambda value: value["expected_dispatch_jobs"].remove("native-shared"),
            lambda value: value["native_batches"].pop(),
            lambda value: value["e2e_batches"].pop(),
            lambda value: value["image_producers"].update({"image-router": []}),
            lambda value: value.update(full_cpu_version=1),
        ):
            candidate = copy.deepcopy(plan)
            mutate(candidate)
            candidate["plan_sha256"] = digest(
                {key: value for key, value in candidate.items() if key != "plan_sha256"}
            )
            self.assertFalse(evaluate_gate(candidate, receipts, builds=builds).passed)

    def test_successful_batch_does_not_replace_a_missing_individual_receipt(self):
        plan, receipts, builds = completed(full=True)
        jobs = {name: {"result": "success"} for name in plan["expected_dispatch_jobs"]}
        self.assertTrue(evaluate_gate(plan, receipts, builds=builds, jobs=jobs).passed)
        for identity in ("native.image-calibration-cpu", "e2e.vela-omni"):
            selected = [receipt for receipt in receipts if receipt["id"] != identity]
            verdict = evaluate_gate(plan, selected, builds=builds, jobs=jobs)
            self.assertFalse(verdict.passed)
            self.assertIn(f"required verification {identity}: missing", verdict.errors)

    def test_cli_missing_artifact_fails_and_explains(self):
        plan, _, _ = completed()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "plan.json").write_text(json.dumps(plan))
            summary = path / "summary.md"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/ci/check_ci_gate.py"),
                    "--plan",
                    str(path / "plan.json"),
                    "--results",
                    str(path),
                ],
                env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("required verification", summary.read_text())


class GateTransportTests(unittest.TestCase):
    def setUp(self):
        errors = []
        self.gate = load_workflows(errors)["ci.yml"].jobs["gate"]
        self.assertEqual(errors, [])
        self.step = next(
            step
            for step in self.gate["steps"]
            if "check_ci_gate.py" in step.get("run", "")
        )

    def environment(self, jobs):
        def substitute(match):
            return jobs[match[1]]["result"]

        return {
            key: re.sub(r"\$\{\{ needs\.([\w-]+)\.result \}\}", substitute, value)
            for key, value in self.step["env"].items()
        }

    def test_full_plan_outputs_never_enter_process_environment(self):
        plan, receipts, builds = completed(full=True)
        jobs = {
            name: {
                "result": (
                    "success" if name in plan["expected_dispatch_jobs"] else "skipped"
                )
            }
            for name in ALL_DISPATCH_JOBS
        }
        jobs["plan"]["outputs"] = {
            **github_outputs(plan),
            "extra_large_output": "x" * (256 * 1024),
        }
        self.assertGreater(len(json.dumps(jobs).encode()), 128 * 1024)
        environment = self.environment(jobs)
        self.assertTrue(
            all(
                len(value.encode()) < MAX_GATE_ENV_BYTES
                for value in environment.values()
            )
        )
        projected = json.loads(environment["EXECUTOR_RESULTS"])
        self.assertEqual(
            projected, {name: {"result": job["result"]} for name, job in jobs.items()}
        )
        self.assertTrue(
            evaluate_gate(plan, receipts, builds=builds, jobs=projected).passed
        )
        projected["image-distribution"]["result"] = "failure"
        self.assertFalse(
            evaluate_gate(plan, receipts, builds=builds, jobs=projected).passed
        )

    def test_workflow_policy_rejects_oversized_or_incomplete_transport(self):
        errors = []
        validate_gate_transport(self.gate, errors)
        self.assertEqual(errors, [])
        expected = json.loads(self.step["env"]["EXECUTOR_RESULTS"])
        incomplete = {
            name: value for name, value in expected.items() if name != "native-build"
        }
        with_output = {**expected, "outputs": "${{ needs.plan.outputs.plan }}"}
        for value in (
            "${{ toJSON(needs) }}",
            json.dumps(incomplete),
            json.dumps(with_output),
        ):
            with self.subTest(value=value[:50]):
                self.step["env"]["EXECUTOR_RESULTS"] = value
                errors = []
                validate_gate_transport(self.gate, errors)
                self.assertTrue(any("without job outputs" in error for error in errors))

    def test_plan_cannot_move_back_from_artifact_to_environment(self):
        self.gate["steps"] = [
            step
            for step in self.gate["steps"]
            if step.get("with", {}).get("name") != "ci-plan"
        ]
        errors = []
        validate_gate_transport(self.gate, errors)
        self.assertTrue(any("independently downloaded" in error for error in errors))

    def test_shell_reconciles_large_file_plan_and_rejects_missing_or_changed_evidence(
        self,
    ):
        plan, receipts, builds = completed(full=True)
        plan["paths"] = [
            f"website/docs/large-change-{index}.md" for index in range(5000)
        ]
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        jobs = {
            name: {
                "result": (
                    "success" if name in plan["expected_dispatch_jobs"] else "skipped"
                )
            }
            for name in ALL_DISPATCH_JOBS
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / ".agent-harness/ci"
            results = directory / "results"
            results.mkdir(parents=True)
            native_bytes = json.dumps({"platform": "linux/amd64"}).encode()
            native_digest = hashlib.sha256(native_bytes).hexdigest()
            for build in builds:
                if build["id"] == "native:cpu":
                    output = results / "ci-build-native-cpu"
                    output.mkdir()
                    (output / "manifest.json").write_bytes(native_bytes)
                    (output / "receipt.json").write_text(
                        json.dumps([{**build, "sha256": native_digest}])
                    )
                else:
                    name = build["id"].removeprefix("image:")
                    output = results / f"ci-build-image-{name}"
                    output.mkdir()
                    (output / "manifest.json").write_text(
                        json.dumps(
                            {
                                "images": [{"platform": "linux/amd64"}],
                                **build,
                                "id": name,
                            }
                        )
                    )
            for receipt in receipts:
                for collection in (
                    receipt["artifacts"],
                    receipt["evidence"]["artifacts"],
                ):
                    for artifact in collection:
                        if artifact["id"] == "native:cpu":
                            artifact["sha256"] = native_digest
                receipt["evidence_sha256"] = digest(receipt["evidence"])
                output = results / f"ci-result-{receipt['id']}"
                output.mkdir()
                (output / "result.json").write_text(json.dumps(receipt))
            plan_path = directory / "plan.json"
            plan_path.write_text(json.dumps(plan))
            self.assertGreater(plan_path.stat().st_size, 128 * 1024)
            script = self.step["run"].replace(
                "python3 tools/ci/check_ci_gate.py",
                f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'tools/ci/check_ci_gate.py'))}",
            )
            environment = {**os.environ, **self.environment(jobs), "GITHUB_SHA": SHA}
            environment.pop("GITHUB_STEP_SUMMARY", None)

            def run():
                return subprocess.run(
                    ["bash", "-e", "-c", script],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            actual = run()
            self.assertEqual(actual.returncode, 0, actual.stdout + actual.stderr)
            environment["GITHUB_SHA"] = "f" * 40
            self.assertIn("plan source SHA differs", run().stdout)
            environment["GITHUB_SHA"] = SHA
            plan["paths"].append("tampered")
            plan_path.write_text(json.dumps(plan))
            self.assertIn("plan schema or content digest is invalid", run().stdout)
            plan["paths"].pop()
            plan_path.write_text(json.dumps(plan))
            (results / f"ci-result-{receipts[-1]['id']}" / "result.json").unlink()
            actual = run()
            self.assertEqual(actual.returncode, 1)
            self.assertIn(
                f"required verification {receipts[-1]['id']}: missing", actual.stdout
            )


if __name__ == "__main__":
    unittest.main()
