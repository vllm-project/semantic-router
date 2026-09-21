from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
from check_ci_gate import evaluate_gate  # noqa: E402
from ci_plan import digest, make_plan  # noqa: E402
from ci_results import make_receipt  # noqa: E402
from provider_mocker_image import IMAGE, REGISTRY  # noqa: E402

SHA = "a" * 40


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
        jobs = {
            "plan": {"result": "success"},
            **{
                record["executor"]: {"result": "success"}
                for record in plan["verifications"]
            },
        }
        if plan["native"]:
            jobs["native-build"] = {"result": "success"}
        if plan["images"]:
            jobs["images"] = {"result": "success"}
        self.assertTrue(evaluate_gate(plan, receipts, builds=builds, jobs=jobs).passed)
        for executor in ("native", "native-build"):
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


if __name__ == "__main__":
    unittest.main()
