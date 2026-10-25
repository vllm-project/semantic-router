from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

from check_ci_gate import evaluate_gate  # noqa: E402
from ci_plan import digest, make_plan  # noqa: E402
from ci_policy import approved, partition  # noqa: E402
from ci_results import make_receipt  # noqa: E402
from verification_catalog import full_cpu_ids  # noqa: E402

SHA = "a" * 40
MODEL_PATH = "src/model-runtime/vllm_srun/registry/models.yaml"


def model_plan(authorization=None):
    return make_plan(
        [MODEL_PATH],
        source_sha=SHA,
        require_approval=True,
        authorization=authorization,
    )


def automatic_receipts(plan):
    receipts = []
    for record in plan["verifications"]:
        assert not record["images"], "This fixture only completes automatic checks."
        field = "checks" if record["activity"] in {"quality", "package"} else "cases"
        evidence = {
            field: [{"id": "required", "status": "passed"}],
            f"expected_{field}": ["required"],
            "runtime": record["runtime"],
            "device": record["device"],
            "platform": record["platform"],
        }
        receipts.append(
            make_receipt(
                record,
                evidence,
                source_sha=SHA,
                execution_platform=record["platform"],
                environ={},
            )
        )
    return receipts


def reseal(plan):
    plan["plan_sha256"] = digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


class CIExecutionPolicyTests(unittest.TestCase):
    def test_deferred_contracts_remain_in_the_complete_required_inventory(self):
        unpartitioned = make_plan([MODEL_PATH], source_sha=SHA)
        plan = model_plan()
        self.assertTrue(plan["deferred_verifications"])
        self.assertEqual(
            plan["required_verification_ids"],
            unpartitioned["expected_verification_ids"],
        )
        self.assertFalse(
            set(plan["expected_verification_ids"]) & set(plan["deferred_verifications"])
        )
        self.assertEqual(
            set(plan["expected_verification_ids"])
            | set(plan["deferred_verifications"]),
            set(plan["required_verification_ids"]),
        )

    def test_scope_without_authorization_cannot_enable_models_or_full_cpu(self):
        plan = model_plan({"authorized": False, "scopes": ["affected"], "full": True})
        self.assertEqual(
            plan["deferred_verifications"], model_plan()["deferred_verifications"]
        )
        self.assertFalse(plan["full_cpu"])

    def test_a_models_request_does_not_authorize_unrelated_execution_families(self):
        authorization = {"authorized": True, "scopes": ["models"], "full": False}
        records = {
            "models": {
                "layer": "runtime",
                "executor": "platform",
                "execution": "manual",
            },
            "storage": {
                "layer": "runtime",
                "executor": "storage",
                "execution": "manual",
            },
            "operator": {
                "layer": "integrations",
                "executor": "operator",
                "execution": "manual",
            },
            "performance": {
                "layer": "performance",
                "executor": "performance",
                "execution": "manual",
            },
            "quality": {
                "layer": "source",
                "executor": "quality",
                "execution": "automatic",
            },
        }
        execute, deferred = partition(list(records), records, authorization)
        self.assertEqual(execute, ["models", "quality"])
        self.assertEqual(deferred, ["storage", "operator", "performance"])

    def test_unknown_scope_does_not_fall_back_to_all_execution(self):
        with self.assertRaisesRegex(ValueError, "unknown manual CI scope"):
            approved(
                {"layer": "runtime", "executor": "platform"},
                {"authorized": True, "scopes": ["arbitrary-target"]},
            )

    def test_affected_authorization_releases_every_selected_required_contract(self):
        plan = model_plan({"authorized": True, "scopes": ["affected"], "full": False})
        self.assertEqual(plan["deferred_verifications"], [])
        self.assertEqual(plan["deferred_images"], [])
        self.assertEqual(
            plan["required_verification_ids"], plan["expected_verification_ids"]
        )
        self.assertFalse(plan["full_cpu"])

    def test_full_request_adds_and_executes_the_full_cpu_inventory(self):
        plan = model_plan({"authorized": True, "scopes": [], "full": True})
        self.assertTrue(plan["full_cpu"])
        self.assertFalse(plan["deferred_verifications"])
        self.assertTrue(set(full_cpu_ids()) <= set(plan["expected_verification_ids"]))

    def test_main_push_neither_publishes_nor_runs_manual_contracts(self):
        plan = make_plan([MODEL_PATH], source_sha=SHA, profile="main")
        self.assertTrue(
            all(record["execution"] == "automatic" for record in plan["verifications"])
        )
        self.assertFalse(plan["publish_images"])
        self.assertFalse(plan["publish_helm"])
        self.assertFalse(plan["publish_python"])

    def test_publication_requires_main_and_complete_qualification(self):
        with self.assertRaisesRegex(ValueError, "main profile"):
            make_plan([], source_sha=SHA, publish=True)
        plan = make_plan([], source_sha=SHA, profile="main", publish=True)
        self.assertTrue(plan["full_cpu"])
        self.assertTrue(set(full_cpu_ids()) <= set(plan["expected_verification_ids"]))
        self.assertTrue(plan["publish_images"])
        self.assertTrue(plan["publish_helm"])
        self.assertTrue(plan["publish_python"])


class CIApprovalGateTests(unittest.TestCase):
    def test_successful_automatic_receipts_cannot_hide_deferred_contracts(self):
        plan = model_plan()
        verdict = evaluate_gate(plan, automatic_receipts(plan))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.status, "awaiting-approval")
        self.assertEqual(verdict.errors, ())
        self.assertEqual(verdict.required, tuple(plan["required_verification_ids"]))

    def test_removed_deferred_flag_does_not_erase_the_required_inventory(self):
        plan = model_plan()
        receipts = automatic_receipts(plan)
        plan["deferred_verifications"] = []
        reseal(plan)
        self.assertFalse(evaluate_gate(plan, receipts).passed)

    def test_shrinking_both_partition_lists_cannot_omit_required_models(self):
        plan = model_plan()
        receipts = automatic_receipts(plan)
        plan["deferred_verifications"] = []
        plan["required_verification_ids"] = list(plan["expected_verification_ids"])
        reseal(plan)
        self.assertFalse(evaluate_gate(plan, receipts).passed)

    def test_duplicate_or_overlapping_partitions_fail(self):
        original = model_plan()
        receipts = automatic_receipts(original)
        for field, value in (
            ("deferred_verifications", original["deferred_verifications"] * 2),
            ("required_verification_ids", original["required_verification_ids"] * 2),
            ("deferred_verifications", [original["expected_verification_ids"][0]]),
        ):
            with self.subTest(field=field):
                plan = copy.deepcopy(original)
                plan[field] = value
                reseal(plan)
                self.assertFalse(evaluate_gate(plan, receipts).passed)


if __name__ == "__main__":
    unittest.main()
