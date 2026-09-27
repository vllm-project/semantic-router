import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from training.model.data import digest
from training.model.gradient_projection_parity import (
    CODE_FILES,
    MAX_PROBABILITY_DRIFT,
    PLANNED_UPDATES,
    SCHEMA,
    SELECT_PROBES,
    SELECT_SHA256,
    SOURCE_REVISION,
    TRAIN_SHA256,
    _write_new,
    compare_arms,
    grouped_one_update,
    ordinary_one_update,
    validate_arm,
    validate_comparison,
    validate_plan,
)
from training.model.train import learning_factor


def _vector(index: int, *, flipped: bool = False, drift: float = 0.0):
    probabilities = [0.8 - drift, 0.2 + drift]
    if flipped:
        probabilities.reverse()
    return {
        "token_ids_sha256": f"{index:064x}",
        "prediction_index": 1 if flipped else 0,
        "probabilities": probabilities,
    }


def _arm(name: str):
    return {
        "schema_version": SCHEMA,
        "status": "ARM_COMPLETE",
        "created_utc": "2026-09-28T00:00:00+00:00",
        "arm": name,
        "plan_sha256": "a" * 64,
        "code_sha256": dict.fromkeys(CODE_FILES, "b" * 64),
        "source_revision": SOURCE_REVISION,
        "train_sha256": TRAIN_SHA256,
        "select_sha256": SELECT_SHA256,
        "select32_sha256": digest([f"{index:064x}" for index in range(SELECT_PROBES)]),
        "planned_updates": PLANNED_UPDATES,
        "learning_factor": learning_factor(0, PLANNED_UPDATES, 0.05),
        "optimizer_steps": 1,
        "unclipped_gradient_norm": 0.3,
        "finite_gradients": True,
        "zero_step": [_vector(index) for index in range(SELECT_PROBES)],
        "post_step": [_vector(index) for index in range(SELECT_PROBES)],
        "device_seconds": 1.0,
        "device_gpu_hours": 1 / 3600,
    }


class ReceiptTest(unittest.TestCase):
    def test_sealed_arm_comparison_and_immutable_output(self):
        ordinary = _arm("ordinary")
        grouped = _arm("grouped")
        result = compare_arms(ordinary, grouped)
        self.assertEqual(result["status"], "PASS_ONE_UPDATE_PARITY")
        self.assertEqual(result["post_step"]["categorical_changes"], 0)
        self.assertEqual(result["post_step"]["max_probability_diff"], 0)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            _write_new(target, result)
            self.assertEqual(json.loads(target.read_text()), result)
            with self.assertRaisesRegex(ValueError, "preserved"):
                _write_new(target, result)

    def test_probability_and_category_gates_are_frozen(self):
        ordinary = _arm("ordinary")
        grouped = _arm("grouped")
        grouped["post_step"][0] = _vector(0, drift=MAX_PROBABILITY_DRIFT * 2)
        result = compare_arms(ordinary, grouped)
        self.assertEqual(result["status"], "HOLD_ONE_UPDATE_PARITY")
        self.assertGreater(
            result["post_step"]["max_probability_diff"], MAX_PROBABILITY_DRIFT
        )
        grouped["post_step"][0] = _vector(0, flipped=True)
        result = compare_arms(ordinary, grouped)
        self.assertEqual(result["post_step"]["categorical_changes"], 1)
        self.assertEqual(result["status"], "HOLD_ONE_UPDATE_PARITY")
        result["status"] = "PASS_ONE_UPDATE_PARITY"
        with self.assertRaisesRegex(ValueError, "status disagrees"):
            validate_comparison(result)

    def test_receipts_reject_gold_raw_content_and_mismatched_inputs(self):
        ordinary = _arm("ordinary")
        grouped = _arm("grouped")
        bad = copy.deepcopy(ordinary)
        bad["zero_step"][0]["gold_label"] = 1
        with self.assertRaisesRegex(ValueError, "raw or gold"):
            validate_arm(bad)
        bad = copy.deepcopy(ordinary)
        bad["prompt"] = "private content"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_arm(bad)
        grouped["code_sha256"][CODE_FILES[0]] = "d" * 64
        with self.assertRaisesRegex(ValueError, "different inputs"):
            compare_arms(ordinary, grouped)

    def test_plan_rejects_unpinned_schema(self):
        from training.model.decision_model import PROMPT_VERSION, TASK_TYPES
        from training.model.gradient_projection_parity import (
            ACCUMULATION,
            SEED,
            SOURCE_FILES,
            TRAIN_COUNT,
            TRAIN_TOKENS,
        )

        plan = {
            "schema_version": SCHEMA,
            "status": "PLAN_ONLY",
            "created_utc": "2026-09-28T00:00:00+00:00",
            "source_revision": SOURCE_REVISION,
            "source_files_sha256": SOURCE_FILES,
            "source_all_files_sha256": SOURCE_FILES,
            "code_sha256": dict.fromkeys(CODE_FILES, "b" * 64),
            "train_sha256": TRAIN_SHA256,
            "select_sha256": SELECT_SHA256,
            "train_rows": TRAIN_COUNT,
            "train_native_tokens": TRAIN_TOKENS,
            "select_probes": SELECT_PROBES,
            "prompt_version": PROMPT_VERSION,
            "seed": SEED,
            "accumulation": ACCUMULATION,
            "planned_updates": PLANNED_UPDATES,
            "first_step_learning_factor": learning_factor(0, PLANNED_UPDATES, 0.05),
            "schedule_sha256": "c" * 64,
            "first_window_sha256": "d" * 64,
            "first_window_type_counts": {
                kind: count for kind, count in zip(TASK_TYPES, (8, 7, 1))
            },
            "select32_sha256": "e" * 64,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        }
        validate_plan(plan)
        plan["first_step_learning_factor"] = 1.0
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_plan(plan)
        plan["first_step_learning_factor"] = learning_factor(0, PLANNED_UPDATES, 0.05)
        plan["raw_label"] = 0
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_plan(plan)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch unavailable")
class OneUpdateTest(unittest.TestCase):
    def test_independent_ordinary_and_grouped_updates_match_without_projection(self):
        import torch

        from training.model.gradient_projection_parity import _predictions

        class TinyDecision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 4)
                self.head = torch.nn.Linear(4, 2)

            def forward(self, input_ids, candidate_mask, **unused):
                encoded = torch.nn.functional.one_hot(
                    input_ids[:, 0] % 3, num_classes=3
                ).float()
                return self.head(torch.tanh(self.backbone(encoded))).masked_fill(
                    ~candidate_mask, -float("inf")
                )

        def optimizer(model):
            opt = torch.optim.AdamW(
                [
                    {
                        "params": list(model.backbone.parameters()),
                        "lr": 2e-5,
                        "peak_lr": 2e-5,
                        "name": "backbone",
                    },
                    {
                        "params": list(model.head.parameters()),
                        "lr": 2e-4,
                        "peak_lr": 2e-4,
                        "name": "head",
                    },
                ],
                weight_decay=0.01,
                foreach=True,
            )
            for group in opt.param_groups:
                group["lr"] = group["peak_lr"] * learning_factor(
                    0, PLANNED_UPDATES, 0.05
                )
            return opt

        torch.manual_seed(20260926)
        control = TinyDecision()
        torch.manual_seed(20260926)
        grouped = TinyDecision()
        items = []
        for index in range(16):
            kind = ("choice", "noul", "score")[index % 3]
            items.append(
                {
                    "id": f"toy-{index}",
                    "ids": [index % 3, 1, 2],
                    "candidate_positions": [0, 1],
                    "query_position": 2,
                    "keys": ["0", "1"],
                    "label": index % 2,
                    "task_type": kind,
                    "teacher_probs": None,
                    "token_ids_sha256": f"{index:064x}",
                }
            )
        probes = copy.deepcopy(items)
        for item in probes:
            item["label"] = 0
        device = torch.device("cpu")
        zero_a = _predictions(control, probes, 0, device)
        zero_b = _predictions(grouped, probes, 0, device)
        self.assertEqual(zero_a, zero_b)
        normal_norm = ordinary_one_update(control, optimizer(control), items, 0, device)
        grouped_norm = grouped_one_update(grouped, optimizer(grouped), items, 0, device)
        self.assertAlmostEqual(normal_norm, grouped_norm, places=6)
        for a, b in zip(control.parameters(), grouped.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
        post_a = _predictions(control, probes, 0, device)
        post_b = _predictions(grouped, probes, 0, device)
        self.assertTrue(
            all(
                a["prediction_index"] == b["prediction_index"]
                for a, b in zip(post_a, post_b)
            )
        )
        self.assertLessEqual(
            max(
                abs(x - y)
                for a, b in zip(post_a, post_b)
                for x, y in zip(a["probabilities"], b["probabilities"])
            ),
            MAX_PROBABILITY_DRIFT,
        )
        # SELECT labels are not used in prediction; an arbitrary label change
        # leaves the gold-free vector unchanged.
        for item in probes:
            item["label"] = 1
        self.assertEqual(_predictions(control, probes, 0, device), post_a)


if __name__ == "__main__":
    unittest.main()
