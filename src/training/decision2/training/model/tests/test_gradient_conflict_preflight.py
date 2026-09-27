import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from training.model.gradient_conflict_preflight import (
    ACCUMULATION,
    COSINE_THRESHOLD,
    MIN_CONFLICT_WINDOWS,
    SCHEMA,
    SEED,
    SOURCE_FILES,
    SOURCE_REVISION,
    TRAIN_COUNT,
    TRAIN_SHA256,
    TRAIN_TOKENS,
    WINDOW_COUNT,
    measure_window,
    select_windows,
    write_receipt,
)
from training.model.plan import epoch_batches


class WindowSelectionTest(unittest.TestCase):
    def test_exact_trainer_order_and_gold_independence(self):
        kinds = ("choice", "noul", "score")
        rows = [
            {
                "id": f"private-row-{index}",
                "input_sha256": f"{index:064x}",
                "task_type": kinds[index % 3],
                "label": index % 2,
                "state": "private prompt",
            }
            for index in range(480)
        ]
        lengths = [100 + index % 7 for index in range(len(rows))]
        windows, manifest, schedule_hash = select_windows(rows, lengths)
        self.assertEqual(len(windows), WINDOW_COUNT)
        self.assertEqual(len(manifest), WINDOW_COUNT)
        self.assertEqual(len(schedule_hash), 64)
        baseline = epoch_batches(
            lengths,
            [],
            epoch=0,
            seed=20260926,
            microbatch=1,
            replay_fraction=0.0,
        )
        for (ordinal, indices), metadata in zip(windows, manifest):
            expected = [
                item[1]
                for batch in baseline[
                    ordinal * ACCUMULATION : (ordinal + 1) * ACCUMULATION
                ]
                for item in batch
            ]
            self.assertEqual(indices, expected)
            self.assertEqual(metadata["rows"], ACCUMULATION)
            self.assertEqual(sum(metadata["type_counts"].values()), ACCUMULATION)
        self.assertNotIn("private-row", json.dumps(manifest))
        self.assertNotIn("private prompt", json.dumps(manifest))
        for row in rows:
            row["label"] = 99
        self.assertEqual(
            (windows, manifest, schedule_hash), select_windows(rows, lengths)
        )

    def test_requires_three_types_and_valid_lengths(self):
        rows = [
            {"id": str(i), "input_sha256": f"{i:064x}", "task_type": "choice"}
            for i in range(128)
        ]
        with self.assertRaisesRegex(ValueError, "eight three-type"):
            select_windows(rows, [10] * len(rows))
        with self.assertRaisesRegex(ValueError, "Invalid TRAIN"):
            select_windows(rows, [10] * (len(rows) - 1))

    def test_receipt_round_trip_is_aggregate(self):
        receipt = {
            "schema_version": SCHEMA,
            "status": "HOLD_TECHNICAL",
            "created_utc": "2026-09-28T00:00:00+00:00",
            "failure_class": "ValueError",
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "receipt.json"
            write_receipt(output, receipt)
            self.assertEqual(json.loads(output.read_text()), receipt)
            self.assertFalse(output.with_name("receipt.json.pending").exists())
            with self.assertRaisesRegex(ValueError, "invalid top-level"):
                write_receipt(output, {**receipt, "gold_label": "private"})
            self.assertEqual(json.loads(output.read_text()), receipt)

    def test_success_plan_schema_rejects_raw_window_content(self):
        from training.model.decision_model import PROMPT_VERSION

        rows = [
            {
                "id": str(index),
                "input_sha256": f"{index:064x}",
                "task_type": ("choice", "noul", "score")[index % 3],
            }
            for index in range(480)
        ]
        _, windows, schedule_hash = select_windows(rows, [100] * len(rows))
        code_files = (
            "gradient_conflict_preflight.py",
            "data.py",
            "decision_model.py",
            "loss.py",
            "plan.py",
        )
        receipt = {
            "schema_version": SCHEMA,
            "status": "PLAN_ONLY",
            "created_utc": "2026-09-28T00:00:00+00:00",
            "source_revision": SOURCE_REVISION,
            "source_files_sha256": SOURCE_FILES,
            "code_sha256": {name: "0" * 64 for name in code_files},
            "train_sha256": TRAIN_SHA256,
            "train_rows": TRAIN_COUNT,
            "train_native_tokens": TRAIN_TOKENS,
            "prompt_version": PROMPT_VERSION,
            "seed": SEED,
            "microbatch": 1,
            "accumulation": ACCUMULATION,
            "planned_control_updates": 466,
            "schedule_sha256": schedule_hash,
            "windows": windows,
            "conflict_cosine_threshold": COSINE_THRESHOLD,
            "min_conflict_windows": MIN_CONFLICT_WINDOWS,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "plan.json"
            write_receipt(output, receipt)
            self.assertEqual(json.loads(output.read_text()), receipt)
            bad = json.loads(json.dumps(receipt))
            bad["windows"][0]["raw_label"] = 1
            with self.assertRaisesRegex(ValueError, "raw or invalid"):
                write_receipt(output, bad)


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "CPU torch is unavailable in this workspace"
)
class GradientMeasurementTest(unittest.TestCase):
    def test_conflict_reconstructs_standard_without_updating_weights(self):
        import torch

        from training.model.decision_model import collate
        from training.model.loss import per_example_loss

        class TinyDecision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 1, bias=False)
                self.head = torch.nn.Identity()
                with torch.no_grad():
                    self.backbone.weight.zero_()

            def forward(self, input_ids, candidate_mask, **unused):
                encoded = torch.nn.functional.one_hot(
                    input_ids[:, 0] % 3, num_classes=3
                ).float()
                logit = self.backbone(encoded)
                return torch.cat((logit, -logit), dim=-1).masked_fill(
                    ~candidate_mask, -float("inf")
                )

        model = TinyDecision().train()
        before = model.backbone.weight.detach().clone()
        items = []
        for task_type, label in (("choice", 0), ("noul", 0), ("score", 1)):
            for index in range(2):
                items.append(
                    {
                        "id": f"{task_type}-{index}",
                        "ids": [1, 2, 3],
                        "candidate_positions": [0, 1],
                        "query_position": 2,
                        "keys": ["0", "1"],
                        "label": label,
                        "task_type": task_type,
                        "teacher_probs": None,
                    }
                )
        result = measure_window(model, items, pad_id=0, device=torch.device("cpu"))
        self.assertLess(result["max_reconstruction_error"], 1e-6)
        self.assertLessEqual(
            result["pairwise_cosines"]["choice_score"], COSINE_THRESHOLD
        )
        self.assertTrue(result["choice_or_score_conflict"])
        torch.testing.assert_close(model.backbone.weight, before, rtol=0, atol=0)
        self.assertIsNone(model.backbone.weight.grad)
        # Independent ordinary backward accumulation is the frozen control
        # calculation. Its backbone norm must match the grouped diagnostic.
        for item in items:
            batch = collate([item], pad_id=0)
            logits = model(**batch)
            loss = per_example_loss(
                logits,
                batch["labels"],
                batch["candidate_mask"],
                objective="ce_brier",
                brier_weight=0.5,
            )["total"].sum() / len(items)
            loss.backward()
        self.assertAlmostEqual(
            result["ordinary_backbone_norm"],
            float(model.backbone.weight.grad.norm().item()),
            places=6,
        )
        torch.testing.assert_close(model.backbone.weight, before, rtol=0, atol=0)
        model.backbone.weight.grad = None
        with self.assertRaisesRegex(ValueError, "all three"):
            measure_window(model, items[:4], pad_id=0, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
