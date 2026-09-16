"""A replay order binds eligible source contents and accounts for every draw."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.sequence_repair.data import file_receipts
from src.training.model_classifier.sequence_repair.train import (
    token_budget_microbatches,
)
from src.training.model_classifier.sequence_repair.training_order import (
    load_training_order,
    ordered_ids_sha256,
)


class TrainingOrderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = [{"id": key, "text": key} for key in ["short", "长文", "unused"]]
        self.train = self.root / "train.jsonl"
        self.train.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.order = self.root / "order.json"
        ids = ["长文", "short", "short", "长文", "长文", "short"]
        self.value = {
            "version": 1,
            "train_files": file_receipts([self.train]),
            "eligible_ids_sha256": ordered_ids_sha256([row["id"] for row in self.rows]),
            "steps": 2,
            "global_batch": 3,
            "ids": ids,
            "ids_sha256": ordered_ids_sha256(ids),
        }

    def load(self, value=None, rows=None):
        self.order.write_text(json.dumps(self.value if value is None else value))
        return load_training_order(
            self.order,
            self.rows if rows is None else rows,
            [self.train],
            steps=2,
            global_batch=3,
        )

    def test_repeated_draws_and_length_microbatches_preserve_planned_order(self):
        order = self.load()
        trace = []
        for step in [1, 2]:
            planned = order.indices_for_step(step)
            batches = token_budget_microbatches(planned, [3, 8, 5], 8)
            trace.append(order.record_step(step, batches))
        self.assertEqual(trace[0]["planned_ids"], ["长文", "short", "short"])
        self.assertEqual(trace[0]["microbatch_ids"], [["short", "short"], ["长文"]])
        self.assertEqual(order.finish()["completed_draws"], 6)
        self.assertEqual(order.receipt["unique_rows"], 2)
        with self.assertRaises(ValueError):
            order.indices_for_step(3)

    def test_incomplete_duplicate_or_skipped_steps_fail(self):
        order = self.load()
        with self.assertRaises(ValueError):
            order.finish()
        with self.assertRaises(ValueError):
            order.indices_for_step(2)
        with self.assertRaises(ValueError):
            order.record_step(1, [[1, 0]])
        with self.assertRaises(ValueError):
            order.record_step(1, [[1, 1, 0]])
        order.record_step(1, [[1, 0, 0]])
        with self.assertRaises(ValueError):
            order.record_step(1, [[1, 0, 0]])
        with self.assertRaises(ValueError):
            order.finish()

    def test_unknown_or_filtered_id_is_not_silently_removed(self):
        value = copy.deepcopy(self.value)
        value["ids"][0] = "filtered"
        value["ids_sha256"] = ordered_ids_sha256(value["ids"])
        with self.assertRaisesRegex(ValueError, "unknown or filtered"):
            self.load(value)
        with self.assertRaisesRegex(ValueError, "eligible TRAIN"):
            self.load(rows=self.rows[:-1])

    def test_changed_source_bytes_fail_even_with_same_ids(self):
        self.train.write_text(self.train.read_text().replace("unused", "changed", 1))
        with self.assertRaisesRegex(ValueError, "TRAIN file contents"):
            self.load()

    def test_eligible_order_and_unique_ids_are_bound(self):
        with self.assertRaisesRegex(ValueError, "eligible TRAIN"):
            self.load(rows=list(reversed(self.rows)))
        for rows in [[self.rows[0], self.rows[0]], [{"id": 1}], [{"id": ""}], []]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load(rows=rows)

    def test_malformed_and_wrong_budget_orders_fail(self):
        variants = []
        for key in self.value:
            missing = copy.deepcopy(self.value)
            del missing[key]
            variants.append(missing)
        variants.extend(
            [
                {**self.value, "unexpected": True},
                {**self.value, "version": True},
                {**self.value, "version": 2},
                {**self.value, "steps": 1},
                {**self.value, "global_batch": 2},
                {**self.value, "steps": 2.0},
                {**self.value, "ids_sha256": "0" * 64},
                {**self.value, "eligible_ids_sha256": "0" * 64},
                {**self.value, "ids": self.value["ids"][:-1]},
                {**self.value, "ids": self.value["ids"] + ["short"]},
                {**self.value, "ids": [1] * 6},
                [],
            ]
        )
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(value)

    def test_duplicate_json_keys_fail(self):
        self.order.write_text('{"version":1,"version":1}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_training_order(
                self.order, self.rows, [self.train], steps=2, global_batch=3
            )


if __name__ == "__main__":
    unittest.main()
