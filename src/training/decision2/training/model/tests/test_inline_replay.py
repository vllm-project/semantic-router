import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.model.inline_replay import SCHEMA, attach_inline_teacher, roster_sha256
from training.model.inline_teacher import source_pairs_for_selection
from training.model.tests.test_data import row


class InlineReplayTest(unittest.TestCase):
    def test_parity_preserves_original_select_batch_mates(self):
        rows = [{"id": f"r{index}"} for index in range(5)]
        pairs = source_pairs_for_selection(rows, {"r0", "r3", "r4"})
        self.assertEqual(
            [[row["id"] for row in pair] for pair in pairs],
            [["r0", "r1"], ["r2", "r3"], ["r4"]],
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "teacher.json"
        self.train = [row("r1", group="g1"), row("r2", group="g2")]
        self.source = {"backbone/model.safetensors": "a" * 64}

    def artifact(self):
        selected = self.train[:1]
        return {
            "schema_version": SCHEMA,
            "train_sha256": "b" * 64,
            "source_files_sha256": self.source,
            "source_merged_model_sha256": "e" * 64,
            "materialization_receipt_sha256": "f" * 64,
            "roster_sha256": roster_sha256(selected),
            "zero_step_parity": {
                "status": "PASS",
                "count": 32,
                "max_absolute_probability_drift": 0.0,
                "control_baseline_sha256": "c" * 64,
                "parity_roster_sha256": "d" * 64,
            },
            "rows": [
                {
                    "id": selected[0]["id"],
                    "input_sha256": selected[0]["input_sha256"],
                    "teacher_probs": {"a": 0.25, "b": 0.75},
                }
            ],
        }

    def attach(self, document):
        self.path.write_text(json.dumps(document), encoding="utf-8")
        return attach_inline_teacher(
            self.path,
            self.train,
            train_sha256="b" * 64,
            source_files_sha256=self.source,
            expected_source_model_sha256="e" * 64,
            expected_materialization_receipt_sha256="f" * 64,
            expected_roster_sha256=roster_sha256(self.train[:1]),
            expected_control_baseline_sha256="c" * 64,
            expected_parity_roster_sha256="d" * 64,
        )

    def test_exact_binding_and_in_memory_only(self):
        document = self.artifact()
        self.assertEqual(self.attach(document), 1)
        self.assertEqual(self.train[0]["teacher_probs"], {"a": 0.25, "b": 0.75})
        self.assertNotIn("teacher_probs", self.train[1])

    def test_rejects_source_train_roster_and_input_mismatch(self):
        for field, value, message in (
            ("train_sha256", "c" * 64, "TRAIN hash"),
            ("source_files_sha256", {}, "source differs"),
            ("source_merged_model_sha256", "0" * 64, "merged-model"),
            ("materialization_receipt_sha256", "1" * 64, "materialization receipt"),
            ("roster_sha256", "d" * 64, "roster hash"),
        ):
            with self.subTest(field=field):
                document = self.artifact()
                document[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.attach(document)
                self.assertNotIn("teacher_probs", self.train[0])
        document = self.artifact()
        document["rows"][0]["input_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "input hash"):
            self.attach(document)

    def test_requires_frozen_zero_step_parity(self):
        document = self.artifact()
        document["zero_step_parity"]["max_absolute_probability_drift"] = 0.01
        with self.assertRaisesRegex(ValueError, "zero-step parity"):
            self.attach(document)
        self.assertNotIn("teacher_probs", self.train[0])

    def test_rejects_duplicate_or_malformed_probabilities_atomically(self):
        document = self.artifact()
        document["rows"].append(copy.deepcopy(document["rows"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.attach(document)
        self.assertNotIn("teacher_probs", self.train[0])
        for values in ({"a": 0.4}, {"a": 0.5, "b": 0.6}, {"a": -0.1, "b": 1.1}):
            document = self.artifact()
            document["rows"][0]["teacher_probs"] = values
            with self.assertRaises(ValueError):
                self.attach(document)
            self.assertNotIn("teacher_probs", self.train[0])


if __name__ == "__main__":
    unittest.main()
