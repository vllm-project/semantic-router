"""Contract tests for the escalation risk scaffold (#3282).

Run from the repository root:
    python -m unittest discover -s src/training/model_classifier/escalation_risk/tests -p "test_*.py"
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.training.model_classifier.escalation_risk import fixtures

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class FixtureTest(unittest.TestCase):
    def test_same_seed_gives_identical_rows(self):
        self.assertEqual(fixtures.generate(200, "s1"), fixtures.generate(200, "s1"))

    def test_different_seed_gives_different_rows(self):
        self.assertNotEqual(fixtures.generate(200, "s1"), fixtures.generate(200, "s2"))

    def test_rows_are_marked_synthetic_and_carry_no_text(self):
        for row in fixtures.generate(50, "s1"):
            self.assertTrue(row["synthetic"])
            self.assertEqual(len(row["input_digest"]), 64)  # a hash, not a prompt
            self.assertEqual(len(row["primary"]["output_digest"]), 64)

    def test_split_is_a_pure_function_of_seed_and_id(self):
        for row in fixtures.generate(200, "s1"):
            self.assertEqual(row["split"], fixtures.split_for("s1", row["id"]))

    def test_split_proportions_follow_the_weights(self):
        rows = fixtures.generate(5000, "s1")
        share = {
            n: sum(r["split"] == n for r in rows) / len(rows)
            for n, _ in fixtures.SPLITS
        }
        self.assertAlmostEqual(share["train"], 0.7, delta=0.03)
        self.assertAlmostEqual(share["calibration"], 0.1, delta=0.03)
        self.assertAlmostEqual(share["test"], 0.2, delta=0.03)

    def test_every_feature_has_an_explicit_status(self):
        allowed = {fixtures.PRESENT, fixtures.ABSENT, fixtures.NOT_APPLICABLE}
        for row in fixtures.generate(300, "s1"):
            for name, feat in row["features"].items():
                self.assertIn(feat["status"], allowed, name)
                if feat["status"] != fixtures.PRESENT:
                    self.assertIsNone(
                        feat["value"], f"{name} is missing but has a value"
                    )

    def test_missing_data_actually_occurs(self):
        rows = fixtures.generate(500, "s1")
        statuses = {r["features"]["recent_no_progress_turns"]["status"] for r in rows}
        self.assertIn(fixtures.ABSENT, statuses)
        tool_statuses = {r["features"]["tool_count"]["status"] for r in rows}
        self.assertIn(fixtures.NOT_APPLICABLE, tool_statuses)

    def test_unclear_verdicts_are_excluded_not_guessed(self):
        for verdict in ("tie", "abstain", "both_failed"):
            self.assertIsNone(fixtures.VERDICT_LABELS[verdict])
        for row in fixtures.generate(500, "s1"):
            self.assertEqual(row["label"], fixtures.VERDICT_LABELS[row["verdict"]])


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class TrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep the optional sklearn dependency lazy so fixture tests run without it.
        from src.training.model_classifier.escalation_risk import train  # noqa: PLC0415

        cls.train = train

    def run_train(self, data_path, out_path):
        argv = ["train.py", "--data", str(data_path), "--out", str(out_path)]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
            self.train.main()
        return json.loads(out_path.read_text())

    def write_data(self, directory, rows=3000, seed="s1"):
        path = Path(directory) / "data.jsonl"
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in fixtures.generate(rows, seed))
        )
        return path

    def test_missing_value_is_encoded_with_a_flag_not_as_a_plain_zero(self):
        row = fixtures.generate(1, "s1")[0]["features"]
        row["context_fill_ratio"] = {"value": None, "status": fixtures.ABSENT}
        encoded = dict(
            zip(self.train.feature_names(), self.train.encode(row), strict=True)
        )
        self.assertEqual(encoded["context_fill_ratio"], 0.0)
        self.assertEqual(encoded["context_fill_ratio:missing"], 1.0)

    def encoded(self, **overrides):
        row = fixtures.generate(1, "s1")[0]["features"]
        row.update(overrides)
        return dict(
            zip(self.train.feature_names(), self.train.encode(row), strict=True)
        )

    def test_absent_boolean_is_not_encoded_as_confirmed_false(self):
        absent = self.encoded(has_tools={"value": None, "status": fixtures.ABSENT})
        false = self.encoded(has_tools={"value": False, "status": fixtures.PRESENT})
        true = self.encoded(has_tools={"value": True, "status": fixtures.PRESENT})
        self.assertNotEqual(absent, false)
        self.assertEqual((absent["has_tools"], absent["has_tools:missing"]), (0.0, 1.0))
        self.assertEqual((false["has_tools"], false["has_tools:missing"]), (0.0, 0.0))
        self.assertEqual((true["has_tools"], true["has_tools:missing"]), (1.0, 0.0))

    def test_absent_category_sets_its_missing_flag(self):
        encoded = self.encoded(decision={"value": None, "status": fixtures.ABSENT})
        self.assertEqual(encoded["decision:missing"], 1.0)
        self.assertEqual(
            sum(v for k, v in encoded.items() if k.startswith("decision=")), 0.0
        )

    def test_every_feature_has_a_missing_indicator(self):
        names = set(self.train.feature_names())
        for key in fixtures.generate(1, "s1")[0]["features"]:
            self.assertIn(f"{key}:missing", names)

    def test_unknown_status_or_category_is_rejected(self):
        with self.assertRaises(ValueError):
            self.encoded(has_tools={"value": None, "status": "unknown"})
        with self.assertRaises(ValueError):
            self.encoded(decision={"value": "new_topic", "status": fixtures.PRESENT})

    def test_training_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            data = self.write_data(d)
            a = self.run_train(data, Path(d) / "a.json")
            b = self.run_train(data, Path(d) / "b.json")
            self.assertEqual(a["digest"], b["digest"])

    def test_artifact_is_marked_test_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            art = self.run_train(self.write_data(d), Path(d) / "a.json")
            self.assertEqual(art["qualification"], "test-evidence")
            self.assertEqual(len(art["weights"]), len(art["feature_names"]))

    def test_classifier_beats_chance_and_is_calibrated(self):
        with tempfile.TemporaryDirectory() as d:
            m = self.run_train(self.write_data(d, rows=5000), Path(d) / "a.json")[
                "test_metrics"
            ]
            self.assertGreater(m["classifier"]["auroc"], 0.65)
            self.assertLess(m["classifier"]["ece"], 0.08)
            # catches more failures than the simple low-confidence rule
            self.assertLess(
                m["classifier"]["false_negative_rate"],
                m["baseline_low_confidence"]["false_negative_rate"],
            )


if __name__ == "__main__":
    unittest.main()
