import json
import sys
import unittest
from pathlib import Path

from jsonschema import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.training.control_plane.contracts import CONTRACT_ROOT
from src.training.model_classifier.sequence_repair.promotion_gate import (
    RUNTIME,
    decide,
)

GATE = {
    "promote_if": {
        "macro_f1_min_delta": -0.01,
        "per_language_macro_f1_min_delta": -0.03,
    }
}
DATA = [{"file": "test.jsonl", "sha256": "fixture"}]


def evaluation(macro, languages, data=DATA):
    return {
        "macro_f1": macro,
        "data_files": data,
        "breakdowns": {
            "language": {
                language: {"macro_f1": value} for language, value in languages.items()
            }
        },
    }


BASELINE = evaluation(0.80, {"en": 0.82, "fr": 0.78})
SAME = {"a": 1, "b": 2, "c": 3, "d": 4}


NEURAL = json.loads((CONTRACT_ROOT / "testdata/neural.json").read_text())


def qualification(runtime=RUNTIME, compatible=True):
    return {**NEURAL["qualification"], "runtime": runtime, "compatible": compatible}


PASSED = qualification()


class PromotionGateTests(unittest.TestCase):
    def test_promotes_within_tolerance(self):
        candidate = evaluation(0.795, {"en": 0.81, "fr": 0.76})
        receipt = decide(GATE, candidate, BASELINE, SAME, SAME, PASSED)
        self.assertEqual(receipt["decision"], "promote")
        self.assertEqual(receipt["failures"], [])
        self.assertEqual(receipt["prediction_agreement"], 1.0)

    def test_rejects_an_aggregate_regression(self):
        candidate = evaluation(0.78, {"en": 0.82, "fr": 0.78})
        receipt = decide(GATE, candidate, BASELINE, SAME, SAME, PASSED)
        self.assertEqual(receipt["decision"], "reject")
        self.assertEqual(receipt["failures"], ["macro_f1"])

    def test_rejects_one_language_hidden_by_the_aggregate(self):
        candidate = evaluation(0.80, {"en": 0.86, "fr": 0.74})
        receipt = decide(GATE, candidate, BASELINE, SAME, SAME, PASSED)
        self.assertEqual(receipt["decision"], "reject")
        self.assertEqual(receipt["failures"], ["language:fr"])

    def test_rejects_an_incompatible_qualification(self):
        receipt = decide(
            GATE, BASELINE, BASELINE, SAME, SAME, qualification(compatible=False)
        )
        self.assertEqual(receipt["decision"], "reject")
        self.assertEqual(receipt["failures"], ["qualification:compatible"])

    def test_rejects_a_qualification_on_another_runtime(self):
        other = qualification({"name": "router", "version": "1"})
        receipt = decide(GATE, BASELINE, BASELINE, SAME, SAME, other)
        self.assertEqual(receipt["decision"], "reject")
        self.assertEqual(receipt["failures"], ["qualification:runtime"])

    def test_refuses_a_file_that_is_not_a_qualification(self):
        with self.assertRaises(ValidationError):
            decide(GATE, BASELINE, BASELINE, SAME, SAME, {"checks": []})

    def test_reports_prediction_agreement(self):
        changed = {**SAME, "a": 0}
        receipt = decide(GATE, BASELINE, BASELINE, changed, SAME, PASSED)
        self.assertEqual(receipt["prediction_agreement"], 0.75)

    def test_refuses_different_data_or_rows(self):
        other = evaluation(0.80, {"en": 0.82, "fr": 0.78}, data=[{"file": "x"}])
        with self.assertRaisesRegex(ValueError, "different data"):
            decide(GATE, other, BASELINE, SAME, SAME, PASSED)
        with self.assertRaisesRegex(ValueError, "different rows"):
            decide(GATE, BASELINE, BASELINE, {"a": 1}, SAME, PASSED)


if __name__ == "__main__":
    unittest.main()
