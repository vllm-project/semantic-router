import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.training.model_classifier.sequence_repair.promotion_gate import (
    RECEIPT_SCHEMA,
    REQUIRED_CHECKS,
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


def conformance(passed, failed=()):
    return {
        "schema_version": RECEIPT_SCHEMA,
        "checks": [{"name": name, "passed": True} for name in passed]
        + [{"name": name, "passed": False} for name in failed],
    }


PASSED = conformance(REQUIRED_CHECKS)


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

    def test_rejects_a_failed_runtime_check(self):
        failed = conformance(REQUIRED_CHECKS - {"label_parity"}, ["label_parity"])
        receipt = decide(GATE, BASELINE, BASELINE, SAME, SAME, failed)
        self.assertEqual(receipt["decision"], "reject")
        self.assertEqual(receipt["failures"], ["conformance:label_parity"])

    def test_rejects_missing_runtime_checks(self):
        receipt = decide(GATE, BASELINE, BASELINE, SAME, SAME, conformance([]))
        self.assertEqual(receipt["decision"], "reject")
        self.assertEqual(
            receipt["failures"],
            [f"conformance:{name}" for name in sorted(REQUIRED_CHECKS)],
        )
        partial = conformance(["label_parity"])
        receipt = decide(GATE, BASELINE, BASELINE, SAME, SAME, partial)
        self.assertEqual(receipt["decision"], "reject")
        self.assertEqual(
            receipt["failures"],
            [
                "conformance:deadline_behavior",
                "conformance:input_bounds",
                "conformance:unavailable_behavior",
            ],
        )

    def test_refuses_a_conformance_file_that_is_not_a_receipt(self):
        with self.assertRaisesRegex(ValueError, "not a compatibility receipt"):
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
