"""Zero-step comparison keeps option and native-input parity strict."""

from __future__ import annotations

import unittest

from research.autojev_kl06_zero_compare import compare


def item(probability: float) -> dict:
    return {
        "id": "one",
        "prompt_sha256": "prompt",
        "token_ids_sha256": "tokens",
        "prediction_key": "a" if probability > 0.5 else "b",
        "answer": {
            "type": "choice",
            "probabilities": {"a": probability, "b": 1 - probability},
        },
    }


class ZeroComparisonTests(unittest.TestCase):
    def test_equal_native_outputs_pass(self) -> None:
        result = compare([item(0.8)], [item(0.8)])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["max_probability_drift"], 0)

    def test_input_and_category_changes_fail(self) -> None:
        self.assertEqual(compare([item(0.8)], [item(0.2)])["status"], "HOLD")
        changed = item(0.8)
        changed["token_ids_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "request differs"):
            compare([item(0.8)], [changed])


if __name__ == "__main__":
    unittest.main()
