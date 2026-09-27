"""CPU-only contract tests for the bounded Sol 2B zero-step gate."""

from __future__ import annotations

import copy
import unittest

from training.model.zero_step_repeat import compare_historical, compare_repeats


def records() -> tuple[list[dict], dict[str, dict]]:
    actual = []
    control = {}
    for number in range(32):
        identifier = f"select-{number}"
        row = {
            "id": identifier,
            "prompt_sha256": f"prompt-{number}",
            "token_ids_sha256": f"tokens-{number}",
            "task_type": "choice",
            "prediction_key": "a",
            "probabilities": {"a": 0.75, "b": 0.25},
        }
        actual.append(row)
        control[identifier] = {
            key: row[key]
            for key in (
                "id",
                "prompt_sha256",
                "token_ids_sha256",
                "task_type",
                "prediction_key",
            )
        }
        control[identifier]["answer"] = {"probabilities": dict(row["probabilities"])}
    return actual, control


class ZeroStepRepeatTests(unittest.TestCase):
    def test_historical_pass_and_numeric_failure(self):
        actual, control = records()
        self.assertEqual(compare_historical(actual, control)["status"], "PASS")
        actual[0]["probabilities"]["a"] += 2e-4
        self.assertEqual(compare_historical(actual, control)["status"], "FAIL")

    def test_historical_identity_and_category_fail(self):
        actual, control = records()
        actual[0]["prediction_key"] = "b"
        self.assertEqual(compare_historical(actual, control)["categorical_changes"], 1)
        actual[0]["prompt_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "identity"):
            compare_historical(actual, control)

    def test_repeats_require_same_lock_model_backend_and_inputs(self):
        actual, _ = records()
        first = {
            "lock_sha256": "lock",
            "source_model_sha256": "model",
            "backend": {"gated_delta_backend": "reference"},
            "predictions": actual,
        }
        second = copy.deepcopy(first)
        self.assertEqual(compare_repeats(first, second)["status"], "PASS")
        second["predictions"][0]["probabilities"]["a"] += 2e-6
        self.assertEqual(compare_repeats(first, second)["status"], "FAIL")
        second = copy.deepcopy(first)
        second["lock_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "locks"):
            compare_repeats(first, second)


if __name__ == "__main__":
    unittest.main()
