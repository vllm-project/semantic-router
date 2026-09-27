"""Exercise strict native-input and probability comparison for the 4B gate."""

import unittest

from research.nox4b_compare_paired_zero_step import CELLS, compare


def row(probability: float) -> dict:
    return {
        "id": "example",
        "task_type": "noul",
        "prompt_sha256": "p",
        "token_ids_sha256": "t",
        "answer": {"noul": probability},
        "prediction_key": "true",
    }


class PairedZeroStepTest(unittest.TestCase):
    def test_pass_and_probability_stop(self) -> None:
        rows = {cell: [row(0.8)] for cell in CELLS}
        self.assertEqual(compare(rows)["status"], "PASS")
        rows["treatment_b"] = [row(0.79)]
        self.assertEqual(compare(rows)["status"], "STOP")

    def test_native_input_mismatch_stops(self) -> None:
        rows = {cell: [row(0.8)] for cell in CELLS}
        rows["control_b"][0]["token_ids_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Native input"):
            compare(rows)


if __name__ == "__main__":
    unittest.main()
