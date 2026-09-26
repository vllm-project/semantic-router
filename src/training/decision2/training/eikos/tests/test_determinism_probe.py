import unittest

from training.eikos.determinism_probe import rank_reference_disagreements


def prediction(choice: str, probability: float) -> dict:
    return {
        "answers": {
            "q": {
                "type": "choice",
                "choice": choice,
                "probabilities": {"a": probability, "b": 1 - probability},
            }
        }
    }


class DisagreementSelectionTest(unittest.TestCase):
    def test_categorical_change_precedes_larger_probability_drift(self):
        left = {
            "stable": prediction("a", 0.95),
            "changed": prediction("a", 0.51),
            "stable_drift": prediction("a", 0.99),
        }
        right = {
            "stable": prediction("a", 0.95),
            "changed": prediction("b", 0.49),
            "stable_drift": prediction("a", 0.6),
        }
        ranked = rank_reference_disagreements(left, right, set(left))
        self.assertEqual(
            [row[2] for row in ranked], ["changed", "stable_drift", "stable"]
        )
        self.assertEqual([row[0] for row in ranked], [True, False, False])

    def test_rejects_reference_panel_mismatch(self):
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            rank_reference_disagreements(
                {"a": prediction("a", 0.9)},
                {"b": prediction("a", 0.9)},
                {"a", "b"},
            )


if __name__ == "__main__":
    unittest.main()
