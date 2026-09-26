import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from training.eikos.verify_export import (
    PUBLICATION_DOCUMENTS,
    compare_answers,
    verify_sums,
)


class PackagedAnswerComparisonTest(unittest.TestCase):
    def test_choice_and_probability_drift_are_separate(self):
        expected = {
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.5001, "b": 0.4999},
        }
        actual = {
            "type": "choice",
            "choice": "b",
            "probabilities": {"a": 0.4998, "b": 0.5002},
        }
        same, drift, pmax_drift = compare_answers(expected, actual)
        self.assertFalse(same)
        self.assertAlmostEqual(drift, 0.0003)
        self.assertAlmostEqual(pmax_drift, 0.0001)

    def test_noul_boolean_decision_is_checked(self):
        expected = {"type": "noul", "value": True, "probability": 0.501}
        actual = {"type": "noul", "value": False, "probability": 0.499}
        same, drift, pmax_drift = compare_answers(expected, actual)
        self.assertFalse(same)
        self.assertAlmostEqual(drift, 0.002)
        self.assertAlmostEqual(pmax_drift, 0)

    def test_score_uses_modal_level_not_continuous_expectation(self):
        expected = {
            "type": "score",
            "native_score": 2,
            "score": 1.51,
            "probabilities": {"1": 0.49, "2": 0.51},
        }
        actual = {
            "type": "score",
            "native_score": 2,
            "score": 1.52,
            "probabilities": {"1": 0.48, "2": 0.52},
        }
        same, drift, pmax_drift = compare_answers(expected, actual)
        self.assertTrue(same)
        self.assertAlmostEqual(drift, 0.01)
        self.assertAlmostEqual(pmax_drift, 0.01)


class FunctionalPackageRosterTest(unittest.TestCase):
    def test_publication_documents_do_not_change_model_identity(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "config.json").write_text("{}", encoding="utf-8")
            digest = sha256(b"{}").hexdigest()
            (folder / "SHA256SUMS").write_text(
                f"{digest}  config.json\n", encoding="utf-8"
            )
            model_sha = sha256((folder / "SHA256SUMS").read_bytes()).hexdigest()
            self.assertEqual(verify_sums(folder), 1)
            for name in PUBLICATION_DOCUMENTS:
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("publication", encoding="utf-8")
            self.assertEqual(verify_sums(folder), 1)
            self.assertEqual(
                sha256((folder / "SHA256SUMS").read_bytes()).hexdigest(), model_sha
            )
            (folder / "unlisted.py").write_text("pass", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set differs"):
                verify_sums(folder)


if __name__ == "__main__":
    unittest.main()
