import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from research.calibration_transport import transform


class CalibrationTransportTest(unittest.TestCase):
    def test_inverse_temperature_restores_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "cal.json"
            calibration.write_text(
                json.dumps(
                    {
                        "model_sha256": "a" * 64,
                        "temperature_by_type": {"choice": 0.5, "noul": 2, "score": 1},
                    }
                )
            )
            digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "model_sha256": "a" * 64,
                        "calibration_sha256": digest,
                        "answers": {
                            "choice": {
                                "type": "choice",
                                "choice": "a",
                                "probabilities": {"a": 0.8, "b": 0.2},
                            },
                            "noul": {"type": "noul", "noul": 0.8},
                            "score": {
                                "type": "score",
                                "score": 1,
                                "probabilities": {"0": 0.2, "1": 0.8},
                            },
                        },
                    }
                )
                + "\n"
            )
            output = root / "out.jsonl"
            receipt = transform(predictions, calibration, output)
            answers = json.loads(output.read_text())["answers"]
            self.assertEqual(
                receipt["questions_by_type"], {"choice": 1, "noul": 1, "score": 1}
            )
            self.assertTrue(
                math.isclose(answers["choice"]["probabilities"]["a"], 2 / 3)
            )
            self.assertTrue(math.isclose(answers["noul"]["noul"], 16 / 17))
            self.assertEqual(answers["score"]["probabilities"], {"0": 0.2, "1": 0.8})
            self.assertEqual(answers["choice"]["choice"], "a")
            self.assertEqual(answers["score"]["score"], 0.8)
            self.assertNotIn("calibration_sha256", json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
