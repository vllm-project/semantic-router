import unittest

from training.eikos.calibrate import fit, reweight, summarize


class TestCalibration(unittest.TestCase):
    def test_typewise_fit_softens_overconfident_errors(self):
        rows = [
            {
                "task_type": "noul",
                "probabilities": [0.99, 0.01],
                "label": 0 if i < 70 else 1,
            }
            for i in range(100)
        ]
        temperature = fit(rows)
        self.assertGreater(temperature, 1)
        raw = summarize(rows, {"choice": 1, "noul": 1, "score": 1})
        calibrated = summarize(rows, {"choice": 1, "noul": temperature, "score": 1})
        self.assertLess(calibrated["nll"], raw["nll"])

    def test_probability_reweighting(self):
        self.assertAlmostEqual(reweight([0.2, 0.8], 1)[0], 0.2)
        self.assertLess(reweight([0.2, 0.8], 2)[1], 0.8)


if __name__ == "__main__":
    unittest.main()
