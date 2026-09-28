import importlib
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - local environments without torch
    torch = None

soup = importlib.import_module("v2.06b.soup")
noul = importlib.import_module("v2.06b.noul_diag")


@unittest.skipIf(torch is None, "torch is not installed")
class UniformAverageTest(unittest.TestCase):
    def test_float_tensors_are_averaged_and_dtype_kept(self):
        a = {"w": torch.tensor([1.0, 2.0]), "step": torch.tensor([3])}
        b = {"w": torch.tensor([3.0, 6.0]), "step": torch.tensor([3])}
        out = soup.uniform_average([a, b])
        self.assertTrue(torch.equal(out["w"], torch.tensor([2.0, 4.0])))
        self.assertEqual(out["w"].dtype, torch.float32)
        self.assertTrue(torch.equal(out["step"], torch.tensor([3])))

    def test_key_shape_and_integer_mismatches_are_rejected(self):
        a = {"w": torch.zeros(2)}
        with self.assertRaises(ValueError):
            soup.uniform_average([a, {"v": torch.zeros(2)}])
        with self.assertRaises(ValueError):
            soup.uniform_average([a, {"w": torch.zeros(3)}])
        with self.assertRaises(ValueError):
            soup.uniform_average([{"n": torch.tensor([1])}, {"n": torch.tensor([2])}])


class StripTest(unittest.TestCase):
    def test_only_declared_fields_are_ignored(self):
        spec = {"arm": "x-s1", "seed": "s1", "start": {"head_seed": 1, "path": "p"}}
        other = {"arm": "x-s2", "seed": "s2", "start": {"head_seed": 2, "path": "p"}}
        allowed = ["arm", "seed", "start.head_seed"]
        self.assertEqual(soup.strip(spec, allowed), soup.strip(other, allowed))
        self.assertNotEqual(
            soup.strip(spec, ["arm", "seed"]), soup.strip(other, ["arm", "seed"])
        )
        self.assertIn("head_seed", spec["start"])


class NoulDiagnosisTest(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(noul.auc([0.9, 0.8, 0.1], [True, True, False]), 1.0)
        self.assertEqual(noul.auc([0.3, 0.3], [True, False]), 0.5)
        self.assertIsNone(noul.auc([0.3], [True]))

    def test_bias_recovers_shifted_calibration(self):
        scores = [
            noul.sigmoid(z - 1.0) for z in (-2.0, -1.0, 1.0, 2.0) for _ in range(50)
        ]
        labels = [z > 0 for z in (-2.0, -1.0, 1.0, 2.0) for _ in range(50)]
        bias = noul.fit_bias(scores, labels)
        self.assertGreater(bias, 0.5)
        self.assertEqual(
            sum(
                (noul.sigmoid(noul.logit(s) + bias) > 0.5) == y
                for s, y in zip(scores, labels)
            ),
            200,
        )

    def test_threshold_prefers_half_on_ties(self):
        self.assertEqual(noul.fit_threshold([0.2, 0.8], [False, True]), 0.5)
        self.assertEqual(noul.fit_threshold([0.3, 0.4], [False, True]), 0.3)


if __name__ == "__main__":
    unittest.main()
