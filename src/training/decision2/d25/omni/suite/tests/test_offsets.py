"""Offset fits on synthetic reference models: exact recovery, leave-one-out residuals, public LOO."""

import math
import random
import unittest

from d25.omni.suite import offsets, score


class FitBenchmarkTest(unittest.TestCase):
    def test_linear_fit_recovers_the_map_and_loo_is_zero_without_noise(self):
        pairs = {
            f"m{i}": (x, 2.0 + 0.9 * x) for i, x in enumerate([10, 30, 45, 60, 80])
        }
        fit = offsets.fit_benchmark(pairs, "linear")
        self.assertAlmostEqual(fit["a"], 2.0)
        self.assertAlmostEqual(fit["b"], 0.9)
        self.assertAlmostEqual(fit["loo_rmse"], 0.0, places=9)
        self.assertAlmostEqual(fit["r2"], 1.0)

    def test_offset_and_identity_modes(self):
        pairs = {"a": (50.0, 53.0), "b": (60.0, 63.0), "c": (70.0, 73.5)}
        fit = offsets.fit_benchmark(pairs, "offset")
        self.assertEqual(fit["b"], 1.0)
        self.assertAlmostEqual(fit["a"], (3 + 3 + 3.5) / 3)
        self.assertAlmostEqual(fit["loo_residuals"]["c"], 3.5 - 3.0)
        ident = offsets.fit_benchmark(pairs, "identity")
        self.assertEqual((ident["a"], ident["b"]), (0.0, 1.0))
        self.assertAlmostEqual(ident["loo_residuals"]["a"], 3.0)

    def test_loo_residual_is_out_of_sample(self):
        rng = random.Random(1)
        pairs = {
            f"m{i}": (x, 1.0 + 1.1 * x + rng.gauss(0, 1.0))
            for i, x in enumerate(range(20, 90, 7))
        }
        fit = offsets.fit_benchmark(pairs, "linear")
        self.assertGreater(fit["loo_rmse"], fit["rmse"])
        m = "m3"
        rest = {k: v for k, v in pairs.items() if k != m}
        sub = offsets.fit_benchmark(rest, "linear")
        self.assertAlmostEqual(
            fit["loo_residuals"][m], pairs[m][1] - (sub["a"] + sub["b"] * pairs[m][0])
        )

    def test_too_few_points(self):
        with self.assertRaises(ValueError):
            offsets.fit_benchmark({"a": (1, 2), "b": (2, 3)}, "linear")


class FitAllTest(unittest.TestCase):
    def test_public_loo_error_reflects_benchmark_noise(self):
        rng = random.Random(5)
        truth = {
            b: (rng.uniform(-3, 3), rng.uniform(0.9, 1.1)) for b in score.BENCHMARKS
        }
        models = [f"m{i}" for i in range(9)]
        level = {m: rng.uniform(30, 80) for m in models}
        exact, noisy = {}, {}
        for b, (a, k) in truth.items():
            exact[b], noisy[b] = {}, {}
            for m in models:
                local = level[m] + rng.uniform(-15, 15)
                exact[b][m] = (local, a + k * local)
                noisy[b][m] = (local, a + k * local + rng.gauss(0, 2.0))
        clean = offsets.fit_all(exact)
        self.assertAlmostEqual(clean["public_loo_rmse"], 0.0, places=6)
        report = offsets.fit_all(noisy, {"R-Bench-M": "offset"})
        self.assertGreater(report["public_loo_rmse"], 0.05)
        self.assertLess(report["public_loo_rmse"], 2.0)
        self.assertEqual(report["benchmarks"]["R-Bench-M"]["mode"], "offset")
        new = {b: 50.0 for b in score.BENCHMARKS}
        corrected = offsets.apply(report["benchmarks"], new)
        fit = report["benchmarks"]["CV-Bench"]
        self.assertAlmostEqual(corrected["CV-Bench"], fit["a"] + fit["b"] * 50.0)
        self.assertTrue(math.isfinite(score.public_score(corrected)))

    def test_raw_public_error_without_correction(self):
        pairs = {
            b: {"m0": (60.0, 63.0), "m1": (40.0, 43.0), "m2": (50.0, 53.0)}
            for b in score.BENCHMARKS
        }
        report = offsets.fit_all(pairs, {b: "offset" for b in score.BENCHMARKS})
        for v in report["public"].values():
            self.assertAlmostEqual(v["raw_error"], -3.0)
            self.assertAlmostEqual(v["loo_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
