import unittest

from multilingual.public_typed_compare import _section


class PublicTypedCompareTest(unittest.TestCase):
    def test_paired_cluster_delta_and_reproducibility(self):
        # The two wins in g1 must stay together when sampling groups.
        pairs = [("g1", 1, 0), ("g1", 1, 0), ("g2", 0, 1)]
        first = _section(pairs, seed=17, resamples=400)
        second = _section(pairs, seed=17, resamples=400)
        self.assertEqual(first, second)
        self.assertEqual(first["questions"], 3)
        self.assertEqual(first["observable_groups"], 2)
        self.assertEqual(first["candidate_only_correct"], 2)
        self.assertEqual(first["baseline_only_correct"], 1)
        self.assertAlmostEqual(first["delta_accuracy"], 1 / 3)
        self.assertEqual(first["cluster_bootstrap_95_percentile"], [-1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
