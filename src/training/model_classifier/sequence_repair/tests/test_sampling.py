import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.training.model_classifier.sequence_repair.train import (
    sample_length_balanced,
    sample_source_balanced,
    sampling_exposure,
    source_sampling_pools,
    source_sampling_weights,
)


class SourceSamplingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"source": "natural", "label": "a", "length_bucket": 10},
            {"source": "natural", "label": "b", "length_bucket": 100},
            {"source": "authored", "label": "a", "length_bucket": 10},
        ]

    def test_missing_weights_preserve_legacy_rng_sequence(self):
        pools = source_sampling_pools(self.rows, True)
        actual, legacy = random.Random(81), random.Random(81)
        weights = source_sampling_weights(self.rows, None)
        for _ in range(500):
            self.assertEqual(
                sample_source_balanced(pools, actual, True, weights),
                sample_length_balanced(legacy.choice(pools), legacy, True),
            )
        self.assertEqual(actual.getstate(), legacy.getstate())

    def test_weight_order_uses_source_occurrence_and_controls_frequency(self):
        weights = source_sampling_weights(self.rows, '{"authored": 1, "natural": 9}')
        self.assertEqual(weights, [0.9, 0.1])
        pools = source_sampling_pools(self.rows, True)
        rng = random.Random(81)
        drawn = Counter(
            sample_source_balanced(pools, rng, True, weights) for _ in range(10000)
        )
        self.assertGreater(drawn[2], 900)
        self.assertLess(drawn[2], 1100)
        self.assertGreater(drawn[0], 4000)
        self.assertGreater(drawn[1], 4000)
        exposure = sampling_exposure(self.rows, drawn)
        self.assertEqual(exposure["natural"]["unique_rows"], 2)
        self.assertEqual(exposure["natural"]["draws"], drawn[0] + drawn[1])
        self.assertEqual(exposure["natural"]["label_draws"]["b"], drawn[1])

    def test_invalid_or_incomplete_source_weights_are_rejected(self):
        for specification in [
            {},
            [],
            {"natural": 1},
            {"natural": 1, "authored": 1, "unknown": 1},
            *(
                {"natural": value, "authored": 1}
                for value in [True, "1", 0, -1, float("nan"), float("inf")]
            ),
            {"natural": 1e308, "authored": 1e308},
        ]:
            with self.subTest(specification=specification), self.assertRaises(
                ValueError
            ):
                source_sampling_weights(self.rows, json.dumps(specification))
        with self.assertRaises(ValueError):
            source_sampling_weights([{"label": "a"}], '{"natural": 1}')

    def test_unique_coverage_keeps_unseen_rows_visible(self):
        exposure = sampling_exposure(self.rows, Counter({0: 100, 2: 5}))
        self.assertEqual(exposure["natural"]["rows"], 2)
        self.assertEqual(exposure["natural"]["unique_rows"], 1)
        self.assertEqual(exposure["natural"]["label_draws"]["b"], 0)


if __name__ == "__main__":
    unittest.main()
