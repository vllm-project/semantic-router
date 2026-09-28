"""Tests for full and partial gradient-accumulation windows."""

from __future__ import annotations

import unittest

from src.training.model_embeddings.mmbert_32k.foundation_optimization import (
    accumulation_window_size,
    optimizer_steps_per_epoch,
    should_optimizer_step,
)


class FoundationOptimizationTest(unittest.TestCase):
    def test_seventeen_batches_with_sixteen_accumulation_take_two_steps(self) -> None:
        self.assertEqual(optimizer_steps_per_epoch(17, 16), 2)
        self.assertEqual(accumulation_window_size(0, 17, 16), 16)
        self.assertEqual(accumulation_window_size(15, 17, 16), 16)
        self.assertEqual(accumulation_window_size(16, 17, 16), 1)
        self.assertTrue(should_optimizer_step(15, 17, 16))
        self.assertTrue(should_optimizer_step(16, 17, 16))

    def test_exact_sixteen_batch_window_takes_one_step(self) -> None:
        self.assertEqual(optimizer_steps_per_epoch(16, 16), 1)
        self.assertFalse(should_optimizer_step(14, 16, 16))
        self.assertTrue(should_optimizer_step(15, 16, 16))
        self.assertEqual(accumulation_window_size(15, 16, 16), 16)


if __name__ == "__main__":
    unittest.main()
