"""Test block addressing used by source export and target injection."""

from __future__ import annotations

import unittest

import torch

from src.kv_connector.paged_cache import extract_prefix, inject_prefix


class PagedCacheTests(unittest.TestCase):
    def test_noncontiguous_blocks_round_trip_without_touching_other_blocks(
        self,
    ) -> None:
        cache = torch.zeros((5, 2, 4, 4), dtype=torch.float32)
        keys = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
        values = -keys
        inject_prefix(cache, [3, 1], keys, values)
        restored_k, restored_v = extract_prefix(cache, [3, 1], 6, heads=2, head_dim=2)
        torch.testing.assert_close(restored_k, keys)
        torch.testing.assert_close(restored_v, values)
        self.assertEqual(cache[0].count_nonzero().item(), 0)
        self.assertEqual(cache[2].count_nonzero().item(), 0)
        self.assertEqual(cache[4].count_nonzero().item(), 0)
        self.assertEqual(cache[1, :, 2:].count_nonzero().item(), 0)

    def test_rejects_duplicate_or_short_block_map(self) -> None:
        cache = torch.zeros((3, 2, 4, 4))
        keys = torch.zeros((6, 2, 2))
        with self.assertRaisesRegex(ValueError, "not enough blocks"):
            inject_prefix(cache, [1], keys, keys)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inject_prefix(cache, [1, 1], keys, keys)


if __name__ == "__main__":
    unittest.main()
