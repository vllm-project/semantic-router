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
        inject_prefix(cache, [3, 1], keys, values, layout="legacy")
        restored_k, restored_v = extract_prefix(
            cache, [3, 1], 6, heads=2, head_dim=2, layout="legacy"
        )
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
            inject_prefix(cache, [1], keys, keys, layout="legacy")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inject_prefix(cache, [1, 1], keys, keys, layout="legacy")

    def test_two_head_vllm_layout_requires_explicit_selection(self) -> None:
        cache = torch.zeros((2, 2, 2, 4))
        keys = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        values = keys + 100
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            inject_prefix(cache, [0], keys, values)
        inject_prefix(cache, [0], keys, values, layout="lbnhc")
        expected = torch.cat((keys[0], values[0]), dim=-1)
        torch.testing.assert_close(cache[0, :, 0], expected)
        restored_k, restored_v = extract_prefix(
            cache, [0], 2, heads=2, head_dim=2, layout="lbnhc"
        )
        torch.testing.assert_close(restored_k, keys)
        torch.testing.assert_close(restored_v, values)

    def test_vllm_lbnhc_layer_view_round_trip(self) -> None:
        cache = torch.zeros((5, 8, 4, 4), dtype=torch.bfloat16)
        keys = torch.arange(96, dtype=torch.bfloat16).reshape(6, 8, 2)
        values = -keys
        inject_prefix(cache, [3, 1], keys, values)
        restored_k, restored_v = extract_prefix(cache, [3, 1], 6, heads=8, head_dim=2)
        torch.testing.assert_close(restored_k, keys)
        torch.testing.assert_close(restored_v, values)
        self.assertEqual(cache[0].count_nonzero().item(), 0)
        self.assertEqual(cache[1, :, 2:].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
