"""Candidate KV branch tests without loading model weights or benchmark labels."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from training.model.option_isolation_cache import independent_endpoints


class FakeCache:
    def __init__(self, sums: torch.Tensor):
        self.sums = sums

    def batch_repeat_interleave(self, count: int) -> None:
        self.sums = self.sums.repeat_interleave(count, dim=0)


class FakeCausalBackbone:
    training = False

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        past_key_values: FakeCache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del return_dict, cache_position
        own_mask = attention_mask[:, -input_ids.shape[1] :]
        starting = (
            past_key_values.sums
            if past_key_values is not None
            else torch.zeros(input_ids.shape[0], device=input_ids.device)
        )
        cumulatives = (input_ids * own_mask).cumsum(dim=1) + starting[:, None]
        cache = FakeCache(cumulatives[:, -1]) if use_cache else None
        return SimpleNamespace(
            last_hidden_state=cumulatives[..., None], past_key_values=cache
        )


class CachedOptionTests(unittest.TestCase):
    def test_cache_chunks_match_independent_full_passes(self) -> None:
        backbone = FakeCausalBackbone()
        prefix = [5, 3, 7]
        tails = [[2, 4], [1], [9, 8, 6], [10, 3]]
        args = dict(
            pad_id=0,
            device=torch.device("cpu"),
            chunk_size=2,
            max_length=8,
        )
        cached = independent_endpoints(
            backbone, prefix, tails, reuse_prefix=True, **args
        )
        naive = independent_endpoints(
            backbone, prefix, tails, reuse_prefix=False, **args
        )
        self.assertTrue(torch.equal(cached, naive))
        self.assertEqual(cached[:, 0].tolist(), [21.0, 16.0, 38.0, 28.0])

    def test_reordering_only_reorders_branch_vectors(self) -> None:
        backbone = FakeCausalBackbone()
        args = dict(
            pad_id=0,
            device=torch.device("cpu"),
            chunk_size=2,
            max_length=8,
            reuse_prefix=True,
        )
        original = independent_endpoints(
            backbone, [5, 3], [[1], [7, 2], [4, 3]], **args
        )
        reordered = independent_endpoints(
            backbone, [5, 3], [[4, 3], [1], [7, 2]], **args
        )
        self.assertTrue(torch.equal(reordered, original[[2, 0, 1]]))

    def test_overlength_branch_fails_instead_of_truncating(self) -> None:
        with self.assertRaisesRegex(ValueError, "native length cap"):
            independent_endpoints(
                FakeCausalBackbone(),
                [1, 2, 3],
                [[4, 5]],
                pad_id=0,
                device=torch.device("cpu"),
                chunk_size=1,
                max_length=4,
                reuse_prefix=True,
            )


if __name__ == "__main__":
    unittest.main()
