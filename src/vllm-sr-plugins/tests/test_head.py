from __future__ import annotations

import unittest

from support import has

HAVE_TORCH = has("torch") and has("transformers")


@unittest.skipUnless(HAVE_TORCH, "needs torch and transformers")
class CandidateHeadTest(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from training.model.decision_model import CandidateHead as TrainingHead

        from vllm_sr_plugins.decision2.head import HEAD_PARAMETERS, CandidateHead

        torch.manual_seed(0)
        self.torch = torch
        self.reference = TrainingHead(48, 16)
        for parameter in self.reference.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
        self.head = CandidateHead(48, 16)
        self.head.load_state_dict(self.reference.state_dict(), strict=True)
        self.names = HEAD_PARAMETERS

    def test_state_dict_names_match_training_head(self) -> None:
        self.assertEqual(sorted(self.head.state_dict()), sorted(self.names))
        self.assertEqual(sorted(self.reference.state_dict()), sorted(self.names))

    def test_forward_is_bit_identical(self) -> None:
        torch = self.torch
        candidates = torch.randn(3, 5, 48)
        query = torch.randn(3, 48)
        self.assertTrue(
            torch.equal(self.head(candidates, query), self.reference(candidates, query))
        )

    def test_forward_upcasts_bf16_hidden_states(self) -> None:
        torch = self.torch
        candidates = torch.randn(2, 4, 48).bfloat16()
        query = torch.randn(2, 48).bfloat16()
        out = self.head(candidates, query)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out, self.reference(candidates, query)))

    def test_flat_scoring_matches_per_request_forward(self) -> None:
        torch = self.torch
        sizes = [2, 5, 3]
        candidates = [torch.randn(k, 48) for k in sizes]
        queries = torch.randn(len(sizes), 48)
        owner = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
        flat = self.head.score_flat(torch.cat(candidates), queries, owner)
        for values, rows, query in zip(torch.split(flat, sizes), candidates, queries):
            expected = self.reference(rows[None], query[None])[0]
            torch.testing.assert_close(values, expected, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
