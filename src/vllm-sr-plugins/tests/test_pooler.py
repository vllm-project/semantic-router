from __future__ import annotations

import unittest

from support import has

HAVE_VLLM = has("vllm") and has("torch")


class Cursor:
    def __init__(self, torch, scheduled, seq_lens, prompt_lens):
        self.num_scheduled_tokens_cpu = torch.tensor(scheduled)
        self.seq_lens_cpu = torch.tensor(seq_lens)
        self.prompt_lens_cpu = torch.tensor(prompt_lens)


class Metadata:
    def __init__(self, cursor, params, states):
        self.cursor = cursor
        self.pooling_params = params
        self.pooling_states = states

    def get_pooling_cursor(self):
        return self.cursor


@unittest.skipUnless(HAVE_VLLM, "needs vLLM")
class CandidatePoolerTest(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from training.model.decision_model import CandidateHead as TrainingHead
        from vllm import PoolingParams
        from vllm.v1.pool.metadata import PoolingStates

        from vllm_sr_plugins.decision2.gather import POSITIONS_KEY
        from vllm_sr_plugins.decision2.head import CandidateHead
        from vllm_sr_plugins.decision2.pooler import CandidatePooler

        torch.manual_seed(1)
        self.torch = torch
        self.reference = TrainingHead(48, 16)
        for parameter in self.reference.parameters():
            torch.nn.init.normal_(parameter, std=0.2)
        head = CandidateHead(48, 16)
        head.load_state_dict(self.reference.state_dict())
        self.pooler = CandidatePooler(head)
        self.states = PoolingStates

        def params(candidates, query):
            return PoolingParams(
                task="plugin",
                extra_kwargs={
                    POSITIONS_KEY: {
                        "candidate_positions": candidates,
                        "query_position": query,
                    }
                },
            )

        self.params = params

    def expected(self, hidden, candidates, query):
        return self.reference(hidden[candidates][None], hidden[query][None])[0]

    def test_supports_only_the_plugin_task(self) -> None:
        self.assertEqual(self.pooler.get_supported_tasks(), {"plugin"})
        self.assertEqual(list(self.pooler.parameters()), [])

    def test_chunked_and_whole_prompts_match_the_training_head(self) -> None:
        torch = self.torch
        a, b = torch.randn(10, 48), torch.randn(12, 48)
        pa, pb = self.params([3, 6], 9), self.params([2, 5, 8], 11)
        sa, sb = self.states(), self.states()
        step1 = self.pooler(
            torch.cat([a, b[:7]]),
            Metadata(Cursor(torch, [10, 7], [10, 7], [10, 12]), [pa, pb], [sa, sb]),
        )
        self.assertIsNone(step1[1])
        torch.testing.assert_close(
            step1[0], self.expected(a, [3, 6], 9), rtol=0, atol=1e-6
        )
        step2 = self.pooler(b[7:], Metadata(Cursor(torch, [5], [12], [12]), [pb], [sb]))
        torch.testing.assert_close(
            step2[0], self.expected(b, [2, 5, 8], 11), rtol=0, atol=1e-6
        )
        self.assertEqual((sa.hidden_states_cache, sb.hidden_states_cache), ([], []))

    def test_bf16_hidden_states_give_fp32_logits(self) -> None:
        torch = self.torch
        a = torch.randn(10, 48).bfloat16()
        out = self.pooler(
            a,
            Metadata(
                Cursor(torch, [10], [10], [10]),
                [self.params([3, 6], 9)],
                [self.states()],
            ),
        )
        self.assertEqual(out[0].dtype, torch.float32)
        torch.testing.assert_close(
            out[0], self.expected(a, [3, 6], 9), rtol=0, atol=1e-6
        )

    def test_request_without_positions_gets_nan_not_an_error(self) -> None:
        from vllm import PoolingParams

        torch = self.torch
        out = self.pooler(
            torch.randn(6, 48),
            Metadata(
                Cursor(torch, [6], [6], [6]),
                [PoolingParams(task="plugin")],
                [self.states()],
            ),
        )
        self.assertTrue(torch.isnan(out[0]).all())


if __name__ == "__main__":
    unittest.main()
