"""Keep native inference and exported encoder/head precision boundaries aligned."""

from __future__ import annotations

import unittest

try:
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    from src.training.model_embeddings.mmbert_32k.representation_contract import (
        set_vela_representation_contract,
    )
    from src.training.model_embeddings.mmbert_32k.representation_outputs import (
        select_hidden_state,
    )
    from src.training.model_embeddings.mmbert_32k.reranker_model import (
        Matryoshka2DReranker,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires torch and transformers training dependencies")
class RerankerPrecisionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        model = Matryoshka2DReranker.__new__(Matryoshka2DReranker)
        torch.nn.Module.__init__(model)
        model.encoder = ModernBertModel(
            ModernBertConfig(
                vocab_size=32,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=128,
                local_attention=4,
                reference_compile=False,
                attention_dropout=0.0,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        )
        model.final_norm = model.encoder.final_norm
        model.num_layers = 2
        model.layer_indices = [1, 2]
        model.dim_indices = [8]
        model.pooling_strategy = "cls"
        model.representation_contract = set_vela_representation_contract(
            model.encoder.config, "reranker"
        )
        model.layer_heads = torch.nn.ModuleDict(
            {
                str(layer): torch.nn.ModuleDict({"8": torch.nn.Linear(8, 1)})
                for layer in model.layer_indices
            }
        )
        self.model = model.eval()
        self.ids = torch.tensor([[1, 3, 4, 2], [1, 2, 0, 0]])
        self.mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    def test_inference_ignores_outer_autocast(self):
        with torch.inference_mode():
            expected = self.model(self.ids, self.mask, return_all_scores=True)[
                "all_scores"
            ]
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = self.model(self.ids, self.mask, return_all_scores=True)[
                    "all_scores"
                ]
        for key in expected:
            self.assertEqual(actual[key].dtype, torch.float32)
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_training_keeps_encoder_amp_but_head_and_gradients_are_fp32(self):
        observed = []
        linear = next(
            m for m in self.model.encoder.modules() if isinstance(m, torch.nn.Linear)
        )
        handle = linear.register_forward_hook(
            lambda _m, _i, out: observed.append(out.dtype)
        )
        self.model.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            scores = self.model(self.ids, self.mask, return_all_scores=True)[
                "all_scores"
            ]
            loss = sum(value.square().mean() for value in scores.values())
        handle.remove()
        self.assertTrue(observed and all(dtype == torch.bfloat16 for dtype in observed))
        self.assertTrue(all(value.dtype == torch.float32 for value in scores.values()))
        loss.backward()
        self.assertTrue(
            all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in self.model.parameters()
            )
        )
        self.assertEqual(
            self.model.layer_heads["2"]["8"].weight.grad.dtype, torch.float32
        )

    def test_legacy_reader_keeps_outer_autocast(self):
        self.model.representation_contract = None
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            scores = self.model(self.ids, self.mask, return_all_scores=True)[
                "all_scores"
            ]
        self.assertTrue(all(value.dtype == torch.bfloat16 for value in scores.values()))

    def test_contract_rejects_half_precision_heads(self):
        self.model.layer_heads.half()
        with self.assertRaisesRegex(ValueError, "FP32 heads"):
            self.model._score_dimensions(torch.ones(1, 8), 2, [8])

    def test_full_exit_reuses_final_state_without_another_norm_call(self):
        with torch.inference_mode():
            output = self.model.encoder(
                self.ids, attention_mask=self.mask, output_hidden_states=True
            )
            calls = []
            handle = self.model.encoder.final_norm.register_forward_hook(
                lambda *_: calls.append(1)
            )
            selected = select_hidden_state(
                self.model.encoder,
                output,
                2,
                self.model.representation_contract,
                task="reranker",
            )
            handle.remove()
        self.assertIs(selected, output.last_hidden_state)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
