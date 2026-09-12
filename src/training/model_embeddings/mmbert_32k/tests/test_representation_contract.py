"""Artifact metadata and actual ModernBERT exit contract regression tests."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from src.training.model_embeddings.mmbert_32k.representation_contract import (
    read_representation_contract,
    set_vela_representation_contract,
    vela_representation_contract,
)

try:
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    from src.training.model_embeddings.mmbert_32k.representation_outputs import (
        PhysicalPrefixEncoder,
        masked_mean,
        select_hidden_state,
        truncate_and_normalize,
    )
    from src.training.model_embeddings.mmbert_32k.reranker_model import (
        Matryoshka2DReranker,
    )
except ImportError:
    torch = None


class RepresentationMetadataTest(unittest.TestCase):
    def test_absent_metadata_is_legacy(self):
        self.assertIsNone(read_representation_contract({}, "embedding"))
        self.assertIsNone(read_representation_contract(SimpleNamespace(), "reranker"))

    def test_complete_task_metadata_roundtrips(self):
        for task in ("embedding", "reranker"):
            config = SimpleNamespace()
            expected = set_vela_representation_contract(config, task)
            self.assertEqual(read_representation_contract(config, task), expected)

    def test_unknown_missing_and_wrong_type_fields_are_rejected(self):
        original = vela_representation_contract("embedding")
        for invalid in (
            {**original, "version": True},
            {**original, "version": 2},
            {**original, "extra": "ignored"},
            {**original, "intermediate_normalization": "typo"},
            {key: value for key, value in original.items() if key != "pooling"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                read_representation_contract(
                    {"representation_contract": invalid}, "embedding"
                )

    def test_one_tasks_metadata_cannot_be_used_as_another(self):
        with self.assertRaises(ValueError):
            read_representation_contract(
                {"representation_contract": vela_representation_contract("embedding")},
                "reranker",
            )


@unittest.skipIf(torch is None, "requires torch and transformers training dependencies")
class NativeRepresentationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = ModernBertConfig(
            vocab_size=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=3,
            num_attention_heads=2,
            max_position_embeddings=128,
            global_attn_every_n_layers=2,
            local_attention=4,
            reference_compile=False,
            attention_dropout=0.0,
        )
        self.encoder = ModernBertModel(config).eval()
        with torch.no_grad():
            self.encoder.final_norm.weight.fill_(2.0)
        self.ids = torch.tensor([[1, 2, 3, 4], [2, 3, 0, 0]])
        self.mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    def test_embedding_prefix_matches_raw_early_and_normalized_full(self):
        contract = set_vela_representation_contract(self.encoder.config, "embedding")
        with torch.inference_mode():
            output = self.encoder(
                self.ids, attention_mask=self.mask, output_hidden_states=True
            )
            for layer in (1, 2, 3):
                expected = (
                    output.last_hidden_state
                    if layer == self.encoder.config.num_hidden_layers
                    else output.hidden_states[layer]
                )
                selected = select_hidden_state(
                    self.encoder, output, layer, contract, task="embedding"
                )
                prefix = PhysicalPrefixEncoder(
                    copy.deepcopy(self.encoder), layer, "embedding"
                )
                torch.testing.assert_close(selected, expected)
                torch.testing.assert_close(prefix(self.ids, self.mask), expected)
            self.assertFalse(
                torch.allclose(output.hidden_states[-1], output.last_hidden_state)
            )

    def test_reranker_normalizes_every_exit_exactly_once(self):
        contract = set_vela_representation_contract(self.encoder.config, "reranker")
        with torch.inference_mode():
            output = self.encoder(
                self.ids, attention_mask=self.mask, output_hidden_states=True
            )
            for layer in (1, 2, 3):
                expected = self.encoder.final_norm(output.hidden_states[layer])
                prefix = PhysicalPrefixEncoder(
                    copy.deepcopy(self.encoder), layer, "reranker"
                )
                torch.testing.assert_close(
                    select_hidden_state(
                        self.encoder, output, layer, contract, task="reranker"
                    ),
                    expected,
                )
                torch.testing.assert_close(prefix(self.ids, self.mask), expected)

    def test_pooling_uses_fp32_and_dimension_specific_normalization(self):
        hidden = torch.full((1, 32768, 3), 4.0, dtype=torch.float16)
        mask = torch.ones((1, 32768), dtype=torch.long)
        pooled = masked_mean(hidden, mask)
        self.assertEqual(pooled.dtype, torch.float32)
        torch.testing.assert_close(pooled, torch.full((1, 3), 4.0))
        torch.testing.assert_close(
            truncate_and_normalize(pooled, 2), torch.full((1, 2), 2**-0.5)
        )
        with self.assertRaisesRegex(ValueError, "all-padding"):
            masked_mean(hidden, torch.zeros_like(mask))

    def test_export_requires_metadata_and_valid_layer(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            PhysicalPrefixEncoder(self.encoder, 1, "embedding")
        set_vela_representation_contract(self.encoder.config, "embedding")
        for layer in (0, 4):
            with self.assertRaises(ValueError):
                PhysicalPrefixEncoder(copy.deepcopy(self.encoder), layer, "embedding")

    def test_reranker_legacy_and_explicit_contract_have_distinct_full_heads(self):
        model = Matryoshka2DReranker.__new__(Matryoshka2DReranker)
        torch.nn.Module.__init__(model)
        model.encoder = self.encoder
        model.final_norm = self.encoder.final_norm
        model.num_layers = self.encoder.config.num_hidden_layers
        model.layer_indices = [1, 2, 3]
        model.dim_indices = [8]
        model.pooling_strategy = "cls"
        model.representation_contract = None
        model.layer_heads = torch.nn.ModuleDict(
            {
                str(layer): torch.nn.ModuleDict({"8": torch.nn.Linear(8, 1)})
                for layer in model.layer_indices
            }
        )
        with torch.inference_mode():
            legacy = model(self.ids, self.mask, return_all_scores=True)["all_scores"]
            output = self.encoder(
                self.ids, attention_mask=self.mask, output_hidden_states=True
            )
            expected = model.layer_heads["3"]["8"](
                output.hidden_states[-1][:, 0]
            ).squeeze(-1)
            torch.testing.assert_close(legacy["layer_3_dim_8"], expected)
            model.representation_contract = set_vela_representation_contract(
                self.encoder.config, "reranker"
            )
            vela = model(self.ids, self.mask, return_all_scores=True)["all_scores"]
            expected = model.layer_heads["3"]["8"](
                output.last_hidden_state[:, 0]
            ).squeeze(-1)
            torch.testing.assert_close(vela["layer_3_dim_8"], expected)
            torch.testing.assert_close(legacy["layer_1_dim_8"], vela["layer_1_dim_8"])
            self.assertFalse(
                torch.allclose(legacy["layer_3_dim_8"], vela["layer_3_dim_8"])
            )


if __name__ == "__main__":
    unittest.main()
