"""Explicit ST reader precision and native adaptive-loss normalization."""

import unittest
from types import SimpleNamespace

try:
    import torch
    from sentence_transformers.models import Pooling
    from transformers import ModernBertConfig

    from src.training.model_embeddings.mmbert_32k.embedder_training import _build_loss
    from src.training.model_embeddings.mmbert_32k.representation_contract import (
        set_vela_representation_contract,
    )
    from src.training.model_embeddings.mmbert_32k.representation_sentence_transformers import (
        configure_sentence_transformer_representation,
    )
except ImportError:
    torch = None


@unittest.skipIf(
    torch is None, "requires torch, transformers and sentence-transformers"
)
class SentenceTransformerContractTest(unittest.TestCase):
    def model(self, explicit=True, **kwargs):
        config = ModernBertConfig()
        if explicit:
            set_vela_representation_contract(config, "embedding")
        return [
            SimpleNamespace(auto_model=SimpleNamespace(config=config)),
            Pooling(2, **kwargs),
        ]

    def test_long_fp16_mean_accumulates_in_fp32_and_rejects_empty_rows(self):
        model = self.model()
        configure_sentence_transformer_representation(model)
        configure_sentence_transformer_representation(model)
        self.assertEqual(len(model[1]._forward_pre_hooks), 1)
        features = {
            "token_embeddings": torch.full((1, 32768, 2), 3, dtype=torch.float16),
            "attention_mask": torch.ones((1, 32768), dtype=torch.int64),
        }
        output = model[1](features)["sentence_embedding"]
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, torch.full((1, 2), 3.0))
        features["attention_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "all-padding"):
            model[1](features)

    def test_legacy_reader_unchanged_and_unsupported_explicit_pool_rejected(self):
        legacy = self.model(explicit=False)
        self.assertIsNone(configure_sentence_transformer_representation(legacy))
        self.assertEqual(len(legacy[1]._forward_pre_hooks), 0)
        mixed = self.model(pooling_mode_max_tokens=True)
        with self.assertRaisesRegex(ValueError, "mean only"):
            configure_sentence_transformer_representation(mixed)

    def test_norm_all_metadata_cannot_silently_train_raw_adaptive_exits(self):
        model = self.model()
        model[0].auto_model.config.representation_contract[
            "intermediate_normalization"
        ] = "final_norm"
        args = SimpleNamespace(use_adaptive_layer=True, use_2d_matryoshka=False)
        with self.assertRaisesRegex(ValueError, "raw intermediate"):
            _build_loss(args, model)


if __name__ == "__main__":
    unittest.main()
