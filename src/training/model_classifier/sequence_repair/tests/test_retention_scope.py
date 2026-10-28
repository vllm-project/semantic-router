"""A synthetic native ModernBERT update preserves all frozen tensor bytes."""

# ruff: noqa: PLC0415

import os
import unittest

from src.training.model_classifier.sequence_repair.optimization import (
    assert_frozen_parameters_preserved,
    configure_trainable_tail,
    frozen_parameter_receipt,
    optimizer_groups,
)


class RetentionScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import transformers
        except ImportError as error:
            raise unittest.SkipTest(
                "Optional native tests need Torch and Transformers"
            ) from error
        cls.torch, cls.transformers = torch, transformers
        cls.device = os.environ.get("SEQUENCE_REPAIR_TEST_DEVICE", "cpu")
        torch.set_num_threads(1)

    def model(self):
        config = self.transformers.ModernBertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            max_position_embeddings=64,
            local_attention=8,
            global_attn_every_n_layers=2,
            num_labels=2,
            classifier_pooling="cls",
            classifier_dropout=0.0,
            attention_dropout=0.0,
            embedding_dropout=0.0,
            mlp_dropout=0.0,
            reference_compile=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        config._attn_implementation = "sdpa"
        return (
            self.transformers.ModernBertForSequenceClassification(config)
            .float()
            .to(self.device)
        )

    def test_real_backward_updates_tail_and_head_preserving_frozen_parameters(self):
        torch = self.torch
        torch.manual_seed(1729)
        model = self.model()
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.parameters())
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        scope = configure_trainable_tail(model, 1)
        self.assertEqual(scope["layer_indices"], [len(model.model.layers) - 1])
        selected = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        expected = {
            name
            for name in before
            if name.startswith(("model.layers.2.", "head.", "classifier."))
        }
        self.assertEqual(selected, expected)
        receipt = frozen_parameter_receipt(model)
        self.assertIn("model.final_norm.weight", receipt)
        groups = optimizer_groups(model, 1e-3, 1e-3)
        parameters = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(
            len({id(parameter) for parameter in parameters}), len(selected)
        )
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        inputs = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]], device=self.device)
        loss = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            labels=torch.tensor([0, 1], device=self.device),
        ).loss
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.parameters()
                if not parameter.requires_grad
            )
        )
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))
        optimizer.step()
        assert_frozen_parameters_preserved(receipt, model)
        for prefix in ("model.layers.2.", "head.", "classifier."):
            self.assertTrue(
                any(
                    not torch.equal(before[name], parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith(prefix)
                )
            )
        with torch.no_grad():
            model.model.final_norm.weight.add_(0.01)
        with self.assertRaises(ValueError):
            assert_frozen_parameters_preserved(receipt, model)

    def test_invalid_scope_does_not_partially_mutate_model(self):
        model = self.model()
        for value in (0, -1, True, 1.5, len(model.model.layers) + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure_trainable_tail(model, value)
            self.assertTrue(
                all(parameter.requires_grad for parameter in model.parameters())
            )
        model.peft_config = {}
        with self.assertRaises(ValueError):
            configure_trainable_tail(model, 1)
        with self.assertRaises(ValueError):
            configure_trainable_tail(self.torch.nn.Linear(2, 2), 1)

    def test_all_blocks_still_excludes_encoder_final_norm(self):
        model = self.model()
        scope = configure_trainable_tail(model, len(model.model.layers))
        self.assertEqual(scope["layer_indices"], list(range(len(model.model.layers))))
        self.assertFalse(model.model.final_norm.weight.requires_grad)
        self.assertFalse(model.model.embeddings.tok_embeddings.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
