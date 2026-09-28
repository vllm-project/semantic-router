"""Unequal observed-label counts retain equal example weights after batching."""

import copy
import importlib.util
import random
import unittest
from collections import Counter

from src.training.model_classifier.safety_classifier.train_vela_hazard import (
    grouped_pools,
    sample_grouped,
)
from src.training.model_classifier.safety_classifier.vela_hazard import masked_loss
from src.training.model_classifier.sequence_repair.train import (
    token_budget_microbatches,
)


class HazardTokenBudgetTests(unittest.TestCase):
    def test_grouping_retains_weighted_draws_duplicates_and_rng_state(self):
        rows = [
            {"targets": [1, 0], "source": "external"},
            {"targets": [0, 1], "source": "external"},
            {"targets": [0, 0], "source": "external"},
            {"targets": [1, 0], "source": "authored"},
            {"targets": [0, 0], "source": "authored"},
        ]
        pools = grouped_pools(rows, 2, True, False)
        before, after = random.Random(19), random.Random(19)
        lengths = [3, 32, 7, 19, 5]
        token_budget = 32
        for _ in range(30):
            previous = [sample_grouped(pools, before, [0.95, 0.05]) for _ in range(8)]
            drawn = [sample_grouped(pools, after, [0.95, 0.05]) for _ in range(8)]
            groups = token_budget_microbatches(drawn, lengths, token_budget)
            self.assertEqual(before.getstate(), after.getstate())
            self.assertEqual(previous, drawn)
            self.assertEqual(Counter(previous), Counter(i for g in groups for i in g))
            self.assertTrue(
                all(max(lengths[i] for i in g) * len(g) <= token_budget for g in groups)
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "Optional numerical test requires Torch and Transformers",
    )
    def test_modernbert_masked_bce_gradient_matches_unsplit_examples(self):
        import torch  # noqa: PLC0415 - optional dependency
        from transformers import (  # noqa: PLC0415 - optional dependency
            ModernBertConfig,
            ModernBertForSequenceClassification,
        )

        torch.set_num_threads(1)
        torch.manual_seed(31)
        config = ModernBertConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=64,
            local_attention=8,
            global_attn_every_n_layers=2,
            num_labels=4,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            classifier_dropout=0.0,
            classifier_pooling="mean",
            problem_type="multi_label_classification",
            reference_compile=False,
        )
        config._attn_implementation = "sdpa"
        full = ModernBertForSequenceClassification(config).float().train()
        grouped = copy.deepcopy(full)
        lengths = [3, 12, 32, 7, 5, 19, 8, 4]
        examples = [torch.randint(3, 64, (length,)) for length in lengths]
        targets = torch.randint(0, 2, (8, 4)).float()
        observed = torch.tensor(
            [[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]] * 2
        ).float()

        def accumulate(model, indices):
            ids = torch.zeros(
                len(indices), max(lengths[i] for i in indices), dtype=torch.long
            )
            attention = torch.zeros_like(ids)
            for row, index in enumerate(indices):
                ids[row, : lengths[index]] = examples[index]
                attention[row, : lengths[index]] = 1
            logits = model(input_ids=ids, attention_mask=attention).logits
            loss = masked_loss(logits, targets[indices], observed[indices]) * (
                len(indices) / 8
            )
            loss.backward()

        indices = list(range(8))
        accumulate(full, indices)
        batches = token_budget_microbatches(indices, lengths, 32)
        self.assertGreater(len(batches), 2)
        for batch in batches:
            accumulate(grouped, batch)
        for (name, expected), (_, actual) in zip(
            full.named_parameters(), grouped.named_parameters(), strict=True
        ):
            self.assertIsNotNone(expected.grad, name)
            self.assertIsNotNone(actual.grad, name)
            torch.testing.assert_close(
                expected.grad, actual.grad, rtol=2e-5, atol=2e-6, msg=name
            )


if __name__ == "__main__":
    unittest.main()
